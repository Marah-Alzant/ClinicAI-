"""bot/handlers/patient.py — Patient-facing handler for booking and inquiries."""
from telegram import Update
from telegram.ext import ContextTypes
from bot.keyboards import main_menu_keyboard
from database.db import get_db
from database import crud
from fsm.patient_fsm import PatientFSM
from voice.stt import transcribe_voice
from nlp.gemini_client import gemini

_sessions: dict[int, PatientFSM] = {}


def _is_repeat_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower().strip()
    return any(token in lowered for token in ["repeat", "كرر", "تكرار", "من جديد", "من اول", "restart", "ابدأ من جديد", "بدء جديد"])


def _get_fsm(user_id: int, reset: bool = False) -> PatientFSM:
    session = _sessions.get(user_id)
    if reset or session is None:
        session = PatientFSM(user_id=user_id)
        _sessions[user_id] = session
    return session


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user is not None:
        with get_db() as db:
            crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
            crud.log_message(db, user.id, "inbound", "command", "/start")

    reply = "👋 مرحباً! أنا مساعد الحجز. أرسل اسمك وسبب زيارتك لأساعدك بالحجز."
    await update.message.reply_text(reply, reply_markup=main_menu_keyboard())

    if user is not None:
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.edited_message or update.channel_post or update.effective_message
    if message is None:
        return

    text = message.text or ""
    user = message.from_user or update.effective_user
    if user is None:
        return

    user_id = user.id
    with get_db() as db:
        crud.get_or_create_conversation(db, user_id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user_id, "inbound", "text", text)

    if text.strip() == "/start" or _is_repeat_request(text):
        await handle_start(update, context)
        return

    fsm = _get_fsm(user_id)
    reply, keyboard = await fsm.handle(text)
    # If FSM couldn't understand, ask Gemini (context-aware) as a fallback
    if reply.strip() == "عفواً، ما فهمت. ممكن تعيد؟" and gemini._available:
        try:
            g_reply = await gemini.build_response(fsm.state.name, {**fsm.data, "last_user_message": text})
            if g_reply:
                reply = g_reply
        except Exception:
            pass

    await message.reply_text(reply, reply_markup=keyboard)

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.effective_message
    if message is None:
        return
    user = update.effective_user
    if user is None:
        return

    # Download voice file and transcribe
    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        ogg_bytes = await voice_file.download_as_bytearray()
        # create a filename prefix using username + timestamp
        from datetime import datetime
        uname = (user.username or user.first_name or str(user.id)).strip()
        ts = datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        prefix = f"{uname}_{ts}"
        result = transcribe_voice(bytes(ogg_bytes), filename_prefix=prefix)
        text = result.get("text", "").strip()
    except Exception:
        text = ""

    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "voice", text)

    # If transcription failed, inform the user
    if not text:
        reply = "🎙️ عذراً، لم أستطع التعرف على الصوت. هل يمكنك إعادة الإرسال كنص؟"
        await update.message.reply_text(reply)
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)
        return

    # Pass transcribed text into the FSM flow
    user_id = user.id
    fsm = _get_fsm(user_id)
    reply, keyboard = await fsm.handle(text)

    # If FSM couldn't understand, ask Gemini for a context-aware reply
    if reply.strip() == "عفواً، ما فهمت. ممكن تعيد؟" and gemini._available:
        try:
            g_reply = await gemini.build_response(fsm.state.name, {**fsm.data, "last_user_message": text})
            if g_reply:
                reply = g_reply
        except Exception:
            pass

    await update.message.reply_text(reply, reply_markup=keyboard)

    with get_db() as db:
        crud.log_bot_reply(db, user.id, reply)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data or ""
    user_id = query.from_user.id
    await query.answer()

    with get_db() as db:
        crud.get_or_create_conversation(db, user_id, query.from_user.username, query.from_user.first_name, query.from_user.last_name)
        crud.log_message(db, user_id, "inbound", "callback", data)

    if data == "menu:book":
        reply = "📅 جيد! أرسل اسمك ومشكلتك لنبدأ الحجز."
        await query.edit_message_text(reply, reply_markup=main_menu_keyboard())
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    if data == "menu:inquiry":
        reply = "🔍 لإستعلام عن موعد، أرسل اسمك أو رقم التيلغرام وسأبحث عن الحجز."
        await query.edit_message_text(reply, reply_markup=main_menu_keyboard())
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    if data == "menu:cancel":
        reply = "❌ لإلغاء موعدك، الرجاء إرسال التفاصيل أو التواصل مع العيادة."
        await query.edit_message_text(reply, reply_markup=main_menu_keyboard())
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    if data == "menu:contact":
        reply = "📞 تواصل مع العيادة عبر الرقم الموجود على الموقع أو أرسل طلبك هنا."
        await query.edit_message_text(reply, reply_markup=main_menu_keyboard())
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    fsm = _get_fsm(user_id)
    reply, keyboard = await fsm.handle_callback(data)
    await query.edit_message_text(reply, reply_markup=keyboard)

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply)
