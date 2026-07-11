"""bot/handlers/patient.py — Patient-facing handler for booking and inquiries."""
from datetime import datetime, timezone

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
    return any(token in lowered for token in [
        "repeat", "كرر", "تكرار", "من جديد", "من اول", "restart", "ابدأ من جديد", "بدء جديد", "حجز موعد جديد",
    ])


def _is_inquiry_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "استعلام" in lowered or "موعدي" in lowered or "🔍" in lowered


def _is_cancel_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "إلغاء موعد" in lowered or "الغاء موعد" in lowered or "❌" in lowered


def _is_contact_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "تواصل" in lowered or "📞" in lowered


def _get_fsm(user_id: int, reset: bool = False) -> PatientFSM:
    session = _sessions.get(user_id)
    if reset or session is None:
        session = PatientFSM(user_id=user_id)
        _sessions[user_id] = session
    return session


def _format_appointment(appt) -> str:
    if not appt:
        return "لا يوجد موعد مسجل باسمك حالياً. يمكنك اختيار 📅 حجز موعد جديد."
    date_text = appt.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M") if appt.appt_datetime else "قائمة الانتظار"
    status_ar = {
        "confirmed": "مؤكد",
        "waitlisted": "قائمة انتظار",
        "completed": "مكتمل",
        "no_show": "غياب",
        "cancelled": "ملغي",
    }.get(appt.status, appt.status)
    patient_name = appt.patient.name if appt.patient else "—"
    specialty = appt.specialty_ar or appt.specialty or "—"
    return (
        "📌 آخر موعد مسجل:\n"
        f"رقم الحجز: {appt.appt_id}\n"
        f"المريض: {patient_name}\n"
        f"الوقت: {date_text}\n"
        f"التخصص: {specialty}\n"
        f"الأولوية: {appt.priority_class or '—'}\n"
        f"الحالة: {status_ar}"
    )


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user is not None:
        _get_fsm(user.id, reset=True)
        with get_db() as db:
            crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
            crud.log_message(db, user.id, "inbound", "command", "/start")

    reply = "👋 مرحباً! أنا مساعد الحجز. اختاري/اختر من القائمة أو أرسل اسمك وسبب زيارتك لأساعدك بالحجز."
    await update.message.reply_text(reply, reply_markup=main_menu_keyboard())

    if user is not None:
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)


async def _handle_menu_text(message, user, text: str) -> bool:
    """Return True when the message was handled as a main-menu command."""
    if _is_repeat_request(text):
        _get_fsm(user.id, reset=True)
        reply = "📅 تمام، خلينا نبدأ حجز جديد. ما اسمك الكريم؟"
        await message.reply_text(reply)
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)
        return True

    if _is_inquiry_request(text):
        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, user.id)
            reply = _format_appointment(appt)
            crud.log_bot_reply(db, user.id, reply)
        await message.reply_text(reply, reply_markup=main_menu_keyboard())
        return True

    if _is_cancel_request(text):
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user.id)
            reply = "تم إلغاء آخر موعد مؤكد/منتظر وإرجاع الـ slot كمتاح إذا كان محجوزاً. ✅" if appt else "لا يوجد موعد مؤكد أو منتظر لإلغائه حالياً."
            crud.log_bot_reply(db, user.id, reply)
        await message.reply_text(reply, reply_markup=main_menu_keyboard())
        return True

    if _is_contact_request(text):
        reply = "📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة."
        await message.reply_text(reply, reply_markup=main_menu_keyboard())
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)
        return True

    return False


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

    if text.strip() == "/start":
        await handle_start(update, context)
        return

    if await _handle_menu_text(message, user, text):
        return

    fsm = _get_fsm(user_id)
    reply, keyboard = await fsm.handle(text)

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

    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        ogg_bytes = await voice_file.download_as_bytearray()
        uname = (user.username or user.first_name or str(user.id)).strip()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        prefix = f"{uname}_{ts}"
        result = transcribe_voice(bytes(ogg_bytes), filename_prefix=prefix)
        text = result.get("text", "").strip()
    except Exception:
        text = ""

    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "voice", text)

    if not text:
        reply = "🎙️ عذراً، لم أستطع التعرف على الصوت. هل يمكنك إعادة الإرسال كنص؟"
        await update.message.reply_text(reply)
        with get_db() as db:
            crud.log_bot_reply(db, user.id, reply)
        return

    if await _handle_menu_text(message, user, text):
        return

    fsm = _get_fsm(user.id)
    reply, keyboard = await fsm.handle(text)

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
        _get_fsm(user_id, reset=True)
        reply = "📅 جيد! ما اسمك الكريم؟"
        await query.edit_message_text(reply)
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    if data == "menu:inquiry":
        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, user_id)
            reply = _format_appointment(appt)
            crud.log_bot_reply(db, user_id, reply)
        await query.edit_message_text(reply)
        return

    if data == "menu:cancel":
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user_id)
            reply = "تم إلغاء آخر موعد مؤكد/منتظر. ✅" if appt else "لا يوجد موعد لإلغائه حالياً."
            crud.log_bot_reply(db, user_id, reply)
        await query.edit_message_text(reply)
        return

    if data == "menu:contact":
        reply = "📞 تواصل مع العيادة عبر كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات."
        await query.edit_message_text(reply)
        with get_db() as db:
            crud.log_bot_reply(db, user_id, reply)
        return

    fsm = _get_fsm(user_id)
    reply, keyboard = await fsm.handle_callback(data)
    await query.edit_message_text(reply, reply_markup=keyboard)

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply)
