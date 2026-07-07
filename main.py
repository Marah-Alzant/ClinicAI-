"""
main.py — Task: "Server connecting all components"
Single entry point. Starts:
  1. SQLite DB (creates tables if new)
  2. FastAPI dashboard server (background thread)
  3. Telegram bot (main thread, long-polling)
"""
import asyncio
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import DASHBOARD_HOST, DASHBOARD_PORT, TELEGRAM_BOT_TOKEN
from database.db import init_db
from dashboard.routes import router as dashboard_router

from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters,
)
from bot.router import route_text, route_voice, route_callback, route_start


# ── FastAPI app ────────────────────────────────────────────────────────────────

def create_fastapi_app() -> FastAPI:
    app = FastAPI(title="ClinicAI Dashboard", docs_url=None)
    app.include_router(dashboard_router)
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        pass
    return app


def run_dashboard():
    app = create_fastapi_app()
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",   # quiet — bot logs are enough
    )


# ── Telegram bot ───────────────────────────────────────────────────────────────

def build_bot() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",   route_start))
    app.add_handler(MessageHandler(filters.VOICE,                    route_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  route_text))
    app.add_handler(CallbackQueryHandler(route_callback))

    return app

# Database initialization is now handled by the database package.
# ── Startup ────────────────────────────────────────────────────────────────────

def main():
    print("🏥 ClinicAI starting...")

    # 1. Initialize database
    init_db()
    print("✅ Database ready")

    # 2. Start dashboard in a background daemon thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    print(f"✅ Dashboard → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    # 3. Start Telegram bot (blocking — keeps the process alive)
    print(f"✅ Bot is running. Press Ctrl+C to stop.\n")
    bot = build_bot()
    bot.run_polling(drop_pending_updates=True, timeout=30)


if __name__ == "__main__":
    main()
