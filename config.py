from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # Load environment variables from .env file
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TTS_VOICE = os.getenv("TTS_VOICE", "ar-PS-SamaNeural")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
CLINIC_NAME = os.getenv("CLINIC_NAME", "العيادة")
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_MODEL="gemini-3.1-flash-lite"