"""
voice/stt.py — Task: "Speech recognition for voice messages"
Whisper runs fully locally. Model is loaded once and kept in memory.
"""
import subprocess
import tempfile
from pathlib import Path
from config import WHISPER_MODEL, TEMP_DIR

_whisper = None
_model = None

# Prime Whisper toward Palestinian clinic vocabulary
_INITIAL_PROMPT = (
    "هذا تسجيل من عيادة طبية فلسطينية. المتحدث يستخدم اللهجة الفلسطينية. "
    "كلمات شائعة: موعد، حجز، ألم، دواء، ضغط، سكر، متابعة، كشف، دكتور، عيادة، "
    "بدي، هلق، بكرا، مبارح، وجع، سخونة، زكمة، كحة."
)


def get_model():
    global _model, _whisper
    if _whisper is None:
        try:
            import whisper as _whisper_module
        except Exception as exc:
            raise ImportError(
                "The 'whisper' package is required for voice transcription. "
                "Install it only if you need STT, or remove doctor voice handlers."
            ) from exc
        _whisper = _whisper_module

    if _model is None:
        print(f"[STT] Loading Whisper '{WHISPER_MODEL}' model (first run only)...")
        _model = _whisper.load_model(WHISPER_MODEL)
        print("[STT] Model loaded.")
    return _model


def transcribe_voice(ogg_bytes: bytes, filename_prefix: str | None = None) -> dict:
    """
    Receive raw .ogg bytes from Telegram, transcribe to Arabic text.
    Returns: {text, language, words, duration_sec}
    """
    # Build safe filenames to keep recordings distinguishable
    try:
        import re
    except Exception:
        re = None

    if filename_prefix:
        safe = filename_prefix
        if re is not None:
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", filename_prefix)
    else:
        safe = "input"

    ogg_path = TEMP_DIR / f"In_{safe}.ogg"
    wav_path = TEMP_DIR / f"In_{safe}.wav"

    ogg_path.write_bytes(ogg_bytes)
    _ogg_to_wav(ogg_path, wav_path)

    model = get_model()
    result = model.transcribe(
        str(wav_path),
        language="ar",
        task="transcribe",
        word_timestamps=True,
        initial_prompt=_INITIAL_PROMPT,
    )

    # Cleanup
    ogg_path.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)

    return {
        "text":         result["text"].strip(),
        "language":     result.get("language", "ar"),
        "words":        result.get("words", []),
        "duration_sec": result.get("duration", 0),
    }


def _ogg_to_wav(ogg_path: Path, wav_path: Path):
    """Convert Telegram opus/ogg → 16kHz mono WAV (Whisper's required format)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(ogg_path),
            "-ar", "16000",   # 16 kHz sample rate
            "-ac", "1",       # mono
            "-f", "wav",
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )
