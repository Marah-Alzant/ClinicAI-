"""
voice/tts.py — Task: "Speech generation for response"
edge-tts uses ar-PS-SamaNeural (Palestinian Arabic, female) — near-human quality.
Requires internet only for the TTS API call; no hosting needed.
"""
import asyncio
import subprocess
import re
from pathlib import Path
from config import TTS_VOICE, TEMP_DIR


async def text_to_ogg(text: str) -> bytes:
    """
    Convert Arabic text → .ogg bytes ready to send as Telegram voice note.
    Cleans markdown before synthesis.
    """
    clean = _strip_markdown(text)

    mp3_path = TEMP_DIR / "reply.mp3"
    ogg_path = TEMP_DIR / "reply.ogg"

    try:
        import edge_tts
    except Exception as exc:
        raise ImportError(
            "The 'edge_tts' package is required for TTS. "
            "Install it only if you need voice response support."
        ) from exc

    communicate = edge_tts.Communicate(text=clean, voice=TTS_VOICE, rate="-5%")
    await communicate.save(str(mp3_path))

    _mp3_to_ogg(mp3_path, ogg_path)

    data = ogg_path.read_bytes()
    mp3_path.unlink(missing_ok=True)
    ogg_path.unlink(missing_ok=True)
    return data


def _strip_markdown(text: str) -> str:
    """Remove Telegram markdown so TTS doesn't read asterisks and underscores."""
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # [label](url) → label
    text = re.sub(r"📅|📆|✅|🏥|🔴|🟡|🟢|👤|🩺|⏱|🔬|💊|🔭|👋|😊|🌿|🎙️|📋|✏️|❓", "", text)
    return text.strip()


def _mp3_to_ogg(mp3_path: Path, ogg_path: Path):
    """ffmpeg: mp3 → opus/ogg (Telegram voice format)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(mp3_path),
            "-c:a", "libopus",
            "-b:a", "32k",
            str(ogg_path),
        ],
        capture_output=True,
        check=True,
    )


def synthesize_sync(text: str) -> bytes:
    """Sync wrapper — use in non-async contexts."""
    return asyncio.run(text_to_ogg(text))
