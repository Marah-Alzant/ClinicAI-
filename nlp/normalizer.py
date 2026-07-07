"""
nlp/normalizer.py — Levantine/Palestinian Arabic normalization pipeline.
"""
import re
import json
import unicodedata
from pathlib import Path

_VOCAB_PATH = Path(__file__).parent.parent / "data" / "levantine" / "vocab.json"
_vocab: dict = {}


def _load_vocab():
    global _vocab
    if not _vocab and _VOCAB_PATH.exists():
        _vocab = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    _load_vocab()
    text = unicodedata.normalize("NFC", text)
    text = _remove_diacritics(text)
    text = _normalize_letters(text)
    text = _apply_dialect_map(text)
    text = _normalize_whitespace(text)
    return text


def _remove_diacritics(text: str) -> str:
    return re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", text)


def _normalize_letters(text: str) -> str:
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة\b", "ه", text)  # ta marbuta at word end → ha
    return text


def _apply_dialect_map(text: str) -> str:
    if not _vocab:
        return text
    words = text.split()
    return " ".join(_vocab.get(w, w) for w in words)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
