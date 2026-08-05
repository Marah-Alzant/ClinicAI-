"""
nlp/gemini_client.py — resilient LLM router.

Provider order:
1. Gemini when configured.
2. OpenAI API when configured.
3. Empty-string fallback so the deterministic FSM/rules continue normally.

No provider exception is allowed to stop the booking flow.
"""
from __future__ import annotations

import asyncio
import logging

from config import (
    CLINIC_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

try:
    import google.genai as genai
    from google.genai import types as google_types
except ImportError:  # pragma: no cover
    genai = None
    google_types = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

GEMINI_AVAILABLE = bool(GEMINI_API_KEY and genai is not None)
OPENAI_AVAILABLE = bool(OPENAI_API_KEY and OpenAI is not None)

_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_AVAILABLE else None
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_AVAILABLE else None

SYSTEM_CONTEXT = f"""
أنت مساعد إداري ذكي لـ {CLINIC_NAME}.
مهمتك الوحيدة هي مساعدة المرضى في حجز المواعيد وتقديم معلومات إدارية.
لا تقدم تشخيصات طبية أو نصائح علاجية بأي شكل.
تحدث دائماً باللهجة الفلسطينية العامية بشكل ودود وواضح.
إجاباتك قصيرة ومباشرة ولا تتجاوز ثلاثة أسطر إلا إذا طُلب منك أكثر.
""".strip()


class GeminiClient:
    """Backward-compatible name used by the rest of the project."""

    def __init__(self):
        self._gemini_available = GEMINI_AVAILABLE
        self._openai_available = OPENAI_AVAILABLE
        self._available = self._gemini_available or self._openai_available
        self._model = GEMINI_MODEL if self._gemini_available else None
        self._openai_model = OPENAI_MODEL if self._openai_available else None
        self._timeout = max(int(LLM_TIMEOUT_SECONDS or 20), 1)

    async def _ask_gemini(self, prompt: str, max_tokens: int) -> str:
        if not self._gemini_available or _gemini_client is None:
            return ""

        def _request():
            return _gemini_client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=google_types.GenerateContentConfig(
                    system_instruction=SYSTEM_CONTEXT,
                    max_output_tokens=max_tokens,
                ),
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_request),
                timeout=self._timeout,
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # network/model/key/timeout: continue to fallback
            logger.warning("Gemini request failed; trying fallback: %s", exc)
            return ""

    async def _ask_openai(self, prompt: str, max_tokens: int) -> str:
        if not self._openai_available or _openai_client is None:
            return ""

        def _request():
            return _openai_client.responses.create(
                model=self._openai_model,
                instructions=SYSTEM_CONTEXT,
                input=prompt,
                max_output_tokens=max_tokens,
            )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_request),
                timeout=self._timeout,
            )
            return (getattr(response, "output_text", "") or "").strip()
        except Exception as exc:
            logger.warning("OpenAI fallback request failed; using local FSM: %s", exc)
            return ""

    async def ask(self, prompt: str, max_tokens: int = 300) -> str:
        """Never raises: Gemini -> optional OpenAI -> local empty fallback."""
        if not prompt:
            return ""

        response = await self._ask_gemini(prompt, max_tokens)
        if response:
            return response

        response = await self._ask_openai(prompt, max_tokens)
        if response:
            return response

        return ""

    async def build_response(self, fsm_state: str, data: dict) -> str:
        prompt = (
            f"حالة المحادثة: {fsm_state}\n"
            f"بيانات المحادثة الحالية: {data}\n"
            "اكتب ردًا ودودًا وقريبًا من طريقة الكلام البشري، باللغة العربية الفلسطينية العامية."
        )
        return await self.ask(prompt)

    async def extract_missing_field(self, text: str, missing_field: str) -> str:
        field_prompts = {
            "name": "استخرج اسم المريض فقط. أجب بالاسم وحده بدون أي كلمة إضافية. إذا لا يوجد اسم أجب: لا يوجد",
            "complaint": "استخرج العرَض أو الشكوى فقط كما قالها المريض، بحد أقصى 8 كلمات، بلا مقدمة أو سؤال. إذا لا توجد شكوى أجب: لا يوجد",
            "urgency": "هل الحالة عاجلة أم متوسطة أم روتينية؟ أجب بكلمة واحدة فقط.",
            "time_pref": "متى يريد المريض الموعد؟ أجب بكلمة أو عبارة قصيرة فقط.",
        }
        instruction = field_prompts.get(
            missing_field,
            "استخرج المعلومة المطلوبة فقط بدون أي إضافة.",
        )
        raw = await self.ask(
            f"الرسالة: '{text}'\n{instruction}",
            max_tokens=30,
        )
        return self._clean_extraction(raw)

    @staticmethod
    def _clean_extraction(raw: str) -> str:
        if not raw:
            return ""
        value = raw.strip().strip('"\'`').splitlines()[0].strip()
        if not value or "لا يوجد" in value:
            return ""
        if len(value) > 60 or "؟" in value or "?" in value:
            return ""
        chatty_markers = [
            "سلامتك", "اهلا", "أهلا", "أهلاً", "مرحبا", "مرحباً", "بقدر",
            "يمكنني", "انا هنا", "أنا هنا", "احجزلك", "أحجزلك", "تفضل",
            "بالتاكيد", "بالتأكيد", "عذرا", "عذراً", "للمساعده", "للمساعدة",
        ]
        if any(marker in value for marker in chatty_markers):
            return ""
        return value

    async def generate_voice_response(self, text: str) -> str:
        prompt = (
            "حوّل النص التالي إلى جملة عربية طبيعية تصلح للصوت، "
            f"بدون رموز أو قوائم:\n{text}"
        )
        response = await self.ask(prompt, max_tokens=150)
        return response or text

    def provider_status(self) -> dict[str, bool]:
        return {
            "gemini": self._gemini_available,
            "openai": self._openai_available,
            "local_fallback": True,
        }


gemini = GeminiClient()
