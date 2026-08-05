"""
nlp/gemini_client.py — Wrapper around Gemini API.

All Gemini calls go through this single client so the key,
model, and retry logic live in one place.
"""
import asyncio
from config import GEMINI_API_KEY, GEMINI_MODEL, CLINIC_NAME

try:
    import google.genai as genai
    from google.genai import types as google_types
except ImportError:  # pragma: no cover
    genai = None
    google_types = None

GEMINI_AVAILABLE = bool(GEMINI_API_KEY and genai is not None)
if GEMINI_AVAILABLE:
    client = genai.Client(api_key=GEMINI_API_KEY)

# System context injected into every conversation
SYSTEM_CONTEXT = f"""
أنت مساعد إداري ذكي لـ {CLINIC_NAME}.
مهمتك الوحيدة هي مساعدة المرضى في حجز المواعيدات وتقديم معلومات إدارية.
لا تقدم تشخيصات طبية أو نصائح علاجية بأي شكل.
تحدث دائماً باللهجة الفلسطينية العامية بشكل ودود وواضح.
إجاباتك قصيرة ومباشرة ولا تتجاوز ثلاثة أسطر إلا إذا طُلب منك أكثر.
""".strip()


class GeminiClient:
    def __init__(self):
        self._available = GEMINI_AVAILABLE
        self._model = None
        if self._available:
            self._model = GEMINI_MODEL

    async def ask(self, prompt: str, max_tokens: int = 300) -> str:
        """Single-turn question → answer."""
        if not self._available:
            raise RuntimeError("Gemini API key is not configured or google-generativeai is unavailable.")

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=google_types.GenerateContentConfig(
                system_instruction=SYSTEM_CONTEXT,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()

    async def build_response(self, fsm_state: str, data: dict) -> str:
        """
        Generate a context-aware Arabic response for a given FSM state.
        """
        if not self._available:
            return ""

        prompt = (
            f"حالة المحادثة: {fsm_state}\n"
            f"بيانات المحادثة الحالية: {data}\n"
            "اكتب ردًا ودودًا وقريبًا من طريقة الكلام البشري، باللغة العربية الفلسطينية العامية."
        )
        return await self.ask(prompt)

    async def extract_missing_field(self, text: str, missing_field: str) -> str:
        """Ask Gemini to extract a specific field from a free-form message."""
        if not self._available:
            return ""

        field_prompts = {
            "name":      "استخرج اسم المريض فقط. أجب بالاسم وحده بدون أي كلمة إضافية. إذا لا يوجد اسم أجب: لا يوجد",
            "complaint": "استخرج العرَض أو الشكوى الطبية فقط كما ذكرها المريض، بحد أقصى 8 كلمات، بدون مقدمة أو نصيحة أو سؤال. إذا لا توجد شكوى أجب: لا يوجد",
            "urgency":   "هل الحالة عاجلة أم متوسطة أم روتينية؟ أجب بكلمة واحدة فقط.",
            "time_pref": "متى يريد المريض الموعد؟ أجب بكلمة أو عبارة قصيرة فقط.",
        }
        instruction = field_prompts.get(missing_field, "استخرج المعلومة المطلوبة فقط بدون أي إضافة.")
        prompt = f"الرسالة: '{text}'\n{instruction}"
        raw = await self.ask(prompt, max_tokens=30)
        return self._clean_extraction(raw)

    @staticmethod
    def _clean_extraction(raw: str) -> str:
        """
        FIX (2026-07-14): validate extraction output before it is stored.
        Team testing showed chatty replies (e.g. "سلامتك، هاد الموضوع بيحتاج فحص
        سريري...") being saved as the patient's complaint and appearing in the
        dashboard's الشكوى column. Reject anything that looks like a chat reply
        instead of an extracted field value.
        """
        if not raw:
            return ""
        value = raw.strip().strip('"\'`').splitlines()[0].strip()
        if not value or "لا يوجد" in value:
            return ""
        if len(value) > 60 or "؟" in value or "?" in value:
            return ""
        chatty_markers = [
            "سلامتك", "اهلا", "أهلا", "أهلاً", "مرحبا", "مرحباً", "بقدر", "يمكنني",
            "انا هنا", "أنا هنا", "احجزلك", "أحجزلك", "تفضل", "بالتاكيد", "بالتأكيد",
            "عذرا", "عذراً", "للمساعده", "للمساعدة",
        ]
        if any(marker in value for marker in chatty_markers):
            return ""
        return value

    async def generate_voice_response(self, text: str) -> str:
        """
        Convert a structured bot reply into natural spoken Arabic
        suitable for TTS (no markdown, no bullet points).
        """
        if not self._available:
            return text

        prompt = (
            f"حوّل هذا النص إلى جملة عربية طبيعية تصلح للتحويل إلى صوت "
            f"(بدون رموز أو نقاط أو أرقام):\n{text}"
        )
        return await self.ask(prompt, max_tokens=150)


# Singleton — import this everywhere
gemini = GeminiClient()
