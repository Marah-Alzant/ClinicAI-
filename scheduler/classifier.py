"""
scheduler/classifier.py — Task: "Classifying patients for proper clinic"

Maps a patient's complaint + stated symptoms → correct specialty/clinic.
Uses a layered approach: keyword rules first, Gemini fallback for ambiguous cases.
"""
import re
import logging
from nlp.normalizer import normalize

logger = logging.getLogger(__name__)
# ── Specialty routing rules ────────────────────────────────────────────────────
# Each entry: pattern (regex on normalized Arabic) → specialty key
# Patterns run on normalized Arabic (ة→ه, أ/إ/آ→ا). Prefer phrases over single words.
ROUTING_RULES: list[tuple[str, str]] = [
    # Neurology — acute neuro symptoms (not mild dizziness alone)
    (
        r"صداع شديد|شلل|رعشه|تنميل|اعصاب|اغماء|تشنج|صرع|"
        r"ضعف مفاجئ|صعوبه نطق|تشوش بالكلام|فقدان توازن|جلطه|سكت[هة]"
        r"|دوخه شديده|دوخه مع تنميل",
        "neurology",
    ),
    # Orthopedics — musculoskeletal (avoid bare "ظهر")
    (
        r"كسر|التواء|عظام|مفاصل|الركبه|الورك|الكتف|"
        r"الم ظهر|وجع ظهر|الم ركبه|الم مفصل|عمود فقري|"
        r"\bوتر\b|الم عضله|وقعه|لا يستطيع المشي|لا عم يمشي|"
        r"الم\s*ظهر|وجع\s*ظهر|الم\s*رقبه|الم\s*كتف|الم\s*ركبه",
        "orthopedics",
    ),
    # Gynecology — OB/GYN context (avoid bare "نساء")
    (
        r"حمل|ولاده|دوره شهريه|الم رحم|الم مبيض|الم بطن للحامل|"
        r"نزيف رحمي|مشاكل الحمل|عياده نساء|توليد|مهبل|"
        r"الدور[هة]\s*الشهري[هة]|اضطرابات.*دور[هة]|عياد[هة]\s*نساء|فحص\s*مهبلي",
        "gynecology",
    ),
    # Dermatology — skin-specific (avoid bare "حبوب" / "جلد")
    (
        r"طفح جلدي|حكه جلد|حساسيه جلد|اكزيما|صدفيه|بقع جلد|"
        r"احمرار جلد|التهاب جلد|شرى|قشره جلد|"
        r"بشر[هة]|حب\s*الشباب|جفاف\s*جلد|احمرار\s*جلد|حك[هة]\s*جلد",
        "dermatology",
    ),
    # Gastroenterology — GI symptoms
    (
        r"الم معده|الم بطن|مغص|اسهال|امساك|قولون|بواسير|كبد|"
        r"ترجيع|غثيان|حرقه معده|انتفاخ بطن|"
        r"الم\s*بطن|وجع\s*بطن|عسر\s*هضم|الم\s*معد[هة]",
        "gastroenterology",
    ),
    # Chronic diseases — program routing (avoid bare "ضغط" → stress)
    (
        r"سكري|سكر|انسولين|هبوط سكر|ارتفاع سكر|السكر التراكمي|"
        r"ضغط الدم|ضغط مرتفع|قراءات ضغط|ادويه الضغط|متابعه الضغط|"
        r"ربو|كولسترول|دهون|غده درقيه|مرض مزمن|ادويه مزمنه|"
        r"متابعه السكر|تجديد وصفه|فحص السكر|متابعه مزمنه|"
        r"متابع[هة]\s*ل?ل?ضغط|فحص\s*روتيني\s*ل?ل?ضغط|قراءات\s*ضغط",
        "chronic_diseases",
    ),
    # Elderly — caregiver / geriatric program
    (
        r"جدي|جدتي|والدي المسن|والدتي المسنه|كبار السن|كبير سن|كبيره سن|"
        r"مسن|مسنه|شيخوخه|خرف|ذاكره|نسيان|اختلاط ادويه|"
        r"عمره\s*(6\d|7\d|8\d|9\d)|عمرها\s*(6\d|7\d|8\d|9\d)",
        "elderly",
    ),
    # General fallback — routine / mild acute
    (
        r"حمى|زكام|كحه خفيفه|ارهاق|وهن|فحص عام|كشف روتيني|"
        r"اعراض بسيطه|مراجعة عامه",
        "general_practice",
    ),
]

# ب) تفعيل 
# Gemini 
# في الـ 
# benchmark
#  للنصوص الغامضة —
#   في الإنتاج يعمل عبر 
#   plan_appointment()، 
#   لكن 
#   benchmark 
#   يستخدم 
#   classify_specialty() 
#   فقط.

# ── Human-readable Arabic specialty names ─────────────────────────────────────
SPECIALTY_NAMES_AR = {
    "neurology":         "طب الأعصاب",
    "orthopedics":       "العظام والمفاصل",
    "gynecology":        "النساء والتوليد",
    "dermatology":       "الأمراض الجلدية",
    "gastroenterology":  "الجهاز الهضمي",
    "chronic_diseases":  "الأمراض المزمنة",
    "elderly":           "كبار السن",
    "general_practice":  "الطب العام",
}

SPECIALTY_KEYS = set(SPECIALTY_NAMES_AR.keys())

_VALID_KEY_PATTERN = re.compile(r"[a-z_]+")


def _parse_gemini_specialty_key(raw: str) -> str | None:
    """Extract a valid specialty key from a Gemini free-text response."""
    if not raw or not raw.strip():
        return None
    text = raw.strip().lower()
    for token in _VALID_KEY_PATTERN.findall(text):
        if token in SPECIALTY_KEYS:
            return token
    compact = re.sub(r"[^a-z_]", "", text)
    for key in sorted(SPECIALTY_KEYS, key=len, reverse=True):
        if key in compact:
            return key
    return None


def classify_specialty(text: str) -> dict:
    """
    Rule-based classifier. Always normalizes input text first.
    Returns specialty key + Arabic name.
    Returns general_practice if no pattern matches.
    """
    normalized_text = normalize(text or "")
    for pattern, specialty in ROUTING_RULES:
        if re.search(pattern, normalized_text):
            return {
                "specialty":    specialty,
                "specialty_ar": SPECIALTY_NAMES_AR.get(specialty, specialty),
                "method":       "rule",
                "confidence":   0.9,  # realistic confidence for rule match
            }
    return {
        "specialty":    "general_practice",
        "specialty_ar": SPECIALTY_NAMES_AR["general_practice"],
        "method":       "default",
        "confidence":   0.5,
    }

async def classify_with_gemini_fallback(text: str, gemini_client,) -> dict:
    """
    Try rules first. If unresolved, ask Gemini and validate response.
    Use this for ambiguous or multi-complaint messages.
    """
    result = classify_specialty(text)

    if result["method"] == "default":
        normalized_text = normalize(text or "")
        prompt = (
            f"المريض يقول: '{normalized_text}'\n\n"
            f"اختر التخصص الأنسب من هذه المفاتيح فقط:\n"
            + "\n".join(f"- {k}: {v}" for k, v in SPECIALTY_NAMES_AR.items())
            + "\n\nأجب بمفتاح واحد فقط من القائمة بالإنجليزية (مثل: neurology). بدون أي نص إضافي."
        )
        try:
            gemini_result = await gemini_client.ask(prompt, max_tokens=20)
            if not (gemini_result or "").strip():
                raise ValueError("Empty response from Gemini")
            specialty_key = _parse_gemini_specialty_key(gemini_result)
            if specialty_key:
                result = {
                    "specialty":    specialty_key,
                    "specialty_ar": SPECIALTY_NAMES_AR[specialty_key],
                    "method":       "gemini",
                    "confidence":   0.85,
                }
            else:
                logger.warning("Gemini returned unknown specialty key: %s", gemini_result)
        except Exception as exc:
            logger.exception("Gemini fallback failed in classify_with_gemini_fallback: %s", exc)

    return result
