"""
scheduler/classifier.py — Task: "Classifying patients for proper clinic"

Maps a patient's complaint + stated symptoms → correct specialty/clinic.
Uses a layered approach: keyword rules first, Gemini fallback for ambiguous cases.
"""
import re
from typing import Optional


# ── Specialty routing rules ────────────────────────────────────────────────────
# Each entry: pattern (regex on normalized Arabic) → specialty key
ROUTING_RULES: list[tuple[str, str]] = [
    # Cardiology
    (r"ألم صدر|وجع صدر|ضغط على الصدر|ضغط دم|شريان|قلب|نبض|تخطيط قلب|إيكو", "cardiology"),
    # Neurology
    (r"صداع شديد|دوخة|شلل|رعشة|تنميل|أعصاب|إغماء|تشنج|صرع", "neurology"),
    # Endocrinology
    (r"سكر|سكري|هرمون|غدة درقية|غدد|هرمونات|تعب زيادة عن العادي", "endocrinology"),
    # Orthopedics
    (r"كسر|مفاصل|ركبة|ظهر|عمود فقري|عظام|التواء|وتر|عضلة", "orthopedics"),
    # Gynecology
    (r"نساء|حمل|ولادة|دورة شهرية|رحم|مبيض|حضانة|مهبل", "gynecology"),
    # Pediatrics
    (r"طفل|رضيع|أطفال|ولد صغير|بنت صغيرة|حديث الولادة", "pediatrics"),
    # Ophthalmology
    (r"عين|نظر|بصر|ضباب|احمرار عين|قرنية|نظارة", "ophthalmology"),
    # Dermatology
    (r"جلد|حساسية جلد|بقع|أكزيما|صدفية|حبوب|طفح", "dermatology"),
    # Dentistry
    (r"أسنان|ضرس|لثة|تسوس|جذر|تقويم|أسنان اصطناعية", "dentistry"),
    # Respiratory
    (r"رئة|تنفس|ربو|كحة مزمنة|بلغم|التهاب رئة", "pulmonology"),
    # Gastroenterology
    (r"معدة|أمعاء|هضم|إسهال|إمساك|قولون|بواسير|كبد", "gastroenterology"),
    # ENT
    (r"أذن|حلق|أنف|لوزتين|جيوب أنفية|سمع|صوت", "ent"),
    # Psychiatry
    (r"اكتئاب|قلق|وسواس|نوم|نفسي|ضغط نفسي|مزاج", "psychiatry"),
    # General fallback
    (r"حمى|زكام|كحة|إرهاق|وهن|فحص عام|كشف روتيني", "general_practice"),
]

# ── Human-readable Arabic specialty names ─────────────────────────────────────
SPECIALTY_NAMES_AR = {
    "cardiology":        "طب القلب",
    "neurology":         "طب الأعصاب",
    "endocrinology":     "الغدد الصماء",
    "orthopedics":       "العظام والمفاصل",
    "gynecology":        "النساء والتوليد",
    "pediatrics":        "طب الأطفال",
    "ophthalmology":     "طب العيون",
    "dermatology":       "الأمراض الجلدية",
    "dentistry":         "طب الأسنان",
    "pulmonology":       "طب الصدر والجهاز التنفسي",
    "gastroenterology":  "الجهاز الهضمي",
    "ent":               "أنف وأذن وحنجرة",
    "psychiatry":        "الطب النفسي",
    "general_practice":  "الطب العام",
}


def classify_specialty(normalized_text: str) -> dict:
    """
    Rule-based classifier. Returns specialty key + Arabic name.
    Returns general_practice if no pattern matches.
    """
    for pattern, specialty in ROUTING_RULES:
        if re.search(pattern, normalized_text):
            return {
                "specialty":    specialty,
                "specialty_ar": SPECIALTY_NAMES_AR.get(specialty, specialty),
                "method":       "rule",
                "confidence":   1.0,
            }
    return {
        "specialty":    "general_practice",
        "specialty_ar": SPECIALTY_NAMES_AR["general_practice"],
        "method":       "default",
        "confidence":   0.5,
    }


async def classify_with_gemini_fallback(
    normalized_text: str,
    gemini_client,
) -> dict:
    """
    Try rules first. If confidence is low, ask Gemini to classify.
    Use this for ambiguous or multi-complaint messages.
    """
    result = classify_specialty(normalized_text)

    if result["method"] == "default":
        # Ask Gemini to classify the specialty
        prompt = (
            f"المريض يقول: '{normalized_text}'\n\n"
            f"بناءً على ما ذكره، ما هو التخصص الطبي الأنسب من هذه الخيارات:\n"
            + "\n".join(f"- {k}: {v}" for k, v in SPECIALTY_NAMES_AR.items())
            + "\n\nأجب بمفتاح التخصص الإنجليزي فقط (مثال: cardiology). لا تكتب أي شيء آخر."
        )
        try:
            gemini_result = await gemini_client.ask(prompt, max_tokens=20)
            specialty_key = gemini_result.strip().lower().split()[0]
            if specialty_key in SPECIALTY_NAMES_AR:
                result = {
                    "specialty":    specialty_key,
                    "specialty_ar": SPECIALTY_NAMES_AR[specialty_key],
                    "method":       "gemini",
                    "confidence":   0.85,
                }
        except Exception:
            pass  # silently keep the rule-based default

    return result
