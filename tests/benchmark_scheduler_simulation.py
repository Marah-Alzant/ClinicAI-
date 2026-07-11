"""Benchmark scheduler classifier + priority against 120 Arabic cases.

Default mode mirrors production: rules first, then real Gemini fallback
(via scheduler._classify) when no rule matches.

Usage:
    python -m tests.benchmark_scheduler              # production path + Gemini
    python -m tests.benchmark_scheduler --rules-only   # rules only (no API)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.classifier import classify_specialty
from scheduler.priority import score_and_classify
from scheduler.scheduler import _classify, normalize_priority_class, sanitize_input
TEST_DATASET = [
    {'id': 1, 'input': "أشعر ببعض الوجع في الظهر بعد الجلوس طويلاً.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P3'}},
    {'id': 2, 'input': "فحص دوري.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P3'}},
    {'id': 3, 'input': "استشارة عامة عن العناية بالبشرة.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P3'}},
    {'id': 4, 'input': "فحص روتيني لكبار السن.", 'expected': {'Clinic': 'elderly', 'Priority': 'P3'}},
    {'id': 5, 'input': "استشارة بخصوص أعراض بسيطة.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P3'}},
    {'id': 6, 'input': "والدي المسن يحتاج لفحص دوري.", 'expected': {'Clinic': 'elderly', 'Priority': 'P3'}},
    {'id': 7, 'input': "ألم في الصدر مع صعوبة في البلع.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P1'}},
    {'id': 8, 'input': "فحص روتيني في عيادة النساء.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P3'}},
    {'id': 9, 'input': "وقعت على ظهري وأشعر بألم شديد جداً.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P1'}},
    {'id': 10, 'input': "لدي اضطرابات في الدورة الشهرية.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P2'}},
    {'id': 11, 'input': "جدتي عمرها 75 سنة وتحتاج لمتابعة عامة.", 'expected': {'Clinic': 'elderly', 'Priority': 'P2'}},
    {'id': 12, 'input': "بدي متابعة للضغط لأن القراءات عندي بالأيام الأخيرة صارت فوق 160 على 100.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P2'}},
    {'id': 13, 'input': "أعاني من كحة خفيفة وإرهاق.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P2'}},
    {'id': 14, 'input': "هبوط سكر حاد وأشعر بدوخة شديدة.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P1'}},
    {'id': 15, 'input': "لدي إسهال حاد مع جفاف شديد.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P1'}},
    {'id': 16, 'input': "احمرار في الجلد مع حكة خفيفة.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P2'}},
    {'id': 17, 'input': "ولادة مبكرة وأحتاج لرعاية عاجلة.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P1'}},
    {'id': 18, 'input': "حرقة معدة مستمرة بعد الأكل.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P2'}},
    {'id': 19, 'input': "أعاني من قولون عصبي ومغص متكرر.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P2'}},
    {'id': 20, 'input': "صدفية متفاقمة وتغطي مساحات كبيرة من الجلد.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P1'}},
    {'id': 21, 'input': "فحص مهبلي روتيني.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P2'}},
    {'id': 22, 'input': "دوخة بسيطة عند الوقوف.", 'expected': {'Clinic': 'neurology', 'Priority': 'P3'}},
    {'id': 23, 'input': "فحص روتيني لمفاصلي.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P3'}},
    {'id': 24, 'input': "اشتباه في تسمم غذائي مع غثيان وترجيع.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P1'}},
    {'id': 25, 'input': "فحص عام للأعصاب.", 'expected': {'Clinic': 'neurology', 'Priority': 'P3'}},
    {'id': 26, 'input': "متابعة لحالة جلدية بسيطة.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P3'}},
    {'id': 27, 'input': "ارتفاع سكر خفيف وأحتاج لتعديل جرعة الأنسولين.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P2'}},
    {'id': 28, 'input': "أعاني من التهاب جلدي حاد مع تقرحات.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P1'}},
    {'id': 29, 'input': "تجديد وصفة أدوية الضغط.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P2'}},
    {'id': 30, 'input': "ألم خفيف في البطن.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P3'}},
    {'id': 31, 'input': "بدي متابعة للضغط لأن القراءات عندي بالأيام الأخيرة صارت فوق 180 على 110.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P1'}},
    {'id': 32, 'input': "فحص عام بسبب شعور بالتعب.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P2'}},
    {'id': 33, 'input': "لدي تشوش في الكلام وفقدان توازن أحياناً.", 'expected': {'Clinic': 'neurology', 'Priority': 'P2'}},
    {'id': 34, 'input': "أحتاج لمتابعة روتينية لأعصابي.", 'expected': {'Clinic': 'neurology', 'Priority': 'P3'}},
    {'id': 35, 'input': "قشرة جلدية شديدة في فروة الرأس.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P2'}},
    {'id': 36, 'input': "والدتي المسنة عمرها 68 سنة وتحتاج لفحص روتيني.", 'expected': {'Clinic': 'elderly', 'Priority': 'P2'}},
    {'id': 37, 'input': "أشعر بوهن عام.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P2'}},
    {'id': 38, 'input': "أشعر بتنميل مستمر في الأطراف ورعشة خفيفة.", 'expected': {'Clinic': 'neurology', 'Priority': 'P2'}},
    {'id': 39, 'input': "مراجعة عامة للعظام.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P3'}},
    {'id': 40, 'input': "دوخة خفيفة مع شعور بالوهن.", 'expected': {'Clinic': 'neurology', 'Priority': 'P2'}},
    {'id': 41, 'input': "عندي سكري وبدي فحص روتيني للسكر التراكمي بعدين.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P3'}},
    {'id': 42, 'input': "استشارة بخصوص آلام الرأس الخفيفة.", 'expected': {'Clinic': 'neurology', 'Priority': 'P3'}},
    {'id': 43, 'input': "إمساك مزمن وأحتاج لحل.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P2'}},
    {'id': 44, 'input': "أحتاج لمتابعة حمل روتينية.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P2'}},
    {'id': 45, 'input': "أشعر بانتفاخ خفيف في البطن أحياناً.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P3'}},
    {'id': 46, 'input': "أعاني من كحة قوية جداً مع ضيق في التنفس.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P1'}},
    {'id': 47, 'input': "مراجعة عامة للمعدة.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P3'}},
    {'id': 48, 'input': "كولسترول مرتفع جداً مع آلام في الصدر.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P1'}},
    {'id': 49, 'input': "الم شديد في البطن للحامل مع تقلصات.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P1'}},
    {'id': 50, 'input': "أصبت بجلطة دماغية قبل قليل وفقدت القدرة على الكلام.", 'expected': {'Clinic': 'neurology', 'Priority': 'P1'}},
    {'id': 51, 'input': "مشاكل حمل خطيرة وأحتاج لمتابعة فورية.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P1'}},
    {'id': 52, 'input': "بقعة صغيرة على الجلد أريد فحصها.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P3'}},
    {'id': 53, 'input': "والدي المسن يعاني من نسيان خفيف.", 'expected': {'Clinic': 'elderly', 'Priority': 'P2'}},
    {'id': 54, 'input': "مراجعة عامة للصحة.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P3'}},
    {'id': 55, 'input': "فحص عام للصحة الإنجابية.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P3'}},
    {'id': 56, 'input': "كبير سن يعاني من ضعف عام ولا يستطيع الاعتناء بنفسه.", 'expected': {'Clinic': 'elderly', 'Priority': 'P1'}},
    {'id': 57, 'input': "مغص قوي جداً لا يهدأ مع انتفاخ كبير في البطن.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P1'}},
    {'id': 58, 'input': "شرى حاد مع صعوبة في التنفس.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P1'}},
    {'id': 59, 'input': "لدي غدة درقية وأحتاج لتعديل الأدوية.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P2'}},
    {'id': 60, 'input': "أحتاج لمتابعة لعمودي الفقري.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P2'}},
    {'id': 61, 'input': "جدتي عمرها 90 سنة وأصيبت باختلاط أدوية خطير.", 'expected': {'Clinic': 'elderly', 'Priority': 'P1'}},
    {'id': 62, 'input': "ألم خفيف في البطن خلال الدورة.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P3'}},
    {'id': 63, 'input': "استشارة بخصوص ألم خفيف في الركبة.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P3'}},
    {'id': 64, 'input': "نزيف من الجهاز الهضمي وأحتاج لمتابعة عاجلة.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P1'}},
    {'id': 65, 'input': "دوخة شديدة مع تنميل في الوجه والأطراف.", 'expected': {'Clinic': 'neurology', 'Priority': 'P1'}},
    {'id': 66, 'input': "لدي بواسير مؤلمة وأحتاج لعلاج.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P2'}},
    {'id': 67, 'input': "جدي عمره 65 سنة ويحتاج لمتابعة لأدوية الضغط.", 'expected': {'Clinic': 'elderly', 'Priority': 'P2'}},
    {'id': 68, 'input': "استشارة بخصوص مشاكل نسائية عامة.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P2'}},
    {'id': 69, 'input': "متابعة لمرض مزمن وأشعر ببعض التعب.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P2'}},
    {'id': 70, 'input': "أشعر بألم في المبيض وأحتاج لفحص.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P2'}},
    {'id': 71, 'input': "عندي صداع شديد جداً مع شلل في الجانب الأيسر من الجسم.", 'expected': {'Clinic': 'neurology', 'Priority': 'P1'}},
    {'id': 72, 'input': "أعاني من ألم مزمن في الظهر والرقبة.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P2'}},
    {'id': 73, 'input': "كبير سن يعاني من وهن عام.", 'expected': {'Clinic': 'elderly', 'Priority': 'P2'}},
    {'id': 74, 'input': "حساسية جلدية شديدة تسببت في تورم الوجه.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P1'}},
    {'id': 75, 'input': "التواء شديد في الركبة مع تورم كبير.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P1'}},
    {'id': 76, 'input': "والدتي المسنة عمرها 78 سنة وتعرضت لسقوط مفاجئ.", 'expected': {'Clinic': 'elderly', 'Priority': 'P1'}},
    {'id': 77, 'input': "لدي حمى شديدة جداً مع آلام في الجسم.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P1'}},
    {'id': 78, 'input': "استشارة بخصوص حب الشباب.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P2'}},
    {'id': 79, 'input': "فحص روتيني للجهاز الهضمي.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P3'}},
    {'id': 80, 'input': "فحص روتيني للضغط.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P3'}},
    {'id': 81, 'input': "أعاني من أزمة ربو حادة وأحتاج لمساعدة.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P1'}},
    {'id': 82, 'input': "فحص روتيني عام.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P3'}},
    {'id': 83, 'input': "متابعة عامة لجدتي.", 'expected': {'Clinic': 'elderly', 'Priority': 'P3'}},
    {'id': 84, 'input': "زكام حاد مع إرهاق شديد.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P1'}},
    {'id': 85, 'input': "ألم خفيف في الكتف.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P3'}},
    {'id': 86, 'input': "استشارة بخصوص تنظيم الأسرة.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P3'}},
    {'id': 87, 'input': "استشارة بخصوص الكولسترول.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P3'}},
    {'id': 88, 'input': "أعاني من نوبات صرع متكررة وأحتاج لمتابعة.", 'expected': {'Clinic': 'neurology', 'Priority': 'P2'}},
    {'id': 89, 'input': "الم عضلي شديد بعد مجهود.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P2'}},
    {'id': 90, 'input': "والدي المسن عمره 85 سنة ويعاني من خرف شديد وفقدان ذاكرة حاد.", 'expected': {'Clinic': 'elderly', 'Priority': 'P1'}},
    {'id': 91, 'input': "أعاني من ألم معدة شديد جداً مع ترجيع مستمر.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P1'}},
    {'id': 92, 'input': "الم مفصل الورك لا يحتمل ولا أستطيع الحركة.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P1'}},
    {'id': 93, 'input': "أعاني من أكزيما مزمنة وأحتاج لمتابعة.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P2'}},
    {'id': 94, 'input': "جدي عمره 70 سنة ويعاني من صعوبة في التنفس.", 'expected': {'Clinic': 'elderly', 'Priority': 'P1'}},
    {'id': 95, 'input': "أحتاج لاستشارة بخصوص جفاف الجلد.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P3'}},
    {'id': 96, 'input': "عندي السكر صار عالي اليوم ووصل تقريباً 310 ومعه عطش كثير وتبول متكرر.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P1'}},
    {'id': 97, 'input': "لدي ألم في الركبة يزداد عند صعود الدرج.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P2'}},
    {'id': 98, 'input': "ألم في منطقة الكبد مع تعب عام.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P2'}},
    {'id': 99, 'input': "أشعر بوجع في عظامي ومفاصلي.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P2'}},
    {'id': 100, 'input': "أنا حامل وأعاني من نزيف رحمي شديد.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P1'}},
    {'id': 101, 'input': "أشعر بتنميل خفيف في يدي أحياناً.", 'expected': {'Clinic': 'neurology', 'Priority': 'P3'}},
    {'id': 102, 'input': "استشارة بخصوص رعاية مسن.", 'expected': {'Clinic': 'elderly', 'Priority': 'P3'}},
    {'id': 103, 'input': "مراجعة عامة لأعراض بسيطة.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P2'}},
    {'id': 104, 'input': "لدي حمى خفيفة وزكام.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P2'}},
    {'id': 105, 'input': "ألم في الأعصاب مع ضعف عام.", 'expected': {'Clinic': 'neurology', 'Priority': 'P2'}},
    {'id': 106, 'input': "تعرضت لكسر في الساق ولا أستطيع المشي.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P1'}},
    {'id': 107, 'input': "استشارة بخصوص عسر الهضم.", 'expected': {'Clinic': 'gastroenterology', 'Priority': 'P3'}},
    {'id': 108, 'input': "استشارة بخصوص التغذية لكبار السن.", 'expected': {'Clinic': 'elderly', 'Priority': 'P3'}},
    {'id': 109, 'input': "تجديد وصفة أدوية الربو.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P3'}},
    {'id': 110, 'input': "فحص روتيني للجلد.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P3'}},
    {'id': 111, 'input': "أحتاج لكشف روتيني.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P3'}},
    {'id': 112, 'input': "لدي آلام رحم قوية جداً ومخاوف على الحمل.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P1'}},
    {'id': 113, 'input': "تمزق وتر الكتف وأحتاج لتدخل عاجل.", 'expected': {'Clinic': 'orthopedics', 'Priority': 'P1'}},
    {'id': 114, 'input': "لدي طفح جلدي منتشر في كل الجسم مع حكة شديدة وحرارة.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P1'}},
    {'id': 115, 'input': "لدي بقع جلدية غريبة تظهر وتختفي.", 'expected': {'Clinic': 'dermatology', 'Priority': 'P2'}},
    {'id': 116, 'input': "أشعر بوهن عام وعدم قدرة على الحركة.", 'expected': {'Clinic': 'general_practice', 'Priority': 'P1'}},
    {'id': 117, 'input': "تعرضت لإغماء مفاجئ وتشنجات قوية.", 'expected': {'Clinic': 'neurology', 'Priority': 'P1'}},
    {'id': 118, 'input': "أعاني من ضعف مفاجئ في الأطراف وصعوبة في النطق.", 'expected': {'Clinic': 'neurology', 'Priority': 'P1'}},
    {'id': 119, 'input': "متابعة عامة للأمراض المزمنة.", 'expected': {'Clinic': 'chronic_diseases', 'Priority': 'P3'}},
    {'id': 120, 'input': "متابعة بعد الولادة.", 'expected': {'Clinic': 'gynecology', 'Priority': 'P3'}},
]


def infer_urgency_score(text: str) -> float:
    t = text
    high = ["الآن", "فجأة", "حاد", "شديد", "صعوبة نطق", "ضعف مفاجئ", "وقع", "أزمة", "هبوط سكر"]
    mid = ["دوخة", "تعب", "تنميل", "صفير", "تورم", "رجفة"]
    low = ["متابعة", "دوري", "روتيني", "مراجعة", "بدون أعراض جديدة"]

    if any(k in t for k in high):
        return 0.9
    if any(k in t for k in mid):
        return 0.55
    if any(k in t for k in low):
        return 0.25
    return 0.4


def _build_patient_data(text: str) -> dict:
    """Build FSM-like payload, then sanitize like plan_appointment()."""
    urgency = infer_urgency_score(text)
    raw = {
        "complaint": {"raw": text},
        "urgency_score": urgency,
        "is_followup": any(k in text for k in ("متابعة", "دوري", "روتيني")),
        "time_pref": {"date": None, "phrase": "أي وقت"},
    }
    safe = sanitize_input(raw)
    safe["complaint"]["urgency_score"] = urgency
    safe["complaint"]["specialty"] = None
    return safe


async def predict_case(row: dict, gemini_client=None, *, gemini_delay: float = 0.0) -> dict:
    """Production-aligned path: sanitize → _classify → score_and_classify."""
    text = row["input"]
    if gemini_client is not None and classify_specialty(text)["method"] == "default":
        if gemini_delay > 0:
            await asyncio.sleep(gemini_delay)
    safe_data = _build_patient_data(text)

    spec_result = await _classify(safe_data, gemini_client)
    safe_data["specialty_hint"] = spec_result["specialty"]
    safe_data["specialty_ar"] = spec_result["specialty_ar"]
    safe_data["complaint"]["specialty"] = spec_result["specialty"]

    pr = score_and_classify(safe_data)

    return {
        "pred_clinic": spec_result["specialty"],
        "pred_prio": normalize_priority_class(pr.priority_class),
        "method": spec_result.get("method", "unknown"),
        "confidence": float(spec_result.get("confidence", 0.0)),
        "priority_score": pr.score,
    }


async def measure_performance_async(
    dataset, predict_fn, gemini_client=None, *, gemini_delay: float = 0.0, warmup: int = 3,
):
    for i in range(min(warmup, len(dataset))):
        await predict_fn(dataset[i], gemini_client, gemini_delay=gemini_delay)

    per_case_times = []
    t0 = time.perf_counter()
    for case in dataset:
        c0 = time.perf_counter()
        await predict_fn(case, gemini_client, gemini_delay=gemini_delay)
        per_case_times.append((time.perf_counter() - c0) * 1000.0)
    total_sec = time.perf_counter() - t0
    total_cases = len(dataset)
    throughput = (total_cases / total_sec) if total_sec > 0 else 0.0

    return {
        "total_cases": total_cases,
        "total_time_sec": round(total_sec, 4),
        "avg_case_ms": round(mean(per_case_times), 3) if per_case_times else 0.0,
        "median_case_ms": round(median(per_case_times), 3) if per_case_times else 0.0,
        "p95_case_ms": round(sorted(per_case_times)[int(0.95 * (len(per_case_times) - 1))], 3) if per_case_times else 0.0,
        "throughput_cases_per_sec": round(throughput, 2),
    }


def measure_performance_rules_only(dataset, warmup: int = 5):
    """Timing for rules-only path (no network)."""
    def run_one(case):
        cls = classify_specialty(case["input"])
        urgency = infer_urgency_score(case["input"])
        data = _build_patient_data(case["input"])
        data["specialty_hint"] = cls["specialty"]
        score_and_classify(data)
        return cls["specialty"]

    for i in range(min(warmup, len(dataset))):
        run_one(dataset[i])

    per_case_times = []
    t0 = time.perf_counter()
    for case in dataset:
        c0 = time.perf_counter()
        run_one(case)
        per_case_times.append((time.perf_counter() - c0) * 1000.0)
    total_sec = time.perf_counter() - t0
    n = len(dataset)
    return {
        "total_cases": n,
        "total_time_sec": round(total_sec, 4),
        "avg_case_ms": round(mean(per_case_times), 3) if per_case_times else 0.0,
        "median_case_ms": round(median(per_case_times), 3) if per_case_times else 0.0,
        "p95_case_ms": round(sorted(per_case_times)[int(0.95 * (len(per_case_times) - 1))], 3) if per_case_times else 0.0,
        "throughput_cases_per_sec": round(n / total_sec, 2) if total_sec > 0 else 0.0,
    }
async def run(
    gemini_client=None,
    *,
    rules_only: bool = False,
    gemini_delay: float = 4.0,
    sample: int | None = None,
):
    mode = "rules-only" if rules_only else "production (rules + Gemini fallback)"
    dataset = TEST_DATASET[:sample] if sample else TEST_DATASET

    print(f"Mode: {mode}")
    print(f"Cases: {len(dataset)}/{len(TEST_DATASET)}")
    if rules_only:
        print("Gemini: disabled (--rules-only)")
    elif gemini_client is None:
        print("Gemini: unavailable (missing API key or library) — using rules-only fallback")
    else:
        print("Gemini: enabled (real API, same path as plan_appointment)")
        if gemini_delay > 0:
            print(f"Gemini delay: {gemini_delay}s between fallback calls (rate-limit safe)")

    clinic_ok = 0
    prio_ok = 0
    wrong = []
    method_counts: Counter = Counter()
    confidence_by_method: dict[str, list[float]] = {}
    gemini_candidates = 0

    for row in dataset:
        text = row["input"]
        exp_clinic = row["expected"]["Clinic"]
        exp_prio = row["expected"]["Priority"]

        rule_only = classify_specialty(text)
        if rule_only["method"] == "default":
            gemini_candidates += 1

        result = await predict_case(row, gemini_client, gemini_delay=0.0 if rules_only else gemini_delay)
        pred_clinic = result["pred_clinic"]
        pred_prio = result["pred_prio"]
        method = result["method"]
        confidence = result["confidence"]

        method_counts[method] += 1
        confidence_by_method.setdefault(method, []).append(confidence)

        c_ok = pred_clinic == exp_clinic
        p_ok = pred_prio == exp_prio
        clinic_ok += int(c_ok)
        prio_ok += int(p_ok)

        if not (c_ok and p_ok):
            wrong.append({
                "id": row["id"],
                "text": text,
                "expected": (exp_clinic, exp_prio),
                "pred": (pred_clinic, pred_prio),
                "method": method,
                "confidence": confidence,
                "priority_score": result["priority_score"],
            })

    n = len(dataset)
    print(f"\nTotal: {n}")
    print(f"Clinic accuracy:   {clinic_ok}/{n} = {clinic_ok/n:.2%}")
    print(f"Priority accuracy: {prio_ok}/{n} = {prio_ok/n:.2%}")

    print("\n=== Classification methods ===")
    for method, count in method_counts.most_common():
        scores = confidence_by_method.get(method, [])
        avg_conf = mean(scores) if scores else 0.0
        print(f"{method}: {count}/{n} ({count/n:.1%}), avg confidence={avg_conf:.2f}")

    if not rules_only and gemini_client is not None:
        gemini_used = method_counts.get("gemini", 0)
        print(f"\nGemini fallback candidates (rule=default): {gemini_candidates}/{n}")
        print(f"Gemini successfully classified: {gemini_used}/{gemini_candidates or n}")

    if wrong:
        print("\nMismatches:")
        for w in wrong:
            print(
                f"[{w['id']}] exp={w['expected']} pred={w['pred']} "
                f"method={w['method']} confidence={w['confidence']:.2f} "
                f"priority_score={w['priority_score']:.3f} | {w['text']}"
            )

    print("\n=== Performance ===")
    if rules_only or gemini_client is None:
        perf = measure_performance_rules_only(dataset)
        print("(rules-only timing, no network)")
    else:
        perf = await measure_performance_async(
            dataset, predict_case, gemini_client, gemini_delay=gemini_delay,
        )
        print("(includes real Gemini API latency)")
    for k, v in perf.items():
        print(f"{k}: {v}")


def _resolve_gemini_client(rules_only: bool):
    if rules_only:
        return None
    try:
        from nlp.gemini_client import gemini
        return gemini if gemini._available else None
    except Exception as exc:
        print(f"Gemini init failed: {exc}")
        return None


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Benchmark scheduler on 120 Arabic cases")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Use classify_specialty only (no Gemini API calls)",
    )
    parser.add_argument(
        "--gemini-delay",
        type=float,
        default=4.0,
        help="Seconds to wait before each Gemini fallback call (default 4, free-tier safe)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only the first N cases (quick smoke test)",
    )
    args = parser.parse_args()
    client = _resolve_gemini_client(args.rules_only)
    asyncio.run(
        run(
            client,
            rules_only=args.rules_only,
            gemini_delay=max(0.0, args.gemini_delay),
            sample=args.sample,
        )
    )


if __name__ == "__main__":
    main()