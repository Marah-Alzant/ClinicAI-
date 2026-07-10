import pytest
from datetime import date, timedelta

from scheduler.classifier import classify_specialty, classify_with_gemini_fallback
from scheduler.priority import score_and_classify


# ---------------------------
# Helpers
# ---------------------------

def make_data(
    complaint_raw="شكوى عامة",
    complaint_urgency=0.3,
    complaint_specialty="general_practice",
    urgency_score=0.3,
    is_followup=False,
    specialty_hint="general_practice",
    days_ahead=None,
    time_pref_override=None,
):
    if time_pref_override is not None:
        time_pref = time_pref_override
    else:
        if days_ahead is None:
            time_pref = {"date": None, "phrase": "أي وقت"}
        else:
            d = date.today() + timedelta(days=days_ahead)
            time_pref = {"date": str(d), "phrase": "موعد"}

    return {
        "complaint": {
            "raw": complaint_raw,
            "urgency_score": complaint_urgency,
            "specialty": complaint_specialty,
        },
        "urgency_score": urgency_score,
        "is_followup": is_followup,
        "specialty_hint": specialty_hint,
        "time_pref": time_pref,
    }


class FakeGemini:
    def __init__(self, answer=None, raise_exc=False):
        self.answer = answer
        self.raise_exc = raise_exc

    async def ask(self, prompt, max_tokens=20):
        if self.raise_exc:
            raise RuntimeError("Gemini unavailable")
        return self.answer


# ---------------------------
# CLASSIFIER TESTS (many)
# ---------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("الم شديد في الصدر وخفقان", "cardiology"),
        ("دوخة وتنميل وصداع قوي", "neurology"),
        ("سكر وضغط ومتابعة مزمنة", "endocrinology"),  # حسب ترتيب قواعدك الحالية
        ("الم في الركبة والظهر", "orthopedics"),
        ("مشكلة في الدورة الشهرية", "gynecology"),
        ("طفلي حرارته مرتفعة", "pediatrics"),
        ("احمرار في العين وضعف نظر", "ophthalmology"),
        ("طفح جلدي وحكة", "dermatology"),
        ("وجع ضرس والتهاب لثة", "dentistry"),
        ("ضيق تنفس وكحة مزمنة", "pulmonology"),
        ("مغص بالمعدة وإسهال", "gastroenterology"),
        ("الم في الاذن والتهاب حلق", "ent"),
        ("توتر شديد واكتئاب", "psychiatry"),
        ("والدي المسن يحتاج متابعة", "elderly"),
        ("فحص عام واعراض بسيطة", "general_practice"),
    ],
)
def test_classify_specialty_rules(text, expected):
    result = classify_specialty(text)
    assert result["specialty"] == expected
    assert "specialty_ar" in result
    assert "method" in result
    assert "confidence" in result


def test_classify_default_when_no_match():
    result = classify_specialty("كلام غير طبي تماما")
    assert result["specialty"] == "general_practice"
    assert result["method"] in ("default", "rule")


@pytest.mark.asyncio
async def test_classify_with_gemini_fallback_success():
    gemini = FakeGemini(answer="cardiology")
    result = await classify_with_gemini_fallback("وصف غامض جدا", gemini)
    assert result["specialty"] in {"cardiology", "general_practice"}  # حسب rule/default قبل fallback
    assert "method" in result


@pytest.mark.asyncio
async def test_classify_with_gemini_fallback_unknown_key():
    gemini = FakeGemini(answer="unknown_specialty")
    result = await classify_with_gemini_fallback("وصف غامض جدا", gemini)
    # يجب ألا ينهار
    assert "specialty" in result
    assert "method" in result


@pytest.mark.asyncio
async def test_classify_with_gemini_fallback_exception():
    gemini = FakeGemini(raise_exc=True)
    result = await classify_with_gemini_fallback("وصف غامض جدا", gemini)
    # يجب fallback بدون crash
    assert "specialty" in result
    assert "method" in result


# ---------------------------
# PRIORITY TESTS (many)
# ---------------------------

@pytest.mark.parametrize(
    "data,expected_class",
    [
        (make_data("الم صدر حاد", 0.95, "cardiology", 0.95, False, "cardiology", 0), "P1"),
        (make_data("دوخة قوية", 0.75, "neurology", 0.7, False, "neurology", 1), "P1"),
        (make_data("مراجعة متابعة", 0.4, "general_practice", 0.4, True, "general_practice", 7), "P2"),
        (make_data("حكة جلد بسيطة", 0.2, "dermatology", 0.2, False, "dermatology", 14), "P3"),
        (make_data("تنظيف اسنان", 0.15, "dentistry", 0.1, False, "dentistry", 30), "P3"),
    ],
)
def test_priority_classification(data, expected_class):
    r = score_and_classify(data)
    assert r.priority_class == expected_class
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.breakdown, dict)


def test_priority_handles_bad_urgency_type():
    data = make_data(urgency_score="bad-value")
    r = score_and_classify(data)
    assert 0.0 <= r.score <= 1.0


@pytest.mark.parametrize(
    "time_pref",
    [
        None,
        {},
        {"date": None},
        {"date": "invalid-date"},
        "not-a-dict",
        123,
    ],
)
def test_priority_handles_invalid_time_pref(time_pref):
    data = make_data(time_pref_override=time_pref)
    r = score_and_classify(data)
    assert 0.0 <= r.score <= 1.0


def test_priority_with_past_date_does_not_crash():
    past = str(date.today() - timedelta(days=3))
    data = make_data(time_pref_override={"date": past, "phrase": "امس"})
    r = score_and_classify(data)
    assert 0.0 <= r.score <= 1.0


def test_priority_without_complaint_dict():
    data = {
        "complaint": "string-instead-of-dict",
        "urgency_score": 0.3,
        "is_followup": False,
        "specialty_hint": "general_practice",
        "time_pref": {"date": None, "phrase": "أي وقت"},
    }
    r = score_and_classify(data)
    assert 0.0 <= r.score <= 1.0


@pytest.mark.parametrize("specialty", ["cardiology", "neurology", "general_practice", "unknown"])
def test_priority_specialty_variants(specialty):
    data = make_data(complaint_specialty=specialty, specialty_hint=specialty)
    r = score_and_classify(data)
    assert 0.0 <= r.score <= 1.0


def test_priority_breakdown_keys_present():
    data = make_data()
    r = score_and_classify(data)
    assert set(r.breakdown.keys()) == {"f1", "f2", "f3", "f4", "f5"}