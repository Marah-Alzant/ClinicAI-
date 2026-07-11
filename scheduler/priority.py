"""
scheduler/priority.py — Task: "Priority engine: from conversation to priority score"

Converts raw extracted patient fields into a weighted score,
then maps that score to a priority class (P1 / P2 / P3).
All weights are expert-defined and sum to 1.0.
"""
from dataclasses import dataclass
from nlp.normalizer import normalize

# ── Weight table (must sum to 1.0) ────────────────────────────────────────────
WEIGHTS = {
    "complaint":  0.35,   # f1 — what the patient has (most important signal)
    "urgency":    0.25,   # f2 — patient's stated urgency
    "followup":   0.15,   # f3 — follow-up vs new case
    "specialty":  0.15,   # f4 — specialty urgency level
    "timing":     0.10,   # f5 — how soon they want the appointment
}

# ── Priority thresholds ────────────────────────────────────────────────────────
THETA_P1 = 0.68   # score >= 0.68  →  P1 (High / عاجل)
THETA_P2 = 0.38   # score >= 0.38  →  P2 (Medium / متوسط)
                  # score <  0.38  →  P3 (Routine / روتيني)

# ── Specialty urgency levels (f4 encoding) ────────────────────────────────────
SPECIALTY_SCORES = {
    "cardiology":       1.0,
    "neurology":        0.9,
    "emergency":        1.0,
    "endocrinology":    0.6,
    "orthopedics":      0.5,
    "gynecology":       0.55,
    "ophthalmology":    0.45,
    "dermatology":      0.3,
    "dentistry":        0.35,
    "pediatrics":       0.6,
    "general_practice": 0.3,
}

# ── Timing preference encoding (f5) ───────────────────────────────────────────
TIMING_SCORES = {
    0:  1.0,   # today
    1:  0.85,  # tomorrow
    2:  0.7,   # day after tomorrow
    3:  0.55,  # within 3 days
    7:  0.4,   # this week
    14: 0.25,  # next week
    30: 0.1,   # this month
}


@dataclass
class PriorityResult:
    score:        float
    priority_class: str          # "P1" | "P2" | "P3"
    label_ar:     str            # Arabic label for bot responses
    label_color:  str            # for dashboard: red / yellow / green
    breakdown:    dict           # individual factor scores for audit


def score_and_classify(data: dict) -> PriorityResult:
    """
    Main entry point.  data = accumulated FSM fields dict.

    Expected keys (all optional — missing → 0):
        complaint      → {"urgency_score": float, "specialty": str, ...}
        urgency_score  → float 0–1  (from patient's stated urgency words)
        is_followup    → bool
        specialty_hint → str
        time_pref      → {"date": "YYYY-MM-DD"} or None
    """
    f1 = _complaint_score(data)
    f2 = _urgency_score(data)
    f3 = _followup_score(data)
    f4 = _specialty_score(data)
    f5 = _timing_score(data)

    score = (
        WEIGHTS["complaint"] * f1
        + WEIGHTS["urgency"]  * f2
        + WEIGHTS["followup"] * f3
        + WEIGHTS["specialty"] * f4
        + WEIGHTS["timing"]   * f5
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    if score >= THETA_P1:
        cls, label_ar, color = "P1", "🔴 عاجل", "red"
    elif score >= THETA_P2:
        cls, label_ar, color = "P2", "🟡 متوسط", "yellow"
    else:
        cls, label_ar, color = "P3", "🟢 روتيني", "green"

    return PriorityResult(
        score=score,
        priority_class=cls,
        label_ar=label_ar,
        label_color=color,
        breakdown={"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5},
    )


# ── Factor encoders ────────────────────────────────────────────────────────────

def _complaint_score(data: dict) -> float:
    complaint = data.get("complaint") or {}
    raw = ""
    if isinstance(complaint, dict):
        raw = complaint.get("raw", "") or ""
        base = float(complaint.get("urgency_score", 0.2))
    else:
        raw = str(complaint or "")
        base = 0.2

    text = normalize(raw)
    red_flags = [
        "وجع صدر", "الم صدر", "ضغط على الصدر", "ضيق نفس", "اختناق",
        "شلل", "تشنج", "اغماء", "نزيف", "حرق", "كسر مفتوح",
        "الم شديد", "صداع شديد", "حمى عاليه", "سكري مرتفع", "ضغط عالي",
    ]
    moderate_flags = ["دوخه", "تنميل", "كسر", "قيء", "اسهال شديد", "التهاب", "الم"]

    if any(flag in text for flag in red_flags):
        return 1.0
    if any(flag in text for flag in moderate_flags):
        return max(base, 0.6)
    return base


def _urgency_score(data: dict) -> float:
    return float(data.get("urgency_score", 0.3))


def _followup_score(data: dict) -> float:
    # Follow-ups get a small bump — they already have a relationship with the clinic
    return 0.5 if data.get("is_followup") else 0.2


def _specialty_score(data: dict) -> float:
    # Prefer the final FSM/classifier specialty over a generic complaint fallback.
    complaint = data.get("complaint") or {}
    specialty = (
        data.get("specialty_hint")
        or complaint.get("specialty")
        or "general_practice"
    )
    return SPECIALTY_SCORES.get(specialty, 0.3)


def _timing_score(data: dict) -> float:
    time_pref = data.get("time_pref") or {}
    if not time_pref.get("date"):
        return 0.3  # no preference stated

    from datetime import date
    try:
        pref_date = date.fromisoformat(time_pref["date"])
        delta = (pref_date - date.today()).days
        delta = max(delta, 0)
    except (ValueError, TypeError):
        return 0.3

    # Find the closest bracket
    for threshold in sorted(TIMING_SCORES.keys()):
        if delta <= threshold:
            return TIMING_SCORES[threshold]
    return 0.1
