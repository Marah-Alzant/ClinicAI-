"""
scheduler/scheduler.py — Tier-1 booking orchestration (real-time path only).

Pipeline:
  request → classify → priority → fetch slots → block rules → wave rules
         → rank/select OR waitlist

GA (Tier 2) and Monte Carlo (Tier 3) stay outside this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import crud
from database.models import Appointment, PatientProfile, Slot
from scheduler.classifier import classify_specialty, classify_with_gemini_fallback
from scheduler.priority import score_and_classify

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PRIORITY_CLASSES = frozenset({"P1", "P2", "P3"})
APPOINTMENT_STATUSES = frozenset(
    {"confirmed", "waitlisted", "completed", "no_show", "cancelled"}
)
FALLBACK_SPECIALTY = "general_practice"

BLOCK_ACCESS = {
    "P1": {"P1","P2","P3",None},
    "P2": {"P2","P3",None},
    "P3": {"P3",None}
}

WAVE_HORIZON_DAYS: dict[str, int] = {
    "P1": 2,
    "P2": 7,
    "P3": 30,
}


# ── Types (read-only DTOs) ────────────────────────────────────────────────────

def _parse_doctor_id(notes: str | None) -> int | None:
    if not notes:
        return None
    for part in notes.split():
        if part.startswith("doctor_id:"):
            try:
                return int(part.split(":", 1)[1])
            except (TypeError, ValueError):
                return None
    return None


@dataclass(frozen=True)
class AppointmentSlot:
    """Immutable slot view. DB writes go through book_slot()."""

    slot_id: int
    slot_datetime: datetime
    specialty: str
    priority_class: str | None
    status: str
    doctor_id: int | None = None
    notes: str | None = None

    @classmethod
    def from_orm(cls, row: Slot) -> AppointmentSlot:
        return cls(
            slot_id=row.slot_id,
            slot_datetime=row.slot_datetime,
            specialty=row.specialty,
            priority_class=row.priority_class,
            status=row.status,
            doctor_id=_parse_doctor_id(row.notes),
            notes=row.notes,
        )


@dataclass(frozen=True)
class WaitlistCandidate:
    specialty: str
    priority_class: str
    priority_score: float
    urgency_score: float
    arrival_time: datetime
    telegram_id: int | None = None


@dataclass
class WaitlistEntry:
    specialty: str
    priority_class: str
    priority_score: float
    urgency_score: float
    arrival_time: datetime
    position: int
    estimated_note: str


@dataclass
class ScheduleDecision:
    specialty: str
    specialty_ar: str
    classification_method: str
    classification_confidence: float

    priority_class: str
    priority_score: float
    priority_label_ar: str
    priority_color: str

    slot: Optional[AppointmentSlot]
    waitlisted: bool
    waitlist: Optional[WaitlistEntry] = None


# ── Input helpers ─────────────────────────────────────────────────────────────

def sanitize_input(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data or {})

    complaint = data.get("complaint")
    if not isinstance(complaint, dict):
        complaint = {"raw": str(complaint or "").strip()}
    complaint.setdefault("raw", "")
    data["complaint"] = complaint

    time_pref = data.get("time_pref")
    if not isinstance(time_pref, dict):
        data["time_pref"] = {"date": None, "phrase": "أي وقت متاح"}
    else:
        time_pref.setdefault("date", None)
        time_pref.setdefault("phrase", "أي وقت متاح")

    try:
        if "urgency_score" in data and data["urgency_score"] is not None:
            data["urgency_score"] = float(data["urgency_score"])
    except (TypeError, ValueError):
        data["urgency_score"] = 0.3

    if "arrival_time" not in data:
        data["arrival_time"] = datetime.utcnow().isoformat()

    return data


def normalize_priority_class(raw: str | None) -> str:
    val = (raw or "").strip().upper()
    return val if val in PRIORITY_CLASSES else "P2"


# ── Metrics ───────────────────────────────────────────────────────────────────

def clinic_load_by_day(db: Session, specialty: str) -> dict[tuple[str, date], int]:
    stmt = (
        select(Appointment.specialty, func.date(Appointment.appt_datetime), func.count())
        .where(
            Appointment.status == "confirmed",
            Appointment.appt_datetime.isnot(None),
            Appointment.specialty == specialty,
        )
        .group_by(Appointment.specialty, func.date(Appointment.appt_datetime))
    )
    return {(spec, day): count for spec, day, count in db.execute(stmt).all()}


def doctor_load_by_day(db: Session) -> dict[tuple[int, date], int]:
    loads: dict[tuple[int, date], int] = {}
    stmt = select(Slot).where(Slot.status == "booked")
    for row in db.scalars(stmt).all():
        doc_id = _parse_doctor_id(row.notes)
        if doc_id is None:
            continue
        key = (doc_id, row.slot_datetime.date())
        loads[key] = loads.get(key, 0) + 1
    return loads


def slot_utilization_by_day(db: Session, specialty: str) -> dict[tuple[str, date], float]:
    total_stmt = (
        select(func.date(Slot.slot_datetime), func.count())
        .where(Slot.specialty == specialty)
        .group_by(func.date(Slot.slot_datetime))
    )
    booked_stmt = (
        select(func.date(Slot.slot_datetime), func.count())
        .where(Slot.specialty == specialty, Slot.status == "booked")
        .group_by(func.date(Slot.slot_datetime))
    )
    totals = {day: cnt for day, cnt in db.execute(total_stmt).all()}
    booked = {day: cnt for day, cnt in db.execute(booked_stmt).all()}

    util: dict[tuple[str, date], float] = {}
    for day, total in totals.items():
        b = booked.get(day, 0)
        util[(specialty, day)] = (b / total) if total else 0.0
    return util


# ── Slots ─────────────────────────────────────────────────────────────────────

def get_available_slots(db: Session, specialty: str) -> list[AppointmentSlot]:
    """
    Fetch available slots for the requested specialty.
    Fallback: general_practice only — never random specialties.
    """
    rows = _query_slots(db, specialty)
    if rows:
        return [AppointmentSlot.from_orm(r) for r in rows]

    if specialty != FALLBACK_SPECIALTY:
        rows = _query_slots(db, FALLBACK_SPECIALTY)
        if rows:
            return [AppointmentSlot.from_orm(r) for r in rows]

    return []


def _query_slots(db: Session, specialty: str) -> list[Slot]:
    stmt = (
        select(Slot)
        .where(Slot.specialty == specialty, Slot.status == "available")
        .order_by(Slot.slot_datetime)
    )
    return list(db.scalars(stmt).all())


def apply_block_rules(
    slots: list[AppointmentSlot],
    priority_class: str,
) -> list[AppointmentSlot]:
    allowed = BLOCK_ACCESS.get(priority_class, {None})
    return [s for s in slots if s.priority_class in allowed]


def apply_wave_rules(
    slots: list[AppointmentSlot],
    priority_class: str,
    *,
    now: datetime | None = None,
) -> list[AppointmentSlot]:
    now = now or datetime.utcnow()
    horizon_days = WAVE_HORIZON_DAYS.get(priority_class, 7)
    wave_end = now + timedelta(days=horizon_days)
    filtered = [s for s in slots if now <= s.slot_datetime <= wave_end]
    return filtered or slots


def book_slot(db: Session, slot: AppointmentSlot) -> None:
    """Service-layer write: mark a slot as booked in the database."""
    row = db.get(Slot, slot.slot_id)
    if row is None:
        raise ValueError(f"Slot {slot.slot_id} not found")
    if row.status != "available":
        raise ValueError(f"Slot {slot.slot_id} is not available")
    row.status = "booked"
    db.add(row)
    db.commit()


# ── Ranking ───────────────────────────────────────────────────────────────────

def _build_slot_sort_key(
    slot: AppointmentSlot,
    *,
    pref_day: date | None,
    priority_class: str,
    clinic_load: dict[tuple[str, date], int],
    doctor_load: dict[tuple[int, date], int],
    utilization: dict[tuple[str, date], float],
) -> tuple:
    day = slot.slot_datetime.date()
    day_match = 0 if (pref_day is None or day == pref_day) else 1
    block_match = 0 if slot.priority_class == priority_class else 1
    c_load = clinic_load.get((slot.specialty, day), 0)
    d_load = doctor_load.get((slot.doctor_id, day), 0) if slot.doctor_id else 0
    util = utilization.get((slot.specialty, day), 0.0)

    # P1: earliest slot wins even on busier days; P2/P3: spread load first
    if priority_class == "P1":
        return (day_match, block_match, slot.slot_datetime, c_load, d_load, util)
    return (day_match, block_match, c_load, d_load, util, slot.slot_datetime)


def rank_slots(
    db: Session,
    slots: list[AppointmentSlot],
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    preferred_date: str | None = None,
) -> list[AppointmentSlot]:
    pref_day = _parse_preferred_date(preferred_date)
    clinic_load = clinic_load_by_day(db, specialty)
    doctor_load = doctor_load_by_day(db)
    utilization = slot_utilization_by_day(db, specialty)

    return sorted(
        slots,
        key=lambda s: _build_slot_sort_key(
            s,
            pref_day=pref_day,
            priority_class=priority_class,
            clinic_load=clinic_load,
            doctor_load=doctor_load,
            utilization=utilization,
        ),
    )


def select_best_slot(
    db: Session,
    slots: list[AppointmentSlot],
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    preferred_date: str | None = None,
) -> Optional[AppointmentSlot]:
    ranked = rank_slots(
        db,
        slots,
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        preferred_date=preferred_date,
    )
    return ranked[0] if ranked else None


def _parse_preferred_date(preferred_date: str | None) -> date | None:
    if not preferred_date:
        return None
    try:
        return date.fromisoformat(str(preferred_date))
    except (TypeError, ValueError):
        return None


# ── Waitlist ──────────────────────────────────────────────────────────────────

def _queue_sort_key(c: WaitlistCandidate) -> tuple:
    return (-c.priority_score, c.arrival_time, -c.urgency_score)


def _candidate_from_profile(specialty: str, profile: PatientProfile) -> WaitlistCandidate | None:
    data = profile.data or {}
    wl = data.get("waitlist")
    if not isinstance(wl, dict) or wl.get("specialty") != specialty:
        return None
    try:
        return WaitlistCandidate(
            specialty=specialty,
            priority_class=str(wl.get("priority_class", "P2")),
            priority_score=float(wl.get("priority_score", 0.0)),
            urgency_score=float(wl.get("urgency_score", 0.0)),
            arrival_time=datetime.fromisoformat(str(wl["arrival_time"])),
            telegram_id=profile.telegram_id,
        )
    except (TypeError, ValueError, KeyError):
        return None


def _load_queue(db: Session, specialty: str) -> list[WaitlistCandidate]:
    queue: list[WaitlistCandidate] = []
    profiles = list(db.scalars(select(PatientProfile)).all())
    profile_by_tg = {p.telegram_id: p for p in profiles}

    stmt = select(Appointment).where(
        Appointment.status == "waitlisted",
        Appointment.specialty == specialty,
    )
    for appt in db.scalars(stmt).all():
        patient = appt.patient
        tg_id = patient.telegram_id if patient else None
        profile = profile_by_tg.get(tg_id) if tg_id else None

        if profile:
            cand = _candidate_from_profile(specialty, profile)
            if cand:
                queue.append(cand)
                continue

        queue.append(
            WaitlistCandidate(
                specialty=specialty,
                priority_class=appt.priority_class or "P3",
                priority_score={"P1": 0.9, "P2": 0.5, "P3": 0.2}.get(
                    appt.priority_class or "P3", 0.2
                ),
                urgency_score=0.3,
                arrival_time=appt.created_at or datetime.utcnow(),
                telegram_id=tg_id,
            )
        )

    queue.sort(key=_queue_sort_key)
    return queue


def compute_waitlist_position(db: Session, candidate: WaitlistCandidate) -> int:
    queue = _load_queue(db, candidate.specialty)
    queue.append(candidate)
    queue.sort(key=_queue_sort_key)
    for idx, item in enumerate(queue, start=1):
        if (
            item.telegram_id == candidate.telegram_id
            and item.arrival_time == candidate.arrival_time
            and item.priority_score == candidate.priority_score
        ):
            return idx
    return len(queue)


def persist_waitlist_metadata(
    db: Session,
    *,
    telegram_id: int,
    candidate: WaitlistCandidate,
    position: int,
) -> None:
    crud.upsert_profile(
        db,
        telegram_id,
        {
            "waitlist": {
                "specialty": candidate.specialty,
                "priority_class": candidate.priority_class,
                "priority_score": candidate.priority_score,
                "urgency_score": candidate.urgency_score,
                "arrival_time": candidate.arrival_time.isoformat(),
                "position": position,
            }
        },
    )


def enqueue_waitlist(
    db: Session,
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    urgency_score: float,
    arrival_time: datetime | None = None,
    telegram_id: int | None = None,
) -> WaitlistEntry:
    arrival = arrival_time or datetime.utcnow()
    candidate = WaitlistCandidate(
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        urgency_score=urgency_score,
        arrival_time=arrival,
        telegram_id=telegram_id,
    )
    position = compute_waitlist_position(db, candidate)

    if telegram_id is not None:
        persist_waitlist_metadata(
            db, telegram_id=telegram_id, candidate=candidate, position=position
        )

    return WaitlistEntry(
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        urgency_score=urgency_score,
        arrival_time=arrival,
        position=position,
        estimated_note=(
            f"أنت رقم {position} في قائمة الانتظار "
            f"({specialty}, أولوية {priority_score:.2f})"
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _classify(data: dict[str, Any], gemini_client: Optional[object]) -> dict[str, Any]:
    complaint_text = data.get("complaint", {}).get("raw", "") or ""

    if gemini_client is not None:
        try:
            return await classify_with_gemini_fallback(complaint_text, gemini_client)
        except Exception as exc:
            logger.exception("Gemini classification failed; using rule-based classifier: %s", exc)

    return classify_specialty(complaint_text)

async def plan_appointment(
    data: dict[str, Any],
    db: Session,
    gemini_client: Optional[object] = None,
) -> ScheduleDecision:
    safe_data = sanitize_input(data)

    spec_result = await _classify(safe_data, gemini_client)
    safe_data["specialty_hint"] = spec_result["specialty"]
    safe_data["specialty_ar"] = spec_result["specialty_ar"]

    pr = score_and_classify(safe_data)
    priority_class = normalize_priority_class(pr.priority_class)
    priority_score = pr.score
    safe_data["priority_class"] = priority_class
    safe_data["priority_score"] = priority_score

    specialty = safe_data.get("specialty_hint", "general_practice")
    preferred_date = safe_data.get("time_pref", {}).get("date")
    urgency_score = float(safe_data.get("urgency_score", 0.3))

    try:
        arrival_time = datetime.fromisoformat(str(safe_data.get("arrival_time")))
    except (TypeError, ValueError):
        arrival_time = datetime.utcnow()

    telegram_id = safe_data.get("telegram_id") or safe_data.get("user_id")

    available = get_available_slots(db, specialty)
    available = apply_block_rules(available, priority_class)
    available = apply_wave_rules(available, priority_class)
    slot = select_best_slot(
        db,
        available,
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        preferred_date=preferred_date,
    )

    waitlist = None
    if slot is None:
        waitlist = enqueue_waitlist(
            db,
            specialty=specialty,
            priority_class=priority_class,
            priority_score=priority_score,
            urgency_score=urgency_score,
            arrival_time=arrival_time,
            telegram_id=int(telegram_id) if telegram_id is not None else None,
        )
        logger.info(
            "Waitlisted: specialty=%s score=%.3f position=%s",
            specialty,
            priority_score,
            waitlist.position,
        )

    return ScheduleDecision(
        specialty=specialty,
        specialty_ar=safe_data.get("specialty_ar", "الطب العام"),
        classification_method=str(spec_result.get("method", "unknown")),
        classification_confidence=float(spec_result.get("confidence", 0.0)),
        priority_class=priority_class,
        priority_score=priority_score,
        priority_label_ar=pr.label_ar,
        priority_color=pr.label_color,
        slot=slot,
        waitlisted=(slot is None),
        waitlist=waitlist,
    )
