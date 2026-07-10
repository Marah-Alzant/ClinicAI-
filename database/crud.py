from datetime import datetime, date, timedelta
from sqlalchemy import select, func, update, and_, or_, desc
from sqlalchemy.orm import Session
from .models import Patient, Doctor, Appointment, Session as DoctorSession, MessageLog, Slot, Conversation
from .models import PatientProfile

def _ensure_appt_id(appt_data: dict) -> str:
    return appt_data.get("appt_id") or f"appt_{int(datetime.utcnow().timestamp())}"


def get_doctor_by_telegram(db: Session, telegram_id: int):
    return db.scalar(select(Doctor).where(Doctor.telegram_id == telegram_id))


def get_or_create_patient(db: Session, telegram_id: int, name: str | None = None):
    patient = db.scalar(select(Patient).where(Patient.telegram_id == telegram_id))
    if patient:
        if name and not patient.name:
            patient.name = name
            db.add(patient)
            db.commit()
            db.refresh(patient)
        return patient
    patient = Patient(telegram_id=telegram_id, name=name)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_todays_queue(db: Session, target: date | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = select(Appointment).where(
        Appointment.appt_datetime >= start,
        Appointment.appt_datetime <= end,
    ).order_by(Appointment.appt_datetime)
    return db.scalars(stmt).all()


def daily_stats(db: Session):
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())
    total = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end)) or 0
    p1 = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end, Appointment.priority_class == "P1")) or 0
    p2 = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end, Appointment.priority_class == "P2")) or 0
    p3 = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end, Appointment.priority_class == "P3")) or 0
    completed = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end, Appointment.status == "completed")) or 0
    no_show = db.scalar(select(func.count()).select_from(Appointment).where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end, Appointment.status == "no_show")) or 0
    return {
        "total": total,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "completed": completed,
        "no_show": no_show,
    }


def get_sessions(db: Session, target: date | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = select(DoctorSession).where(
        DoctorSession.session_datetime >= start,
        DoctorSession.session_datetime <= end,
    ).order_by(desc(DoctorSession.session_datetime))
    return db.scalars(stmt).all()


def get_recent_message_logs(db: Session, limit: int = 200):
    stmt = select(MessageLog).order_by(desc(MessageLog.created_at)).limit(limit)
    return db.scalars(stmt).all()


def get_conversations(db: Session, limit: int = 200):
    stmt = select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit)
    return db.scalars(stmt).all()


def get_conversation_messages(db: Session, telegram_id: int, limit: int = 200):
    stmt = (
        select(MessageLog)
        .where(MessageLog.telegram_id == telegram_id)
        .order_by(MessageLog.created_at)
        .limit(limit)
    )
    return db.scalars(stmt).all()


def get_or_create_conversation(db: Session, telegram_id: int, username: str | None = None, first_name: str | None = None, last_name: str | None = None):
    conversation = db.scalar(select(Conversation).where(Conversation.telegram_id == telegram_id))
    if conversation:
        if username and not conversation.username:
            conversation.username = username
        if first_name and not conversation.first_name:
            conversation.first_name = first_name
        if last_name and not conversation.last_name:
            conversation.last_name = last_name
        conversation.updated_at = datetime.utcnow()
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    conversation = Conversation(
        telegram_id=telegram_id,
        username=username,
        first_name=first_name,
        last_name=last_name,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def search_patient(db: Session, q: str):
    stmt = select(Patient).where(Patient.name.ilike(f"%{q}%"))
    return db.scalars(stmt).all()


def update_appointment_status(db: Session, appt_id: str, status: str):
    appt = db.scalar(select(Appointment).where(Appointment.appt_id == appt_id))
    if not appt:
        return None
    appt.status = status
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return appt


def find_next_available_slot(db: Session, specialty: str, priority_class: str, preferred_date: str | None = None):
    stmt = select(Slot).where(
        Slot.specialty == specialty,
        Slot.status == "available",
    )
    if preferred_date:
        try:
            from datetime import datetime
            start = datetime.fromisoformat(preferred_date)
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
            stmt = stmt.where(Slot.slot_datetime >= start, Slot.slot_datetime <= end)
        except ValueError:
            pass
    stmt = stmt.order_by(Slot.slot_datetime)
    slot = db.scalar(stmt)
    if slot:
        return slot

    fallback = select(Slot).where(Slot.status == "available").order_by(Slot.slot_datetime)
    return db.scalar(fallback)


def create_appointment(db: Session, data: dict, slot: object, patient: Patient):
    from scheduler.scheduler import AppointmentSlot, book_slot
    appt_id = _ensure_appt_id(data)
    if isinstance(slot, AppointmentSlot):
        appt_datetime = slot.slot_datetime
    else:
        appt_datetime = getattr(slot, "slot_datetime", None)
    status = "confirmed" if appt_datetime else "waitlisted"
    appointment = Appointment(
        appt_id=appt_id,
        patient_id=patient.patient_id,
        appt_datetime=appt_datetime,
        specialty=data.get("specialty_hint") or data.get("specialty"),
        priority_class=data.get("priority_class"),
        status=status,
    )
    # mark slot as booked when we actually reserved a slot
    if slot is not None and appt_datetime is not None:
        try:
            if isinstance(slot, AppointmentSlot):
                book_slot(db, slot)
            else:
                slot.status = "booked"
                db.add(slot)
                db.commit()
        except Exception:
            pass
    db.add(appointment)
    db.commit()
    db.refresh(appointment)
    return appointment


def create_session(db: Session, session_data: dict, doctor_id: int):
    session = DoctorSession(
        doctor_id=doctor_id,
        patient_name=session_data.get("patient_name"),
        chief_complaint=session_data.get("chief_complaint"),
        diagnosis=session_data.get("diagnosis"),
        medications=session_data.get("medications"),
        investigations=session_data.get("investigations"),
        followup_days=session_data.get("followup_days"),
        raw_transcription=session_data.get("raw_transcription"),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def log_message(db: Session, telegram_id: int, direction: str, message_type: str, content: str):
    log = MessageLog(
        telegram_id=telegram_id,
        direction=direction,
        message_type=message_type,
        content=content,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def log_bot_reply(db: Session, telegram_id: int, content: str):
    return log_message(db, telegram_id, "outbound", "bot_reply", content)


def get_profile(db: Session, telegram_id: int) -> dict | None:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        return None
    return profile.data or {}


def upsert_profile(db: Session, telegram_id: int, data: dict) -> object:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        profile = PatientProfile(telegram_id=telegram_id, data=data)
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile

    # merge existing data with new fields (new keys override old)
    existing = profile.data or {}
    merged = {**existing, **(data or {})}
    profile.data = merged
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def delete_profile(db: Session, telegram_id: int) -> bool:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        return False
    db.delete(profile)
    db.commit()
    return True
