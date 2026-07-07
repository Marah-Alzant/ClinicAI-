from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Text, ForeignKey, Boolean, JSON
)
from sqlalchemy.orm import relationship
from .db import Base


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    appointments = relationship("Appointment", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="doctor")


class Appointment(Base):
    __tablename__ = "appointments"

    appt_id = Column(String(64), primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True)
    appt_datetime = Column(DateTime, nullable=True)
    specialty = Column(String(128), nullable=True)
    priority_class = Column(String(8), nullable=True)
    status = Column(String(32), nullable=False, default="confirmed")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="appointments")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False)
    patient_name = Column(String(255), nullable=True)
    chief_complaint = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=True)
    medications = Column(JSON, nullable=True)
    investigations = Column(JSON, nullable=True)
    followup_days = Column(Integer, nullable=True)
    raw_transcription = Column(Text, nullable=True)
    session_datetime = Column(DateTime, default=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="sessions")


class Slot(Base):
    __tablename__ = "slots"

    slot_id = Column(Integer, primary_key=True, autoincrement=True)
    slot_datetime = Column(DateTime, nullable=False)
    specialty = Column(String(128), nullable=False, default="general_practice")
    priority_class = Column(String(8), nullable=True)
    status = Column(String(32), nullable=False, default="available")
    notes = Column(Text, nullable=True)


class MessageLog(Base):
    __tablename__ = "message_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, nullable=False)
    direction = Column(String(16), nullable=False)
    message_type = Column(String(32), nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    profile_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
