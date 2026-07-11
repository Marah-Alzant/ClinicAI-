"""Scheduler package — Tier-1 real-time booking."""

from scheduler.scheduler import (
    AppointmentSlot,
    ScheduleDecision,
    WaitlistEntry,
    book_slot,
    plan_appointment,
)

__all__ = [
    "plan_appointment",
    "AppointmentSlot",
    "ScheduleDecision",
    "WaitlistEntry",
    "book_slot",
]
