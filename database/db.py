import os
from contextlib import contextmanager
from datetime import datetime, date, timedelta, time
from pathlib import Path

from sqlalchemy import create_engine, text, inspect, select, func
from sqlalchemy.orm import sessionmaker, declarative_base

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "clinic.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Models are imported here to register with SQLAlchemy metadata.
from . import models  # noqa: E402, F401


def _sqlite_column_exists(table_name: str, column_name: str) -> bool:
    inspector = inspect(engine)
    try:
        return column_name in [col["name"] for col in inspector.get_columns(table_name)]
    except Exception:
        return False


def _sqlite_add_column(table_name: str, column_name: str, column_sql: str) -> None:
    """Small additive migration for existing demo SQLite DB files."""
    if _sqlite_column_exists(table_name, column_name):
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))


def migrate_sqlite_schema() -> None:
    """
    Keep the bundled SQLite DB compatible with the newer ORM models.
    This project does not use Alembic, so we only do safe additive changes.
    """
    if not DB_FILE.exists():
        return

    # patients
    _sqlite_add_column("patients", "phone", "VARCHAR(64)")
    _sqlite_add_column("patients", "updated_at", "DATETIME")

    # doctors
    _sqlite_add_column("doctors", "specialty", "VARCHAR(128)")

    # appointments
    _sqlite_add_column("appointments", "slot_id", "INTEGER")
    _sqlite_add_column("appointments", "specialty_ar", "VARCHAR(128)")
    _sqlite_add_column("appointments", "priority_score", "FLOAT")
    _sqlite_add_column("appointments", "complaint_summary", "TEXT")
    _sqlite_add_column("appointments", "time_preference", "JSON")
    _sqlite_add_column("appointments", "updated_at", "DATETIME")

    # slots
    _sqlite_add_column("slots", "created_at", "DATETIME")
    _sqlite_add_column("slots", "updated_at", "DATETIME")

    # sessions
    _sqlite_add_column("sessions", "patient_id", "INTEGER")
    _sqlite_add_column("sessions", "appointment_id", "VARCHAR(64)")

    # patient_profiles
    _sqlite_add_column("patient_profiles", "patient_id", "INTEGER")

    # conversations / message logs linkage
    _sqlite_add_column("conversations", "patient_id", "INTEGER")
    _sqlite_add_column("conversations", "role", "VARCHAR(32) DEFAULT 'patient'")
    _sqlite_add_column("message_logs", "conversation_id", "INTEGER")
    _sqlite_add_column("message_logs", "patient_id", "INTEGER")


def seed_default_slots(days: int = 14) -> None:
    """
    Demo-safe slot seeding: only runs when slots table is empty.
    Real clinics can later replace this with an admin slot-management screen.
    """
    from .models import Slot

    with SessionLocal() as db:
        existing = db.scalar(select(func.count()).select_from(Slot)) or 0
        if existing > 0:
            return

        specialties = [
            "general_practice",
            "cardiology",
            "neurology",
            "orthopedics",
            "pediatrics",
            "gynecology",
            "dentistry",
            "dermatology",
        ]
        # Priority gates reserve early capacity for urgent cases while leaving general slots.
        slot_plan = [
            (time(9, 0), "P1"),
            (time(9, 30), "P1"),
            (time(10, 0), "P2"),
            (time(10, 30), "P2"),
            (time(11, 0), None),
            (time(11, 30), None),
            (time(12, 0), "P3"),
            (time(12, 30), "P3"),
            (time(13, 0), None),
            (time(13, 30), None),
        ]

        today = date.today()
        created = 0
        for offset in range(days):
            day = today + timedelta(days=offset)
            if day.weekday() == 4:  # Friday off in many local clinics
                continue
            for specialty in specialties:
                for slot_time, priority_class in slot_plan:
                    db.add(Slot(
                        slot_datetime=datetime.combine(day, slot_time),
                        specialty=specialty,
                        priority_class=priority_class,
                        status="available",
                        notes="auto-seeded demo slot",
                    ))
                    created += 1
        db.commit()


def init_db():
    """Create database file/tables, migrate safe columns, and seed demo slots if needed."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    migrate_sqlite_schema()
    seed_default_slots()


@contextmanager
def get_db():
    """Context manager for a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_dependency():
    """FastAPI dependency that yields a DB session."""
    with get_db() as db:
        yield db
