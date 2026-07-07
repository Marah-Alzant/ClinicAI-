import os
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Date, Integer, String, Text, ForeignKey, select, func, and_, or_, update
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine

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

# ── Models are imported here to register with SQLAlchemy metadata.
from . import models  # noqa: E402, F401


def init_db():
    """Create database file and tables if they do not exist."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


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
