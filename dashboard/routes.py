"""
dashboard/routes.py — Task: "Dashboards design in details"
FastAPI + Jinja2. Lightweight — no Streamlit overhead.
Connects to the same SQLite DB the bot writes to.
"""
from datetime import date, timedelta
from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database.db import get_db_dependency
from database import crud

router     = APIRouter()
templates  = Jinja2Templates(directory="dashboard/templates")


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db_dependency)):
    stats = crud.daily_stats(db)
    appts = crud.get_todays_queue(db)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "stats":   stats,
            "appointments": appts,
            "today":   date.today().strftime("%A، %d/%m/%Y"),
        },
    )


@router.get("/appointments", response_class=HTMLResponse)
async def appointments_page(
    request: Request,
    day: str = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target = date.fromisoformat(day) if day else date.today()
    appts  = crud.get_todays_queue(db, target)
    return templates.TemplateResponse(
        request,
        "appointments.html",
        {
            "request":      request,
            "appointments": appts,
            "target_date":  target.isoformat(),
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "prev_date":    (target - timedelta(days=1)).isoformat(),
            "next_date":    (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(
    request: Request,
    day: str = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target   = date.fromisoformat(day) if day else date.today()
    sessions = crud.get_sessions(db, target)
    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "request":      request,
            "sessions":     sessions,
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "target_date":  target.isoformat(),
            "prev_date":    (target - timedelta(days=1)).isoformat(),
            "next_date":    (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, db: Session = Depends(get_db_dependency)):
    logs = crud.get_recent_message_logs(db, limit=200)
    return templates.TemplateResponse(
        request,
        "logs.html",
        {
            "request": request,
            "logs":    logs,
        },
    )


@router.get("/conversations", response_class=HTMLResponse)
async def conversations_page(
    request: Request,
    telegram_id: int | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    conversations = crud.get_conversations(db, limit=200)
    selected = None
    messages = []
    if telegram_id is not None:
        selected = next((c for c in conversations if c.telegram_id == telegram_id), None)
        if selected is not None:
            messages = crud.get_conversation_messages(db, telegram_id, limit=200)

    return templates.TemplateResponse(
        request,
        "conversations.html",
        {
            "request": request,
            "conversations": conversations,
            "selected": selected,
            "messages": messages,
            "telegram_id": telegram_id,
        },
    )


@router.get("/patients", response_class=HTMLResponse)
async def patients_page(
    request: Request,
    q: str = Query(default=""),
    db: Session = Depends(get_db_dependency),
):
    patients = crud.search_patient(db, q) if q else []
    return templates.TemplateResponse(
        request,
        "patients.html",
        {
            "request":  request,
            "patients": patients,
            "query":    q,
        },
    )


# ── JSON API (for JS auto-refresh, no page reload) ────────────────────────────

@router.get("/api/queue/today")
async def api_queue(db: Session = Depends(get_db_dependency)):
    appts = crud.get_todays_queue(db)
    return JSONResponse([{
        "appt_id":        a.appt_id,
        "patient_name":   a.patient.name if a.patient else "—",
        "time":           a.appt_datetime.strftime("%H:%M") if a.appt_datetime else "—",
        "specialty":      a.specialty_ar or a.specialty,
        "priority_class": a.priority_class,
        "priority_score": a.priority_score,
        "complaint":      a.complaint_summary,
        "status":         a.status,
    } for a in appts])


@router.get("/api/stats/today")
async def api_stats(db: Session = Depends(get_db_dependency)):
    return JSONResponse(crud.daily_stats(db))


@router.post("/api/appointment/{appt_id}/status")
async def update_status(
    appt_id: str,
    status: str = Query(...),
    db: Session = Depends(get_db_dependency),
):
    appt = crud.update_appointment_status(db, appt_id, status)
    return JSONResponse({"ok": bool(appt)})
