"""
fsm/patient_fsm.py
Task: "Finite states to get all the data required for an appointment
       and make sure the model is list-checking the requirements loop"

States:
  GREETING → COLLECT_NAME → COLLECT_COMPLAINT → COLLECT_URGENCY
  → COLLECT_TIME → VALIDATE (loop back if missing) → CLASSIFY
  → FIND_SLOT → CONFIRM → FINALIZED | WAITLISTED
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from nlp.extractor import extract_patient_fields
from nlp.normalizer import normalize
from scheduler.priority import score_and_classify
from scheduler.classifier import classify_specialty
from nlp.gemini_client import gemini
from bot.keyboards import urgency_keyboard, time_pref_keyboard, confirm_keyboard


class State(Enum):
    GREETING        = auto()
    COLLECT_NAME    = auto()
    COLLECT_COMPLAINT = auto()
    COLLECT_URGENCY = auto()
    COLLECT_TIME    = auto()
    VALIDATE        = auto()   # requirements-loop checkpoint
    CLASSIFY        = auto()   # specialty + priority
    FIND_SLOT       = auto()
    CONFIRM         = auto()
    FINALIZED       = auto()
    WAITLISTED      = auto()
    CANCELLED       = auto()


# ── Required fields — the checklist ───────────────────────────────────────────
REQUIRED_FIELDS: list[str] = ["name", "complaint", "urgency_score", "time_pref"]

FIELD_QUESTIONS_AR: dict[str, str] = {
    "name":         "ما اسمك الكريم؟",
    "complaint":    "شو الشكوى أو سبب الزيارة؟",
    "urgency_score": "هل الموضوع عاجل، متوسط، أو روتيني؟",
    "time_pref":    "متى تحب الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}

CONFIRM_WORDS = {"نعم", "ايوه", "آيوه", "تمام", "ماشي", "اوك", "ok", "yes", "يلا", "احجز"}
CANCEL_WORDS  = {"لا", "الغي", "بدي الغي", "مش حابب", "no"}
CLARIFICATION_WORDS = {
    "فهمت", "فهمتيني", "فهمتني", "فهمت إيه", "إيه اللي فهمته", "أنت فهمت", "ماذا فهمت",
    "أنت فهمتني", "ما الذي فهمته", "اللي فهمته", "أكرر", "أعيد", "أشرح", "إيه اللي قلت",
    "ما الذي قلته", "قلت", "قلتلك", "أنت عرفت", "أنت فهمت", "أفهم",
    "what did you understand", "what do you know", "what did you get", "what do you understand"
}


@dataclass
class PatientFSM:
    user_id:   int
    state:     State = State.GREETING
    data:      dict  = field(default_factory=dict)
    slot:      Optional[object] = None      # Slot ORM object
    priority:  Optional[object] = None      # PriorityResult

    # ── Public entry ──────────────────────────────────────────────────────────

    async def handle(self, text: str) -> tuple[str, object | None]:
        """Process one message, advance state, return Arabic reply plus keyboard."""
        norm = normalize(text)
        await self._absorb(text)            # always try to extract from every message

        if self._is_clarification_request(text):
            return await self._reply_with_context(text)

        if self.state == State.GREETING:
            self.state = State.COLLECT_NAME
            return self._reply(
                "أهلاً وسهلاً 👋 أنا المساعد الذكي للحجز في العيادة.\n" + FIELD_QUESTIONS_AR["name"],
                None,
            )

        if self.state == State.COLLECT_NAME:
            if self.data.get("name"):
                self.state = State.COLLECT_COMPLAINT
                return self._reply(
                    f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"],
                    None,
                )

            # Accept a plain name message (user may just send their name)
            if text and text.strip() and not text.strip().startswith("/"):
                self.data["name"] = text.strip()
                self.state = State.COLLECT_COMPLAINT
                return self._reply(f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"], None)

            return self._reply(FIELD_QUESTIONS_AR["name"], None)

        if self.state == State.COLLECT_COMPLAINT:
            if self.data.get("complaint"):
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            complaint = await self._extract_complaint_from_text(text)
            if complaint:
                self.data["complaint"] = complaint
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            if text.strip():
                self.data["complaint"] = {
                    "raw": text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }
                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], urgency_keyboard())

            return self._reply("ممكن تخبرني أكثر عن سبب زيارتك؟", None)

        if self.state == State.COLLECT_URGENCY:
            self._absorb_urgency(norm)
            self.state = State.COLLECT_TIME
            return self._reply(FIELD_QUESTIONS_AR["time_pref"], time_pref_keyboard())

        if self.state == State.COLLECT_TIME:
            # If the user sent a time preference from the reply keyboard, map it
            mapped = self._parse_time_label(text)
            if mapped:
                self.data["time_pref"] = mapped
                return await self._run_validate()
            return await self._run_validate()

        if self.state == State.CONFIRM:
            return await self._handle_confirm(norm)

        return self._reply("عفواً، ما فهمت. ممكن تعيد؟", None)

    async def handle_callback(self, data: str) -> tuple[str, object | None]:
        if data.startswith("urgency:"):
            level = data.split(":", 1)[1]
            self._set_urgency_from_label(level)
            self.state = State.COLLECT_TIME
            return self._reply(FIELD_QUESTIONS_AR["time_pref"], time_pref_keyboard())

        if data.startswith("time:"):
            self.data["time_pref"] = self._map_time_selection(data.split(":", 1)[1])
            return await self._run_validate()

        if data.startswith("confirm:"):
            if data.endswith("yes"):
                return await self._handle_confirm("نعم")
            return await self._handle_confirm("لا")

        return self._reply("عفواً، لم أفهم اختيارك. حاول مرة أخرى.", None)

    def _reply(self, text: str, keyboard: object | None = None) -> tuple[str, object | None]:
        return text, keyboard

    async def _absorb(self, text: str):
        """Merge newly extracted fields and optionally enrich with AI."""
        extracted = extract_patient_fields(text)
        for k, v in extracted.items():
            if v is not None and not self.data.get(k):
                self.data[k] = v

        if len(text.strip()) > 20 and any(self.data.get(f) is None for f in REQUIRED_FIELDS):
            await self._try_ai_extraction(text)

    async def _extract_complaint_from_text(self, original_text: str) -> dict | None:
        """Extract complaint from text when the state is still waiting for it."""
        complaint = extract_patient_fields(original_text).get("complaint")
        if complaint:
            return complaint

        if gemini._available:
            complaint_text = await gemini.extract_missing_field(original_text, "complaint")
            if complaint_text:
                return {
                    "raw": complaint_text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }

        return None

    async def _reply_with_context(self, text: str) -> tuple[str, object | None]:
        """Use saved data + Gemini for a natural clarification answer."""
        if self.data:
            prompt = (
                f"المستخدم قال: {text}\n"
                f"البيانات المتوفرة عنه: {self.data}\n"
                f"الحالة الحالية: {self.state.name}\n"
                "اكتب ردًا عربيًا فلسطينيًا ودودًا ومباشرًا، مختصرًا، يوضح ما فهمته من المستخدم ويُظهر أن البوت remembers context."
            )
            try:
                reply = await gemini.ask(prompt, max_tokens=120)
                if reply:
                    return self._reply(reply, None)
            except Exception:
                pass

        if self.data.get("name") and self.data.get("complaint"):
            return self._reply(
                f"فهمت إن اسمك {self.data['name']}، وسبب زيارتك هو {self.data['complaint'].get('raw', 'مذكور سابقًا')}. إذا بدك، أقدر أكمّل الحجز أو أساعدك في أي خطوة أخرى.",
                None,
            )

        if self.data.get("name"):
            return self._reply(f"فهمت إن اسمك {self.data['name']}. أقدر أكمّل معك الحجز خطوة بخطوة.", None)

        return self._reply("فهمت إنك بدك مساعدة، وأنا جاهز أكمّل معك خطوة بخطوة من المعلومات اللي عندي.", None)

    def _is_clarification_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        if any(word in lowered for word in ["أنت فهمت", "فهمت", "ماذا فهمت", "إيه اللي فهمته", "أشرح", "قلت", "قلتلك", "إيه اللي قلت", "what did you understand", "what do you know", "what did you get", "what do you understand"]):
            return True
        return False

    async def _try_ai_extraction(self, text: str):
        if not gemini._available:
            return

        if not self.data.get("name"):
            name = await gemini.extract_missing_field(text, "name")
            if name:
                self.data["name"] = name.strip()

        if not self.data.get("complaint"):
            complaint = await gemini.extract_missing_field(text, "complaint")
            if complaint:
                self.data["complaint"] = {
                    "raw": complaint.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }

        if not self.data.get("urgency_score"):
            urgency = await gemini.extract_missing_field(text, "urgency")
            score = self._score_from_label(urgency)
            if score is not None:
                self.data["urgency_score"] = score

        if not self.data.get("time_pref"):
            time_pref = await gemini.extract_missing_field(text, "time_pref")
            if time_pref:
                self.data["time_pref"] = {"date": None, "phrase": time_pref.strip()}

    def _score_from_label(self, label: str | None) -> float | None:
        if not label:
            return None
        label = label.lower()
        if any(w in label for w in ["عاجل", "طارئ", "فوري", "خطر"]):
            return 0.9
        if any(w in label for w in ["متوسط", "متوسطة", "متوسط", "عادي"]):
            return 0.5
        if any(w in label for w in ["روتيني", "مش عاجل", "أي وقت"]):
            return 0.2
        return None

    def _set_urgency_from_label(self, label: str):
        if label == "P1":
            score = 0.9
            priority_class = "urgent"
        elif label == "P2":
            score = 0.5
            priority_class = "medium"
        elif label == "P3":
            score = 0.2
            priority_class = "routine"
        else:
            score = self._score_from_label(label)
            priority_class = label

        if score is not None:
            self.data["urgency_score"] = score
        self.data["priority_class"] = priority_class

    def _map_time_selection(self, selection: str) -> dict:
        from datetime import date, timedelta

        today = date.today()
        if selection == "today":
            return {"date": str(today), "phrase": "اليوم"}
        if selection == "tomorrow":
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if selection == "day_after":
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if selection == "next_week":
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        return {"date": None, "phrase": "أي وقت متاح"}

    def _parse_time_label(self, text: str) -> dict | None:
        """Map a reply-keyboard label or free text to a time_pref dict."""
        if not text:
            return None
        t = text.lower()
        from datetime import date, timedelta

        today = date.today()
        if "اليوم" in t:
            return {"date": str(today), "phrase": "اليوم"}
        if "بكرا" in t or "غدا" in t:
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if "بعد بكرا" in t:
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if "أسبوع" in t or "الأسبوع" in t:
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        if "أي وقت" in t or "لا يهم" in t or "أي وقت متاح" in t:
            return {"date": None, "phrase": "أي وقت متاح"}
        return None

    def _missing_fields(self) -> list[str]:
        """Return list of required fields that are still empty."""
        missing = []
        for f in REQUIRED_FIELDS:
            val = self.data.get(f)
            if val is None:
                missing.append(f)
            elif f == "time_pref" and isinstance(val, dict) and not (val.get("date") or val.get("phrase")):
                missing.append(f)
            elif f == "complaint" and not val:
                missing.append(f)
        return missing

    async def _run_validate(self) -> tuple[str, object | None]:
        """Check every required field. Loop back to first missing one."""
        self.state = State.VALIDATE
        missing = self._missing_fields()
        if missing:
            first_missing = missing[0]
            self.state = State(list(State)[
                list(State).index(State.COLLECT_NAME)
                + REQUIRED_FIELDS.index(first_missing)
            ])
            keyboard = None
            if first_missing == "urgency_score":
                keyboard = urgency_keyboard()
            elif first_missing == "time_pref":
                keyboard = time_pref_keyboard()
            return (
                f"بعدنا محتاجين معلومة واحدة 📋\n"
                f"{FIELD_QUESTIONS_AR[first_missing]}",
                keyboard,
            )
        return await self._classify_and_schedule()

    # ── Classify + schedule ───────────────────────────────────────────────────

    async def _classify_and_schedule(self) -> tuple[str, object | None]:
        self.state = State.CLASSIFY

        # 1. Classify specialty
        norm_complaint = normalize(self.data.get("complaint", {}).get("raw", ""))
        spec_result = classify_specialty(norm_complaint)
        self.data["specialty_hint"] = spec_result["specialty"]
        self.data["specialty_ar"]   = spec_result["specialty_ar"]

        # 2. Compute priority
        self.priority = score_and_classify(self.data)
        self.data["priority_class"] = self.priority.priority_class
        self.data["priority_score"] = self.priority.score

        self.state = State.FIND_SLOT
        return await self._find_slot()

    async def _find_slot(self) -> tuple[str, object | None]:
        from database.db import get_db
        from database import crud
        with get_db() as db:
            slot = crud.find_next_available_slot(
                db,
                specialty=self.data.get("specialty_hint", "general_practice"),
                priority_class=self.priority.priority_class,
                preferred_date=self.data.get("time_pref", {}).get("date"),
            )

        if not slot:
            self.state = State.WAITLISTED
            return (
                "عفواً، ما في مواعيد متاحة حالياً في هذا الاختصاص. 😔\n"
                "تم إضافتك لقائمة الانتظار وسنتواصل معك بأقرب وقت.",
                None,
            )

        self.slot = slot
        self.state = State.CONFIRM
        dt = slot.slot_datetime.strftime("%A، %d/%m/%Y — %H:%M")
        return (
            f"وجدت موعد مناسب! 📅\n\n"
            f"📆 {dt}\n"
            f"🏥 التخصص: {self.data.get('specialty_ar', '')}\n"
            f"{self.priority.label_ar}\n\n"
            f"تأكد الحجز؟",
            confirm_keyboard(),
        )

    async def _handle_confirm(self, norm: str) -> tuple[str, object | None]:
        if any(w in norm for w in CONFIRM_WORDS):
            await self._finalize()
            self.state = State.FINALIZED
            return (
                "✅ تم تأكيد حجزك!\n"
                f"📆 {self.slot.slot_datetime.strftime('%A، %d/%m/%Y — %H:%M')}\n"
                "سنذكّرك قبل الموعد. نتمنى لك الشفاء 🌿",
                None,
            )
        if any(w in norm for w in CANCEL_WORDS):
            self.state = State.CANCELLED
            return ("تم الإلغاء. إذا احتجت أي شيء، أنا هون. 👋", None)
        return ("اكتب نعم لتأكيد الحجز، أو لا للإلغاء.", confirm_keyboard())

    async def _finalize(self):
        from database.db import get_db
        from database import crud
        with get_db() as db:
            patient = crud.get_or_create_patient(db, self.user_id, self.data.get("name"))
            crud.create_appointment(db, self.data, self.slot, patient)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _absorb_urgency(self, norm: str):
        if any(w in norm for w in ["عاجل", "طارئ", "فوري", "خطير"]):
            self.data["urgency_score"] = max(self.data.get("urgency_score", 0), 0.85)
        elif any(w in norm for w in ["روتيني", "مش عاجل", "اي وقت"]):
            self.data["urgency_score"] = min(self.data.get("urgency_score", 0.5), 0.25)
        else:
            self.data.setdefault("urgency_score", 0.4)

    def _enter(self, new_state: State, message: str) -> str:
        self.state = new_state
        return message

    @property
    def is_done(self) -> bool:
        return self.state in (State.FINALIZED, State.WAITLISTED, State.CANCELLED)
