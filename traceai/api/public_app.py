"""Public-facing API: police-published appeals (read-only) and a submit-only tip form.

This process deliberately contains no officer routes, no lead scores and no case data beyond what an
officer chose to publish. A tip returns only a receipt reference: the reporter never learns whether it
matched anything, which is what stops the form being used to probe or locate someone.

Run: uvicorn traceai.api.public_app:app
"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from datetime import date, timedelta

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from traceai import config, db, pipeline
from traceai.api import security
from traceai.api.images import store_image
from traceai.api.ratelimit import RateLimiter
from traceai.db import MissingPerson, Sighting, utcnow
from traceai.geo import gazetteer

SHORT_WINDOW = RateLimiter(config.TIP_RATE_PER_10MIN, 600)
DAY_WINDOW = RateLimiter(config.TIP_RATE_PER_DAY, 86400)

NOTICE = (
    "Only report what you personally saw. Do not approach or follow anyone. If someone is in danger, "
    "call 000. Knowingly making a false report to police is an offence."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="TraceAI public portal", version="0.2.0", lifespan=lifespan, docs_url=None, redoc_url=None)


class AppealOut(BaseModel):
    """Everything the public can ever see about a case, and only if an officer published it."""

    id: int
    name: str
    age: int | None
    summary: str
    clothing: list[str]
    last_seen_city: str  # deliberately coarse: never the exact place or coordinates
    last_seen_date: date
    has_photo: bool


class TipReceipt(BaseModel):
    reference: str
    message: str


def _get_db():
    yield from security.get_db()


def _appeal(p: MissingPerson) -> AppealOut:
    place = gazetteer.get(p.last_place)
    return AppealOut(
        id=p.id, name=p.name, age=p.age, summary=p.public_summary, clothing=p.clothing,
        last_seen_city=place.city if place else "Australia", last_seen_date=p.last_seen_at.date(),
        has_photo=bool(p.photo_path),
    )


def _published(session: Session):
    return session.query(MissingPerson).filter(MissingPerson.published.is_(True), MissingPerson.status == "open")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/public/notice")
def notice():
    return {"notice": NOTICE, "disclaimer": config.DISCLAIMER}


@app.get("/public/appeals", response_model=list[AppealOut])
def appeals(session: Session = Depends(_get_db)):
    return [_appeal(p) for p in _published(session).order_by(MissingPerson.id)]


@app.get("/public/appeals/{appeal_id}/photo")
def appeal_photo(appeal_id: int, session: Session = Depends(_get_db)):
    p = _published(session).filter(MissingPerson.id == appeal_id).first()
    if not p or not p.photo_path:
        raise HTTPException(404, "Not found")
    return FileResponse(p.photo_path)


def _reference() -> str:
    return "TIP-" + secrets.token_hex(5).upper()  # random, so references cannot be enumerated


def _flag(session: Session, source: str, text: str) -> str:
    """Reasons a tip needs extra scepticism. Flags never block a tip; they warn the reviewer."""
    since = utcnow() - timedelta(hours=24)
    reasons = []
    recent = session.query(Sighting).filter(Sighting.source_hash == source, Sighting.reported_at >= since).count()
    if recent + 1 >= config.TIP_FLAG_SOURCE_VOLUME:
        reasons.append(f"{recent + 1} tips from one source in 24 h")
    same = (
        session.query(Sighting)
        .filter(Sighting.source == "public", Sighting.reported_at >= since, Sighting.source_hash != source)
        .all()
    )
    if any(s.text.strip().lower() == text.strip().lower() for s in same):
        reasons.append("identical text already sent from another source")
    return "; ".join(reasons)


@app.post("/public/tips", response_model=TipReceipt, status_code=202)
def submit_tip(
    request: Request,
    text: str = Form(...),
    consent: bool = Form(False),
    appeal_id: int | None = Form(None),
    contact: str | None = Form(None),
    website: str = Form(""),  # honeypot: hidden in the form, so only bots fill it in
    image: UploadFile | None = File(None),
    session: Session = Depends(_get_db),
):
    source = security.source_hash(security.client_ip(request))
    SHORT_WINDOW.check(source)
    DAY_WINDOW.check(source)

    text = text.strip()
    if not consent:
        raise HTTPException(400, "Please confirm the privacy and false-report notice")
    if not 10 <= len(text) <= config.TIP_MAX_TEXT:
        raise HTTPException(400, f"Describe what you saw in 10 to {config.TIP_MAX_TEXT} characters")
    if contact and len(contact) > 120:
        raise HTTPException(400, "Contact details are too long")

    receipt = TipReceipt(
        reference=_reference(),
        message="Thank you. A police officer will review your report. If you have urgent information, call 000.",
    )
    if website:  # a bot: look successful, store nothing
        return receipt

    image_path = None
    if image is not None and image.filename:
        image_path = store_image(image.file.read(config.TIP_MAX_IMAGE_BYTES + 1))

    flag = _flag(session, source, text)
    pipeline.ingest_sighting(
        session, text=text, image_path=image_path, source="public",
        extra=dict(
            reference=receipt.reference, appeal_id=appeal_id, contact=(contact or None) and contact.strip(),
            source_hash=source, flagged=bool(flag), flag_reason=flag,
        ),
    )
    return receipt
