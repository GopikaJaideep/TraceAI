"""Officer API for police and NGO case officers. Everything except /login requires a session token.

Principles enforced here:
  * need-to-know: an officer only ever sees cases they have been granted (admins get no automatic access);
  * every read of case data and every change is written to the hash-chained audit log;
  * opening a case requires a police reference and a family-violence screening attestation;
  * closing a case deletes its photo, face embedding and leads.

Run: uvicorn traceai.api.officer_app:app --port 8010
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from traceai import cases, config, db, pipeline
from traceai.api import audit, security
from traceai.api.images import store_image
from traceai.api.ratelimit import RateLimiter
from traceai.db import AuditLog, CaseAccess, Lead, MissingPerson, User, to_naive_utc, utcnow
from traceai.geo import gazetteer
from traceai.vision import embedder

LOGIN_WINDOW = RateLimiter(config.LOGIN_RATE_PER_MIN, 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="TraceAI officer API", version="0.2.0", description=config.DISCLAIMER, lifespan=lifespan)
get_db = security.get_db
current_user = security.current_user


# --------------------------------------------------------------------------------------- auth ----
class LoginIn(BaseModel):
    username: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok", "face_backend": embedder.available(), "disclaimer": config.DISCLAIMER}


@app.post("/officer/login")
def login(body: LoginIn, request: Request, session: Session = Depends(get_db)):
    LOGIN_WINDOW.check(security.client_ip(request))
    user = security.authenticate(session, body.username, body.password)
    if user is None:
        audit.record(session, "login.failed", username=body.username[:80], request=request)
        raise HTTPException(401, "Incorrect username or password")  # same message for every failure mode
    audit.record(session, "login", user=user, request=request)
    return {"token": security.issue_token(user), "username": user.username, "role": user.role,
            "agency": user.agency, "expires_in": config.TOKEN_TTL_SECONDS}


@app.get("/officer/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "username": user.username, "role": user.role, "agency": user.agency}


# -------------------------------------------------------------------------------------- cases ----
def _case_out(p: MissingPerson) -> dict:
    return {
        "id": p.id, "name": p.name, "age": p.age, "description": p.description, "clothing": p.clothing,
        "last_place": p.last_place, "last_lat": p.last_lat, "last_lng": p.last_lng,
        "last_seen_at": p.last_seen_at, "police_reference": p.police_reference, "status": p.status,
        "published": p.published, "face_matching_authorised": p.face_matching_authorised,
        "has_photo": bool(p.photo_path),
    }


@app.get("/officer/cases")
def list_cases(user: User = Depends(current_user), session: Session = Depends(get_db)):
    ids = [a.person_id for a in session.query(CaseAccess).filter_by(user_id=user.id)]
    cases_ = session.query(MissingPerson).filter(MissingPerson.id.in_(ids)).order_by(MissingPerson.id).all()
    return [_case_out(p) for p in cases_]


@app.post("/officer/cases", status_code=201)
def create_case(
    request: Request,
    name: str = Form(...),
    police_reference: str = Form(...),
    family_violence_screened: bool = Form(False),
    last_place: str = Form(...),
    last_seen_at: datetime = Form(...),
    age: int | None = Form(None),
    description: str = Form(""),
    clothing: str = Form(""),
    last_lat: float | None = Form(None),
    last_lng: float | None = Form(None),
    photo_lawfully_obtained: bool = Form(False),
    publish: bool = Form(False),
    public_summary: str = Form(""),
    photo: UploadFile | None = File(None),
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
):
    if last_lat is None or last_lng is None:
        place = gazetteer.resolve(last_place)
        if place is None:
            raise HTTPException(422, "Last-known place not recognised: supply last_lat and last_lng")
        last_lat, last_lng = place.lat, place.lng
    photo_path = store_image(photo.file.read(config.TIP_MAX_IMAGE_BYTES + 1)) if photo and photo.filename else None
    try:
        person = cases.create_case(
            session, user, name=name, age=age, description=description,
            clothing=[c.strip() for c in clothing.split(",") if c.strip()], last_place=last_place,
            last_lat=last_lat, last_lng=last_lng, last_seen_at=last_seen_at,
            police_reference=police_reference, family_violence_screened=family_violence_screened,
            photo_path=photo_path, photo_lawfully_obtained=photo_lawfully_obtained,
            publish=publish, public_summary=public_summary,
        )
    except cases.CaseError as exc:
        raise HTTPException(422, str(exc))
    audit.record(session, "case.create", user=user, object_type="case", object_id=person.id, request=request,
                 detail={"police_reference": person.police_reference, "published": person.published,
                         "face_matching": person.face_matching_authorised})
    return _case_out(person)


def _lead_out(lead: Lead) -> dict:
    s = lead.sighting
    return {
        "id": lead.id, "score": lead.score, "components": lead.components, "notes": lead.notes,
        "review": lead.review, "text": s.text, "seen_at": s.seen_at, "lat": s.lat, "lng": s.lng,
        "place": (s.extracted or {}).get("place"), "has_image": bool(s.image_path), "source": s.source,
        "reference": s.reference, "flagged": s.flagged, "flag_reason": s.flag_reason,
        "contact": s.contact,
    }


@app.get("/officer/cases/{case_id}/analysis")
def case_analysis(
    case_id: int, request: Request, min_score: float = 0.25,
    user: User = Depends(current_user), session: Session = Depends(get_db),
):
    person = security.case_for_user(session, user, case_id)
    result = pipeline.person_analysis(session, case_id, min_score)
    audit.record(session, "case.view", user=user, object_type="case", object_id=case_id, request=request,
                 detail={"leads": len(result["leads"])})
    return {
        "case": _case_out(person),
        "leads": [_lead_out(l) for l in result["leads"]],
        "clusters": [
            {"label": c.label, "sighting_ids": c.member_ids, "lat": c.lat, "lng": c.lng, "place": c.place,
             "size": c.size, "first_seen": c.first_seen, "last_seen": c.last_seen}
            for c in result["clusters"]
        ],
        "corridor": result["corridor"],
    }


class ReviewIn(BaseModel):
    review: Literal["useful", "dismissed", "unreviewed"]


def _lead_for_user(session: Session, user: User, lead_id: int) -> Lead:
    lead = session.get(Lead, lead_id)
    if lead is None:
        raise HTTPException(404, "Lead not found")
    security.case_for_user(session, user, lead.person_id)  # 404 unless the officer holds the case
    return lead


@app.patch("/officer/leads/{lead_id}")
def review_lead(
    lead_id: int, body: ReviewIn, request: Request,
    user: User = Depends(current_user), session: Session = Depends(get_db),
):
    lead = _lead_for_user(session, user, lead_id)
    lead.review, lead.reviewed_by, lead.reviewed_at = body.review, user.id, utcnow()
    session.commit()
    audit.record(session, "lead.review", user=user, object_type="lead", object_id=lead.id, request=request,
                 detail={"review": body.review, "case_id": lead.person_id})
    return {"id": lead.id, "review": lead.review}


@app.get("/officer/leads/{lead_id}/image")
def lead_image(lead_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_db)):
    lead = _lead_for_user(session, user, lead_id)
    if not lead.sighting.image_path:
        raise HTTPException(404, "No image")
    audit.record(session, "lead.image.view", user=user, object_type="lead", object_id=lead.id, request=request)
    return FileResponse(lead.sighting.image_path)


@app.get("/officer/cases/{case_id}/photo")
def case_photo(case_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_db)):
    person = security.case_for_user(session, user, case_id)
    if not person.photo_path:
        raise HTTPException(404, "No photo")
    audit.record(session, "case.photo.view", user=user, object_type="case", object_id=case_id, request=request)
    return FileResponse(person.photo_path)


class PublishIn(BaseModel):
    published: bool
    public_summary: str | None = None


@app.post("/officer/cases/{case_id}/publish")
def publish_case(
    case_id: int, body: PublishIn, request: Request,
    user: User = Depends(current_user), session: Session = Depends(get_db),
):
    person = security.case_for_user(session, user, case_id)
    if person.status != "open":
        raise HTTPException(409, "Case is closed")
    if body.public_summary is not None:
        person.public_summary = body.public_summary
    if body.published and not person.public_summary.strip():
        raise HTTPException(422, "A public summary is required to publish an appeal")
    person.published = body.published
    session.commit()
    audit.record(session, "case.publish" if body.published else "case.unpublish", user=user,
                 object_type="case", object_id=case_id, request=request)
    return _case_out(person)


@app.post("/officer/cases/{case_id}/close")
def close_case(case_id: int, request: Request, user: User = Depends(current_user), session: Session = Depends(get_db)):
    person = security.case_for_user(session, user, case_id)
    if person.status == "closed":
        raise HTTPException(409, "Case is already closed")
    cases.close_case(session, person)
    audit.record(session, "case.close", user=user, object_type="case", object_id=case_id, request=request,
                 detail={"purged": ["photo", "face_embedding", "leads"]})
    return _case_out(person)


@app.post("/officer/sightings", status_code=201)
def log_sighting(
    request: Request,
    text: str = Form(...),
    seen_at: datetime | None = Form(None),
    image: UploadFile | None = File(None),
    user: User = Depends(current_user),
    session: Session = Depends(get_db),
):
    """An officer records a phone-in or walk-in report. Returns only leads for cases they hold."""
    image_path = store_image(image.file.read(config.TIP_MAX_IMAGE_BYTES + 1)) if image and image.filename else None
    sighting, leads = pipeline.ingest_sighting(
        session, text=text, image_path=image_path, seen_at=to_naive_utc(seen_at), source="officer"
    )
    held = {a.person_id for a in session.query(CaseAccess).filter_by(user_id=user.id)}
    audit.record(session, "sighting.log", user=user, object_type="sighting", object_id=sighting.id, request=request)
    return [_lead_out(l) | {"case_id": l.person_id} for l in leads if l.person_id in held]


# -------------------------------------------------------------------------------------- admin ----
class NewUser(BaseModel):
    username: str
    password: str
    role: Literal["officer", "admin"] = "officer"
    agency: str = ""


@app.post("/officer/admin/users", status_code=201)
def create_user(
    body: NewUser, request: Request, admin: User = Depends(security.require_admin), session: Session = Depends(get_db)
):
    try:
        security.check_password_policy(body.password)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if session.query(User).filter_by(username=body.username).first():
        raise HTTPException(409, "Username already exists")
    user = User(username=body.username, password_hash=security.hash_password(body.password),
                role=body.role, agency=body.agency)
    session.add(user)
    session.commit()
    audit.record(session, "user.create", user=admin, object_type="user", object_id=user.id, request=request,
                 detail={"role": body.role})
    return {"id": user.id, "username": user.username, "role": user.role}


@app.get("/officer/admin/users")
def list_users(_: User = Depends(security.require_admin), session: Session = Depends(get_db)):
    return [{"id": u.id, "username": u.username, "role": u.role, "agency": u.agency, "active": u.active}
            for u in session.query(User).order_by(User.id)]


@app.post("/officer/admin/users/{user_id}/deactivate")
def deactivate_user(
    user_id: int, request: Request, admin: User = Depends(security.require_admin), session: Session = Depends(get_db)
):
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(409, "You cannot deactivate your own account")
    user.active = False
    session.commit()
    audit.record(session, "user.deactivate", user=admin, object_type="user", object_id=user.id, request=request)
    return {"id": user.id, "active": False}


class GrantIn(BaseModel):
    user_id: int


@app.post("/officer/admin/cases/{case_id}/access", status_code=201)
def grant_access(
    case_id: int, body: GrantIn, request: Request,
    admin: User = Depends(security.require_admin), session: Session = Depends(get_db),
):
    if session.get(MissingPerson, case_id) is None or session.get(User, body.user_id) is None:
        raise HTTPException(404, "Case or user not found")
    if not session.query(CaseAccess).filter_by(user_id=body.user_id, person_id=case_id).first():
        session.add(CaseAccess(user_id=body.user_id, person_id=case_id, granted_by=admin.id))
        session.commit()
    audit.record(session, "case.access.grant", user=admin, object_type="case", object_id=case_id, request=request,
                 detail={"to_user": body.user_id})
    return {"case_id": case_id, "user_id": body.user_id}


@app.get("/officer/admin/audit")
def read_audit(
    request: Request, limit: int = 200,
    admin: User = Depends(security.require_admin), session: Session = Depends(get_db),
):
    rows = session.query(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 1000)).all()
    audit.record(session, "audit.view", user=admin, request=request, detail={"rows": len(rows)})  # who watches the watchers
    return [{"id": r.id, "ts": r.ts, "username": r.username, "action": r.action, "object_type": r.object_type,
             "object_id": r.object_id, "ip": r.ip, "detail": r.detail} for r in rows]


@app.get("/officer/admin/audit/verify")
def verify_audit(_: User = Depends(security.require_admin), session: Session = Depends(get_db)):
    ok, bad_id = audit.verify_chain(session)
    return {"intact": ok, "first_bad_id": bad_id}
