"""FastAPI backend. Run with: uvicorn traceai.api.main:app"""
from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from traceai import config, db, pipeline
from traceai.db import Lead, MissingPerson, Sighting
from traceai.vision import embedder

UPLOAD_DIR = config.DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="TraceAI", version="0.1.0", description=config.DISCLAIMER, lifespan=lifespan)


def get_db():
    session = db.session()
    try:
        yield session
    finally:
        session.close()


class PersonOut(BaseModel):
    id: int
    name: str
    age: int | None
    description: str
    clothing: list[str]
    last_place: str
    last_lat: float
    last_lng: float
    last_seen_at: datetime
    has_photo: bool


class LeadOut(BaseModel):
    id: int
    person_id: int
    sighting_id: int
    score: float
    components: dict[str, float | None]
    notes: list[str]
    text: str
    seen_at: datetime | None
    lat: float | None
    lng: float | None
    place: str | None
    has_image: bool


class ClusterOut(BaseModel):
    label: int
    sighting_ids: list[int]
    lat: float
    lng: float
    place: str
    size: int
    first_seen: datetime | None
    last_seen: datetime | None


class AnalysisOut(BaseModel):
    person: PersonOut
    leads: list[LeadOut]
    clusters: list[ClusterOut]
    corridor: str


def _person_out(p: MissingPerson) -> PersonOut:
    return PersonOut(
        id=p.id, name=p.name, age=p.age, description=p.description, clothing=p.clothing,
        last_place=p.last_place, last_lat=p.last_lat, last_lng=p.last_lng,
        last_seen_at=p.last_seen_at, has_photo=bool(p.photo_path),
    )


def _lead_out(l: Lead) -> LeadOut:
    s = l.sighting
    return LeadOut(
        id=l.id, person_id=l.person_id, sighting_id=s.id, score=l.score, components=l.components,
        notes=l.notes, text=s.text, seen_at=s.seen_at, lat=s.lat, lng=s.lng,
        place=(s.extracted or {}).get("place"), has_image=bool(s.image_path),
    )


@app.get("/health")
def health():
    return {"status": "ok", "face_backend": embedder.available(), "disclaimer": config.DISCLAIMER}


@app.get("/persons", response_model=list[PersonOut])
def list_persons(session=Depends(get_db)):
    return [_person_out(p) for p in session.query(MissingPerson).order_by(MissingPerson.id)]


@app.get("/persons/{person_id}/photo")
def person_photo(person_id: int, session=Depends(get_db)):
    p = session.get(MissingPerson, person_id)
    if not p or not p.photo_path:
        raise HTTPException(404, "No photo")
    return FileResponse(p.photo_path)


@app.get("/persons/{person_id}/analysis", response_model=AnalysisOut)
def analysis(person_id: int, min_score: float = 0.25, session=Depends(get_db)):
    person = session.get(MissingPerson, person_id)
    if not person:
        raise HTTPException(404, "Unknown profile")
    result = pipeline.person_analysis(session, person_id, min_score)
    return AnalysisOut(
        person=_person_out(person),
        leads=[_lead_out(l) for l in result["leads"]],
        clusters=[
            ClusterOut(label=c.label, sighting_ids=c.member_ids, lat=c.lat, lng=c.lng, place=c.place,
                       size=c.size, first_seen=c.first_seen, last_seen=c.last_seen)
            for c in result["clusters"]
        ],
        corridor=result["corridor"],
    )


@app.post("/sightings", response_model=list[LeadOut])
def submit_sighting(
    text: str = Form(...),
    seen_at: datetime | None = Form(None),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    image: UploadFile | None = File(None),
    session=Depends(get_db),
):
    """Ingest one report (and optional photo); returns its leads ranked across all profiles."""
    image_path = None
    if image is not None and image.filename:
        suffix = "." + image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(400, "Image must be .jpg or .png")
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with dest.open("wb") as fh:
            shutil.copyfileobj(image.file, fh)
        image_path = str(dest)
    _, leads = pipeline.ingest_sighting(
        session, text=text, image_path=image_path, seen_at=seen_at, lat=lat, lng=lng
    )
    return [_lead_out(l) for l in leads]


@app.get("/sightings/{sighting_id}/image")
def sighting_image(sighting_id: int, session=Depends(get_db)):
    s = session.get(Sighting, sighting_id)
    if not s or not s.image_path:
        raise HTTPException(404, "No image")
    return FileResponse(s.image_path)
