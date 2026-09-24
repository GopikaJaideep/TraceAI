"""Ingestion pipeline: report + optional image -> features -> lead scores -> database."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from traceai import scoring
from traceai.db import Lead, MissingPerson, Sighting, to_naive_utc, utcnow
from traceai.geo import gazetteer
from traceai.geo.clustering import SightingPoint, cluster_sightings, corridor_text
from traceai.nlp import extractor, textsim
from traceai.vision import embedder


def _clothing_overlap(a: list[str], b: list[str]) -> float | None:
    if not a or not b:
        return None
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def description_similarity(person: MissingPerson, text: str, clothing: list[str]) -> float | None:
    """Blend of exact clothing overlap and semantic similarity between the two descriptions."""
    if not text.strip():
        return None
    semantic = textsim.similarity(f"{person.description}. Wearing {', '.join(person.clothing)}", text)
    overlap = _clothing_overlap(person.clothing, clothing)
    return semantic if overlap is None else 0.5 * semantic + 0.5 * overlap


def ingest_sighting(
    db: Session,
    *,
    text: str,
    image_path: str | None = None,
    seen_at: datetime | None = None,
    lat: float | None = None,
    lng: float | None = None,
    reported_at: datetime | None = None,
) -> tuple[Sighting, list[Lead]]:
    """Analyse one sighting against every missing-person profile and persist ranked leads.

    Explicit `seen_at` / `lat` / `lng` override what the NLP module infers from the text.
    """
    reported_at = to_naive_utc(reported_at) or utcnow()
    seen_at = to_naive_utc(seen_at)
    ex = extractor.extract(text, reported_at)
    if lat is not None and lng is not None:
        # Coordinates win over the text, so the label must describe the coordinates, not the prose.
        ex.lat, ex.lng = lat, lng
        place, dist = gazetteer.nearest(lat, lng)
        ex.place_text = place.name if dist < 3 else None
    seen_at = seen_at or ex.seen_at
    ex.seen_at = seen_at  # keep the stored extraction and credibility consistent with the row

    vec = embedder.embed_file(image_path) if image_path else None
    sighting = Sighting(
        reported_at=reported_at, seen_at=seen_at, text=text, image_path=image_path,
        extracted=ex.to_dict(), lat=ex.lat, lng=ex.lng,
        embedding=vec.tolist() if vec is not None else None,
    )
    db.add(sighting)
    db.flush()

    leads = []
    for person in db.query(MissingPerson).all():
        ls = scoring.score_lead(
            person_lat=person.last_lat, person_lng=person.last_lng, last_seen_at=person.last_seen_at,
            ex=ex, seen_at=seen_at, now=reported_at,
            face_cosine=embedder.cosine(person.embedding, sighting.embedding),
            description_sim=description_similarity(person, text, ex.clothing),
            has_photo=image_path is not None,
        )
        lead = Lead(person_id=person.id, sighting_id=sighting.id, score=ls.score,
                    components=ls.components, notes=ls.notes)
        db.add(lead)
        leads.append(lead)
    db.commit()
    return sighting, sorted(leads, key=lambda l: -l.score)


CORRIDOR_MIN_SCORE = 0.6  # the corridor is a claim about movement, so only strong leads feed it


def person_analysis(db: Session, person_id: int, min_score: float = 0.25) -> dict:
    """Ranked leads plus geospatial clusters and the inferred movement corridor for one person."""
    leads = (
        db.query(Lead).filter(Lead.person_id == person_id, Lead.score >= min_score)
        .order_by(Lead.score.desc()).all()
    )
    points = [
        SightingPoint(l.sighting.id, l.sighting.lat, l.sighting.lng, l.sighting.seen_at, l.score)
        for l in leads if l.sighting.lat is not None and l.score >= max(min_score, CORRIDOR_MIN_SCORE)
    ]
    clusters = cluster_sightings(points)
    return {"leads": leads, "clusters": clusters, "corridor": corridor_text(clusters)}
