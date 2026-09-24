"""Lead prioritisation: combines image, description, proximity, recency and credibility signals.

Every component is normalised to [0, 1]. Components that are unavailable (e.g. a text-only report
has no image similarity) are dropped and the remaining weights renormalised, so a lead is never
penalised for a signal it could not provide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from traceai.geo.gazetteer import haversine_km
from traceai.nlp.extractor import Extraction

WEIGHTS = {
    "image": 0.35,
    "description": 0.15,
    "proximity": 0.20,
    "recency": 0.10,
    "credibility": 0.20,
}
PROXIMITY_SCALE_KM = 100.0
MAX_PLAUSIBLE_SPEED_KMH = 150.0
RECENCY_HALF_LIFE_H = 72.0
# Location, recency and credibility say a report is *usable*, not that it is about *this* person.
# Identity evidence (face or description) therefore scales the score between these bounds; with no
# identity evidence at all the factor is neutral.
IDENTITY_FLOOR = 0.4


@dataclass
class LeadScore:
    score: float
    components: dict[str, float | None]
    notes: list[str]


def image_component(cosine: float | None) -> float | None:
    """Map ArcFace cosine similarity onto [0, 1]. Same-identity pairs typically sit above ~0.4."""
    if cosine is None:
        return None
    return min(1.0, max(0.0, (cosine - 0.15) / 0.5))


def proximity_component(
    person_lat: float, person_lng: float, last_seen_at: datetime,
    lat: float | None, lng: float | None, seen_at: datetime | None,
) -> tuple[float | None, str | None]:
    hours = None
    if seen_at is not None:
        hours = (seen_at - last_seen_at).total_seconds() / 3600
        if hours < 0:  # chronology does not depend on knowing where it happened
            return 0.0, "sighting predates the last-known time"
    if lat is None or lng is None:
        return None, None
    dist = haversine_km(person_lat, person_lng, lat, lng)
    if hours is not None and dist > 1 and dist / max(hours, 0.05) > MAX_PLAUSIBLE_SPEED_KMH:
        return 0.0, f"{dist:.0f} km in {hours:.1f} h is not physically plausible"
    return math.exp(-dist / PROXIMITY_SCALE_KM), f"{dist:.1f} km from last-known location"


def recency_component(seen_at: datetime | None, now: datetime) -> float | None:
    if seen_at is None:
        return None
    hours = max(0.0, (now - seen_at).total_seconds() / 3600)
    return 0.5 ** (hours / RECENCY_HALF_LIFE_H)


def credibility_component(ex: Extraction, has_photo: bool) -> float:
    score = 0.5
    score += 0.15 if ex.place_text else 0.0
    score += 0.10 if ex.seen_at else 0.0
    score += 0.10 if ex.clothing else 0.0
    score += 0.10 if ex.certainty_cues else 0.0
    score += 0.05 if has_photo else 0.0
    score -= min(0.3, 0.1 * ex.hedges)
    return min(1.0, max(0.05, score))


def combine(components: dict[str, float | None]) -> float:
    available = {k: v for k, v in components.items() if v is not None}
    total = sum(WEIGHTS[k] for k in available)
    if total == 0:
        return 0.0
    weighted = sum(WEIGHTS[k] * v for k, v in available.items()) / total
    identity = [v for k in ("image", "description") if (v := components.get(k)) is not None]
    factor = 0.5 * (1 + IDENTITY_FLOOR) if not identity else IDENTITY_FLOOR + (1 - IDENTITY_FLOOR) * max(identity)
    return weighted * factor


def score_lead(
    *,
    person_lat: float, person_lng: float, last_seen_at: datetime,
    ex: Extraction, seen_at: datetime | None, now: datetime,
    face_cosine: float | None, description_sim: float | None, has_photo: bool,
) -> LeadScore:
    prox, prox_note = proximity_component(
        person_lat, person_lng, last_seen_at, ex.lat, ex.lng, seen_at
    )
    components = {
        "image": image_component(face_cosine),
        "description": description_sim,
        "proximity": prox,
        "recency": recency_component(seen_at, now),
        "credibility": credibility_component(ex, has_photo),
    }
    notes = [prox_note] if prox_note else []
    if prox == 0.0:
        # An impossible journey cannot be rescued by a good face match.
        return LeadScore(0.0, components, notes)
    if face_cosine is not None:
        notes.append(f"face cosine similarity {face_cosine:.2f}")
    return LeadScore(round(combine(components), 4), components, notes)
