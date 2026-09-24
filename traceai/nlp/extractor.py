"""Structured-information extraction from free-text witness reports.

Rules do the heavy lifting (clothing, time, gazetteer places); spaCy NER, when a model is
installed, adds place-like entities that the gazetteer could not resolve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

from traceai.geo import gazetteer

COLOURS = (
    "black white grey gray red blue navy green yellow orange purple pink brown beige khaki "
    "maroon teal cream"
).split()
GARMENTS = (
    "hoodie jacket jeans shirt t-shirt tshirt tee dress cap beanie hat backpack bag shorts coat "
    "jumper sweater skirt trousers pants sneakers shoes boots scarf"
).split()

_CLOTHING_RE = re.compile(
    r"\b(" + "|".join(COLOURS) + r")\s+(?:\w+\s+)?(" + "|".join(GARMENTS) + r")\b", re.I
)
_HEDGES = re.compile(
    r"\b(maybe|might|not sure|i think|possibly|perhaps|could have been|looked like|"
    r"bit far away|from a distance|hard to tell|probably)\b",
    re.I,
)
_CERTAINTY = re.compile(r"\b(definitely|certain|absolutely|clearly|i'm sure|spoke to|recognis(?:ed|e))\b", re.I)

_CLOCK_RE = re.compile(r"\b(?:at|around|about|near|by)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_DAYPARTS = {"morning": 9, "afternoon": 15, "evening": 19, "night": 22, "lunchtime": 12, "noon": 12}


@dataclass
class Extraction:
    place_text: str | None = None
    lat: float | None = None
    lng: float | None = None
    clothing: list[str] = field(default_factory=list)
    seen_at: datetime | None = None
    unresolved_places: list[str] = field(default_factory=list)
    hedges: int = 0
    certainty_cues: int = 0

    def to_dict(self) -> dict:
        return {
            "place": self.place_text,
            "lat": self.lat,
            "lng": self.lng,
            "clothing": self.clothing,
            "seen_at": self.seen_at.isoformat() if self.seen_at else None,
            "unresolved_places": self.unresolved_places,
            "hedges": self.hedges,
            "certainty_cues": self.certainty_cues,
        }


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except (ImportError, OSError):
        return None  # NER is an optional extra; the rules still work


def _parse_time(text: str, reference: datetime) -> datetime | None:
    lowered = text.lower()
    day = reference.date()
    if "yesterday" in lowered or "last night" in lowered:
        day = day - timedelta(days=1)
    m = _CLOCK_RE.search(text)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        minute = int(m.group(2) or 0)
        if hour > 23 or minute > 59:
            return None
        seen = datetime(day.year, day.month, day.day, hour, minute)
    else:
        seen = None
        for word, hour in _DAYPARTS.items():
            if word in lowered:
                seen = datetime(day.year, day.month, day.day, hour)
                break
        if seen is None:
            return None
    if seen > reference and "yesterday" not in lowered and "last night" not in lowered:
        seen -= timedelta(days=1)  # "around 7pm" reported at 2am means the previous evening
    return seen


def extract(text: str, reported_at: datetime) -> Extraction:
    out = Extraction()

    place = gazetteer.resolve(text)
    if place:
        out.place_text, out.lat, out.lng = place.name, place.lat, place.lng

    seen: set[str] = set()
    for colour, garment in _CLOTHING_RE.findall(text):
        item = f"{'grey' if colour.lower() == 'gray' else colour.lower()} {garment.lower()}"
        if item not in seen:
            seen.add(item)
            out.clothing.append(item)

    out.seen_at = _parse_time(text, reported_at)
    out.hedges = len(_HEDGES.findall(text))
    out.certainty_cues = len(_CERTAINTY.findall(text))

    nlp = _nlp()
    if nlp is not None:
        for ent in nlp(text).ents:
            if ent.label_ in {"GPE", "LOC", "FAC"} and (
                out.place_text is None or ent.text.lower() not in out.place_text.lower()
            ):
                out.unresolved_places.append(ent.text)
    return out
