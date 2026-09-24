"""A small Australian gazetteer for resolving place names in witness reports.

The proposal calls for the full OpenStreetMap / ASGS extracts (gigabytes). The prototype uses a
curated set of well-known places (approximate coordinates) so it runs offline; the interface
(`resolve`, `nearest`) is what a full OSM-backed implementation would keep.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    name: str
    city: str
    lat: float
    lng: float
    aliases: tuple[str, ...] = ()


PLACES: list[Place] = [
    # Melbourne
    Place("Melbourne CBD", "Melbourne", -37.8136, 144.9631, ("melbourne city", "the cbd", "melbourne cbd")),
    Place("Flinders Street Station", "Melbourne", -37.8183, 144.9671, ("flinders street", "flinders st", "flinders st station")),
    Place("Southern Cross Station", "Melbourne", -37.8184, 144.9525, ("southern cross",)),
    Place("Melbourne Central", "Melbourne", -37.8102, 144.9628),
    Place("Federation Square", "Melbourne", -37.8180, 144.9691, ("fed square",)),
    Place("Queen Victoria Market", "Melbourne", -37.8076, 144.9568, ("queen vic market", "vic market")),
    Place("Docklands", "Melbourne", -37.8148, 144.9463),
    Place("Southbank", "Melbourne", -37.8226, 144.9646),
    Place("Richmond", "Melbourne", -37.8241, 144.9903, ("richmond station",)),
    Place("Collingwood", "Melbourne", -37.8043, 144.9880),
    Place("Fitzroy", "Melbourne", -37.7980, 144.9780),
    Place("Carlton", "Melbourne", -37.8002, 144.9669),
    Place("Brunswick", "Melbourne", -37.7667, 144.9603),
    Place("South Yarra", "Melbourne", -37.8386, 144.9927),
    Place("Prahran", "Melbourne", -37.8510, 144.9930),
    Place("St Kilda", "Melbourne", -37.8676, 144.9805, ("saint kilda",)),
    Place("Footscray", "Melbourne", -37.8000, 144.8999),
    Place("Melbourne Airport", "Melbourne", -37.6690, 144.8410, ("tullamarine",)),
    # Sydney
    Place("Sydney CBD", "Sydney", -33.8688, 151.2093, ("sydney city",)),
    Place("Central Station", "Sydney", -33.8832, 151.2067),
    Place("Town Hall", "Sydney", -33.8732, 151.2069, ("town hall station",)),
    Place("Circular Quay", "Sydney", -33.8611, 151.2108),
    Place("Darling Harbour", "Sydney", -33.8737, 151.1990),
    Place("Surry Hills", "Sydney", -33.8850, 151.2117),
    Place("Redfern", "Sydney", -33.8927, 151.2036),
    Place("Kings Cross", "Sydney", -33.8747, 151.2225),
    Place("Newtown", "Sydney", -33.8975, 151.1794),
    Place("Bondi Beach", "Sydney", -33.8915, 151.2767, ("bondi",)),
    Place("Manly", "Sydney", -33.7969, 151.2840),
    Place("Parramatta", "Sydney", -33.8150, 151.0011),
    # Brisbane
    Place("Brisbane CBD", "Brisbane", -27.4698, 153.0251, ("brisbane city", "queen street mall")),
    Place("Roma Street Station", "Brisbane", -27.4657, 153.0187, ("roma street",)),
    Place("South Bank", "Brisbane", -27.4761, 153.0203, ("south bank parklands",)),
    Place("Fortitude Valley", "Brisbane", -27.4570, 153.0350, ("the valley",)),
    Place("West End", "Brisbane", -27.4813, 153.0110),
    # Perth
    Place("Perth CBD", "Perth", -31.9523, 115.8613, ("perth city",)),
    Place("Perth Station", "Perth", -31.9505, 115.8605, ("perth train station",)),
    Place("Northbridge", "Perth", -31.9450, 115.8580),
    Place("Fremantle", "Perth", -32.0569, 115.7439),
    Place("Scarborough", "Perth", -31.8944, 115.7600),
    # Adelaide
    Place("Adelaide CBD", "Adelaide", -34.9285, 138.6007, ("adelaide city",)),
    Place("Rundle Mall", "Adelaide", -34.9231, 138.6030),
    Place("Adelaide Railway Station", "Adelaide", -34.9215, 138.5990, ("adelaide station",)),
    Place("Glenelg", "Adelaide", -34.9800, 138.5150),
    # Other capitals / regions
    Place("Canberra Civic", "Canberra", -35.2809, 149.1300, ("civic", "canberra city")),
    Place("Parliament House", "Canberra", -35.3082, 149.1245),
    Place("Hobart CBD", "Hobart", -42.8821, 147.3272, ("hobart city",)),
    Place("Salamanca Place", "Hobart", -42.8867, 147.3301, ("salamanca",)),
    Place("Darwin CBD", "Darwin", -12.4634, 130.8456, ("darwin city",)),
    Place("Mindil Beach", "Darwin", -12.4470, 130.8290),
    Place("Surfers Paradise", "Gold Coast", -28.0023, 153.4303, ("surfers",)),
]

_BY_NAME = {p.name.lower(): p for p in PLACES}

# (regex, place), longest surface form first so "Flinders Street Station" beats "Flinders Street".
_PATTERNS: list[tuple[re.Pattern[str], Place]] = sorted(
    (
        (re.compile(r"\b" + re.escape(form.lower()) + r"\b"), p)
        for p in PLACES
        for form in (p.name, *p.aliases)
    ),
    key=lambda item: -len(item[0].pattern),
)


def get(name: str) -> Place | None:
    return _BY_NAME.get(name.lower())


def resolve(text: str) -> Place | None:
    """Return the first (longest-form) known place mentioned in `text`, or None."""
    lowered = text.lower()
    for pattern, place in _PATTERNS:
        if pattern.search(lowered):
            return place
    return None


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest(lat: float, lng: float) -> tuple[Place, float]:
    """Nearest gazetteer place to a coordinate and its distance in km."""
    best = min(PLACES, key=lambda p: haversine_km(lat, lng, p.lat, p.lng))
    return best, haversine_km(lat, lng, best.lat, best.lng)
