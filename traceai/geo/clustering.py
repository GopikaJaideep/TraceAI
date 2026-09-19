"""Spatial clustering of sightings and inference of a movement corridor."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN

from traceai.geo import gazetteer

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SightingPoint:
    id: int
    lat: float
    lng: float
    seen_at: datetime | None
    weight: float = 1.0  # the lead score, so strong leads dominate cluster centres


@dataclass
class Cluster:
    label: int
    member_ids: list[int]
    lat: float
    lng: float
    place: str
    first_seen: datetime | None
    last_seen: datetime | None

    @property
    def size(self) -> int:
        return len(self.member_ids)


def cluster_sightings(points: list[SightingPoint], eps_km: float = 2.0, min_samples: int = 2) -> list[Cluster]:
    """DBSCAN over haversine distance. Isolated sightings become singleton clusters so that they
    still contribute to the corridor; use `Cluster.size` to tell dense clusters from one-offs."""
    if not points:
        return []
    coords = np.radians([[p.lat, p.lng] for p in points])
    labels = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine"
    ).fit_predict(coords)

    groups: dict[int, list[SightingPoint]] = {}
    next_singleton = int(labels.max()) + 1 if labels.size else 0
    for point, label in zip(points, labels):
        if label == -1:
            label, next_singleton = next_singleton, next_singleton + 1
        groups.setdefault(int(label), []).append(point)

    clusters = []
    for label, members in groups.items():
        w = np.array([m.weight for m in members]) + 1e-9
        lat = float(np.average([m.lat for m in members], weights=w))
        lng = float(np.average([m.lng for m in members], weights=w))
        times = [m.seen_at for m in members if m.seen_at]
        place, _ = gazetteer.nearest(lat, lng)
        clusters.append(
            Cluster(label, [m.id for m in members], lat, lng, place.name,
                    min(times) if times else None, max(times) if times else None)
        )
    return clusters


def movement_corridor(clusters: list[Cluster]) -> list[Cluster]:
    """Clusters ordered by when they were first seen (undated clusters are excluded)."""
    dated = [c for c in clusters if c.first_seen is not None]
    return sorted(dated, key=lambda c: c.first_seen)


def corridor_text(clusters: list[Cluster]) -> str:
    path: list[str] = []
    for c in movement_corridor(clusters):
        if not path or path[-1] != c.place:
            path.append(c.place)
    return " → ".join(path)
