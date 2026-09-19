"""Synthetic missing-person profiles and witness reports.

Faces come from the public LFW research dataset (via scikit-learn's downloader); every *profile*
(name, age, clothing, last-known location) is invented. A true sighting reuses a *different* LFW
photo of the same identity, so image matching is a genuine cross-photo test; decoys use other
identities, other clothing, and other places.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import cv2
import numpy as np

from traceai import config
from traceai.db import MissingPerson
from traceai.geo import gazetteer
from traceai.vision import embedder

FIRST = ["Jack", "Olivia", "Noah", "Charlotte", "Liam", "Mia", "Oscar", "Isla", "Lucas", "Chloe",
         "Ethan", "Ruby", "Mason", "Zoe", "Hudson", "Grace"]
LAST = ["Nguyen", "Smith", "Patel", "Williams", "Brown", "Taylor", "Wilson", "Kelly", "Singh",
        "Anderson", "Martin", "Thompson", "White", "Harris", "Clarke", "Walker"]
OUTFITS = [["grey hoodie", "blue jeans"], ["black jacket", "white sneakers"], ["red t-shirt", "black shorts"],
           ["green jumper", "blue jeans"], ["navy coat", "black trousers"], ["white shirt", "black jeans"],
           ["yellow jacket", "grey trousers"], ["brown coat", "blue jeans"]]

TRUE_TEMPLATES = [
    "Saw someone matching the description near {place} wearing a {c0} around {t}. Definitely looked like the photo.",
    "I'm sure I saw the person at {place}, in a {c0} and {c1}, at about {t}.",
    "Spotted them outside {place}. They had on a {c0}. This was around {t}.",
]
DECOY_TEMPLATES = [
    "Maybe saw someone like that at {place}? Wearing a {c0}, not sure about the time, possibly {t}.",
    "I think it could have been the person near {place} from a distance. {c0}, hard to tell.",
    "Someone in a {c0} at {place} around {t}. Might be them, probably not.",
]


def _fmt_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    suffix = "am" if dt.hour < 12 else "pm"
    return f"{hour}{suffix}" if dt.minute == 0 else f"{hour}:{dt.minute:02d}{suffix}"


def _load_lfw(min_faces: int = 4):
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces, color=True, resize=1.0,
        slice_=(slice(0, 250), slice(0, 250)), data_home=str(config.DATA_DIR / "lfw"),
    )
    return lfw.images.astype(np.uint8), lfw.target, lfw.target_names  # RGB


def _save(img_rgb: np.ndarray, name: str) -> str:
    folder = config.DATA_DIR / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return str(path)


def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cheap stand-in for a different camera: brightness/contrast shift, blur, mild downscale."""
    out = img.astype(np.float32) * rng.uniform(0.7, 1.25) + rng.uniform(-20, 20)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    scale = rng.uniform(0.6, 0.9)
    small = cv2.resize(out, None, fx=scale, fy=scale)
    return cv2.resize(small, (img.shape[1], img.shape[0]))


def build_dataset(n_persons: int = 6, seed: int = 7, now: datetime | None = None):
    """Return (profiles, sighting_specs). Profiles are *unsaved* MissingPerson objects."""
    rng = random.Random(seed)
    now = now or datetime.utcnow()
    images, target, _ = _load_lfw()
    identities = [int(i) for i in sorted(set(target)) if (target == i).sum() >= 4]
    rng.shuffle(identities)
    chosen = identities[:n_persons]

    cities = ["Melbourne", "Sydney", "Brisbane", "Perth"]
    profiles, specs = [], []
    for idx, ident in enumerate(chosen):
        city = cities[idx % len(cities)]
        anchors = [p for p in gazetteer.PLACES if p.city == city]
        home = rng.choice(anchors)
        outfit = rng.choice(OUTFITS)
        last_seen = now - timedelta(hours=rng.randint(30, 60))
        photo_idx = [int(i) for i in np.where(target == ident)[0]]
        rng.shuffle(photo_idx)
        profile_photo, sighting_photos = photo_idx[0], photo_idx[1:3]

        person = MissingPerson(
            name=f"{rng.choice(FIRST)} {rng.choice(LAST)} (synthetic)",
            age=rng.randint(16, 70),
            description=f"Adult last seen in {city}",
            clothing=outfit, last_place=home.name, last_lat=home.lat, last_lng=home.lng,
            last_seen_at=last_seen,
            photo_path=_save(images[profile_photo], f"profile_{idx}.jpg"),
        )
        vec = embedder.embed_file(person.photo_path)
        person.embedding = vec.tolist() if vec is not None else None
        profiles.append(person)

        near = sorted(anchors, key=lambda p: gazetteer.haversine_km(home.lat, home.lng, p.lat, p.lng))[1:4]
        trail = [home, *near[:2]]
        for step, place in enumerate(trail):
            when = last_seen + timedelta(hours=4 + step * 5)
            text = rng.choice(TRUE_TEMPLATES).format(
                place=place.name, c0=outfit[0], c1=outfit[1], t=_fmt_time(when))
            photo = _augment(images[sighting_photos[step % len(sighting_photos)]], rng) if step < 2 else None
            specs.append({"truth": idx, "text": text, "seen_at": when, "photo": photo, "kind": "true"})
        for _ in range(3):  # decoys: wrong outfit, other identity, hedged
            other = rng.choice([i for i in identities if i != ident])
            oi = rng.choice([int(i) for i in np.where(target == other)[0]])
            place = rng.choice(anchors)
            when = last_seen + timedelta(hours=rng.randint(3, 20))
            text = rng.choice(DECOY_TEMPLATES).format(
                place=place.name, c0=rng.choice([o for o in OUTFITS if o != outfit])[0], t=_fmt_time(when))
            specs.append({"truth": None, "text": text, "seen_at": when,
                          "photo": _augment(images[oi], rng), "kind": "decoy"})
        # one far-away report that is not physically plausible
        far_city = rng.choice([c for c in cities if c != city])
        far = rng.choice([p for p in gazetteer.PLACES if p.city == far_city])
        when = last_seen + timedelta(hours=2)
        specs.append({"truth": None, "kind": "impossible", "seen_at": when, "photo": None,
                      "text": f"Definitely saw them at {far.name}, in a {outfit[0]}, around {_fmt_time(when)}."})
    return profiles, specs


def save_spec_photo(spec: dict, i: int) -> str | None:
    return _save(spec["photo"], f"sighting_{i}.jpg") if spec.get("photo") is not None else None
