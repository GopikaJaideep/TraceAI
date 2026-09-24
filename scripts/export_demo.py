"""Export a static, synthetic-only snapshot for the public demo site: python -m scripts.export_demo

Everything in docs/demo.json is produced by the real pipeline (extraction, scoring, clustering). No photos
are exported: LFW faces are real people, so the demo shows face-match *scores* against abstract avatars.
"""
import json
from datetime import datetime
from pathlib import Path

from traceai import config, db, pipeline, scoring
from traceai.db import MissingPerson
from traceai.geo import gazetteer
from traceai.nlp import extractor

OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.json"
LETTERS = "ABCDEFGH"

# Scenario engine: a fixed reference case so every card is comparable and reproducible.
NOW = datetime(2026, 9, 19, 21, 0)
LAST_SEEN = datetime(2026, 9, 19, 12, 0)
REF = gazetteer.get("Flinders Street Station")
REF_CLOTHING = ["grey hoodie", "blue jeans"]

SCENARIOS = [
    ("Confident witness with a matching photo",
     "I'm sure I saw the person at Federation Square in a grey hoodie around 2pm.", 0.62,
     "Right place, right clothes, plausible timing, and a strong face match. This is what a top lead looks like."),
    ("Same report, no photo",
     "I'm sure I saw the person at Federation Square in a grey hoodie around 2pm.", None,
     "Without a photo the image signal is dropped and the other weights are renormalised, so a text-only report is not penalised for what it could not provide."),
    ("Hedged, vague report",
     "Maybe saw someone like that near Richmond, not sure, hard to tell from a distance.", None,
     "Hedging words lower credibility, and there is no clothing or time detail to check."),
    ("Wrong outfit",
     "Definitely saw them at Southbank in a red t-shirt around 3pm.", None,
     "Location and timing are fine, but the description contradicts the profile. Identity evidence scales the score down, so a well-located report about someone else cannot rank like a match."),
    ("Impossible journey",
     "Definitely saw them at Bondi Beach in a grey hoodie around 1pm.", 0.60,
     "About 700 km in one hour is not physically possible. The lead scores 0 even with a strong face match."),
    ("Predates the last-known time",
     "Saw them near Flinders Street Station at 9am in a grey hoodie.", 0.60,
     "The sighting is before the person was last seen. Chronology is checked even when no location is available."),
    ("Good photo, thin text",
     "Saw someone near Southbank.", 0.55,
     "A strong face match carries a sparse report, but credibility stays modest because there is little detail to verify."),
    ("Perfect text, wrong face",
     "I'm sure I saw the person at Melbourne Central in a grey hoodie around 4pm.", 0.02,
     "The photo does not match the profile, so the image component is 0 and drags an otherwise convincing report down."),
]


def scenario_cards() -> list[dict]:
    ref = MissingPerson(
        name="ref", description="Adult last seen in Melbourne", clothing=REF_CLOTHING, last_place=REF.name,
        last_lat=REF.lat, last_lng=REF.lng, last_seen_at=LAST_SEEN,
    )
    cards = []
    for title, text, cosine, why in SCENARIOS:
        ex = extractor.extract(text, NOW)
        ls = scoring.score_lead(
            person_lat=ref.last_lat, person_lng=ref.last_lng, last_seen_at=LAST_SEEN, ex=ex, seen_at=ex.seen_at,
            now=NOW, face_cosine=cosine, description_sim=pipeline.description_similarity(ref, text, ex.clothing),
            has_photo=cosine is not None,
        )
        cards.append({
            "title": title, "text": text, "why": why, "face_cosine": cosine, "score": ls.score,
            "components": ls.components, "notes": ls.notes,
            "extracted": {"place": ex.place_text, "clothing": ex.clothing,
                          "seen_at": ex.seen_at.isoformat() if ex.seen_at else None, "hedges": ex.hedges},
        })
    return cards


def case_snapshots() -> list[dict]:
    session = db.session()
    cases = []
    for i, p in enumerate(session.query(MissingPerson).order_by(MissingPerson.id)):
        result = pipeline.person_analysis(session, p.id, min_score=0.15)
        place = gazetteer.get(p.last_place)
        leads = [
            {
                "score": round(l.score, 3),
                "components": {k: (round(v, 3) if v is not None else None) for k, v in l.components.items()},
                "notes": l.notes, "text": l.sighting.text, "place": (l.sighting.extracted or {}).get("place"),
                "lat": l.sighting.lat, "lng": l.sighting.lng, "has_photo": bool(l.sighting.image_path),
                "flagged": l.sighting.flagged,
            }
            for l in result["leads"] if (l.sighting.reference or "").startswith("TIP-SEED")
        ]
        cases.append({
            "label": f"Synthetic case {LETTERS[i]}", "city": place.city if place else "Australia",
            "last_place": p.last_place, "lat": p.last_lat, "lng": p.last_lng, "clothing": p.clothing,
            "last_seen": p.last_seen_at.isoformat(), "corridor": result["corridor"], "leads": leads,
            "clusters": [{"lat": c.lat, "lng": c.lng, "place": c.place, "size": c.size} for c in result["clusters"]],
        })
    session.close()
    return cases


def main() -> None:
    data = {
        "generated_note": "Synthetic data produced by the TraceAI pipeline. No real people, cases or photos.",
        "weights": scoring.WEIGHTS, "identity_floor": scoring.IDENTITY_FLOOR,
        "scenarios": scenario_cards(), "cases": case_snapshots(),
        "disclaimer": config.DISCLAIMER,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(data['cases'])} cases, "
          f"{sum(len(c['leads']) for c in data['cases'])} leads, {len(data['scenarios'])} scenarios")


if __name__ == "__main__":
    main()
