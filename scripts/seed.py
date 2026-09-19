"""Populate the database with synthetic profiles and reports: python -m scripts.seed"""
import argparse

from traceai import db, pipeline
from traceai.db import Lead, MissingPerson, Sighting
from traceai.synthetic import build_dataset, save_spec_photo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=6)
    ap.add_argument("--reset", action="store_true", help="wipe existing rows first")
    args = ap.parse_args()

    db.init_db()
    session = db.session()
    if args.reset:
        for model in (Lead, Sighting, MissingPerson):
            session.query(model).delete()
        session.commit()
    elif session.query(MissingPerson).count():
        print("Database already seeded; use --reset to rebuild.")
        return

    print("Building synthetic dataset (first run downloads the LFW research dataset)...")
    profiles, specs = build_dataset(args.persons)
    session.add_all(profiles)
    session.commit()
    for i, spec in enumerate(specs):
        pipeline.ingest_sighting(
            session, text=spec["text"], seen_at=spec["seen_at"], image_path=save_spec_photo(spec, i),
            reported_at=spec["seen_at"],
        )
    print(f"Seeded {len(profiles)} synthetic profiles and {len(specs)} sightings.")


if __name__ == "__main__":
    main()
