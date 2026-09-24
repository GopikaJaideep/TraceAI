"""Populate the database with synthetic cases, users and reports: python -m scripts.seed --reset

Creates one admin and one officer with freshly generated random passwords, printed once. There are no
default credentials anywhere in the code.
"""
import argparse
import os
import secrets

from traceai import db, pipeline
from traceai.api import security
from traceai.db import Base, CaseAccess, User
from traceai.synthetic import build_dataset, save_spec_photo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=6)
    ap.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    args = ap.parse_args()

    if args.reset:
        Base.metadata.drop_all(db.engine())
    db.init_db()
    session = db.session()
    if session.query(User).count():
        print("Database already seeded; use --reset to rebuild.")
        return

    passwords = {}
    for username, role in (("admin", "admin"), ("officer", "officer")):
        password = os.getenv(f"TRACEAI_SEED_{username.upper()}_PASSWORD") or secrets.token_urlsafe(12)
        passwords[username] = password
        session.add(User(username=username, password_hash=security.hash_password(password), role=role,
                         agency="Demo Police (synthetic)"))
    session.commit()
    officer = session.query(User).filter_by(username="officer").one()

    print("Building synthetic dataset (first run downloads the LFW research dataset)...")
    profiles, specs = build_dataset(args.persons)
    for p in profiles:
        p.created_by = p.dv_screened_by = officer.id
    session.add_all(profiles)
    session.commit()
    for p in profiles:
        session.add(CaseAccess(user_id=officer.id, person_id=p.id, granted_by=officer.id))
    session.commit()

    for i, spec in enumerate(specs):
        # Seeded as public tips so the review queue looks like real life.
        pipeline.ingest_sighting(
            session, text=spec["text"], seen_at=spec["seen_at"], image_path=save_spec_photo(spec, i),
            reported_at=spec["seen_at"], source="public",
            extra=dict(reference=f"TIP-SEED{i:04d}", source_hash=secrets.token_hex(8)),
        )
    print(f"Seeded {len(profiles)} synthetic cases and {len(specs)} tips.")
    print("\nLogin credentials (shown once; not stored anywhere in plain text):")
    for username, password in passwords.items():
        print(f"  {username:8} {password}")


if __name__ == "__main__":
    main()
