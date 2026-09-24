"""Case lifecycle: opening (with mandatory safeguards), closing (with biometric purge), retention."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from traceai import config
from traceai.db import CaseAccess, Lead, MissingPerson, Sighting, User, to_naive_utc, utcnow
from traceai.vision import embedder


class CaseError(ValueError):
    """A safeguard was not met; the message is safe to show to the officer."""


def _remove_file(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def create_case(
    db: Session, officer: User, *, name: str, age: int | None, description: str, clothing: list[str],
    last_place: str, last_lat: float, last_lng: float, last_seen_at: datetime, police_reference: str,
    family_violence_screened: bool, photo_path: str | None = None, photo_lawfully_obtained: bool = False,
    publish: bool = False, public_summary: str = "",
) -> MissingPerson:
    """Open a case. Refuses unless the officer supplies a police reference and attests that the
    family-violence / protection-order screening was done (the abuser-files-a-missing-report case).
    Face matching switches on only when the officer also attests the photo was lawfully obtained."""
    if not police_reference.strip():
        raise CaseError("A police reference number is required: cases are opened from a real report only.")
    if not family_violence_screened:
        raise CaseError(
            "Confirm the family-violence and protection-order screening has been completed before "
            "opening a case. A missing-person report can be used to locate someone who is hiding."
        )
    if publish and not public_summary.strip():
        raise CaseError("A public summary is required to publish an appeal.")
    face_ok = bool(photo_path) and photo_lawfully_obtained
    vec = embedder.embed_file(photo_path) if face_ok else None
    now = utcnow()
    person = MissingPerson(
        name=name, age=age, description=description, clothing=clothing, last_place=last_place,
        last_lat=last_lat, last_lng=last_lng, last_seen_at=to_naive_utc(last_seen_at),
        photo_path=photo_path, embedding=vec.tolist() if vec is not None else None,
        police_reference=police_reference.strip(), status="open", published=publish,
        public_summary=public_summary, face_matching_authorised=face_ok and vec is not None,
        created_by=officer.id, dv_screened_by=officer.id, dv_screened_at=now,
    )
    db.add(person)
    db.flush()
    db.add(CaseAccess(user_id=officer.id, person_id=person.id, granted_by=officer.id))
    db.commit()
    return person


def close_case(db: Session, person: MissingPerson) -> None:
    """Close a case and delete what no longer needs to exist: the photo, the face embedding, and all
    leads. Public tips that fed it are kept only if an officer marked one useful. The case record
    itself (reference, dates) stays so the audit log still resolves."""
    for lead in db.query(Lead).filter(Lead.person_id == person.id).all():
        db.delete(lead)
    _remove_file(person.photo_path)
    person.photo_path = None
    person.embedding = None
    person.face_matching_authorised = False
    person.published = False
    person.status = "closed"
    person.closed_at = utcnow()
    db.commit()


def purge_expired_tips(db: Session, days: int | None = None) -> int:
    """Delete public tips older than the retention period, unless an officer marked a lead from the
    tip useful. Returns how many were removed."""
    cutoff = utcnow() - timedelta(days=days if days is not None else config.TIP_RETENTION_DAYS)
    removed = 0
    for tip in db.query(Sighting).filter(Sighting.source == "public", Sighting.reported_at < cutoff).all():
        if any(l.review == "useful" for l in tip.leads):
            continue
        _remove_file(tip.image_path)
        db.delete(tip)
        removed += 1
    db.commit()
    return removed
