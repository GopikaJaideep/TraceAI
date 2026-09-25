"""SQLAlchemy models. Embeddings are stored as JSON lists so SQLite and PostgreSQL both work."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from traceai import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC keeps SQLite/Postgres alike


def to_naive_utc(dt: datetime | None) -> datetime | None:
    """Timezone-aware datetimes (e.g. ISO strings with an offset or Z) -> naive UTC."""
    if dt is None or dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Law-enforcement / NGO staff. The public never has an account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20), default="officer")  # officer | admin
    agency: Mapped[str] = mapped_column(String(120), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class MissingPerson(Base):
    """A *synthetic* profile. No real missing-person data is ever stored."""

    __tablename__ = "missing_persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, default="")
    clothing: Mapped[list] = mapped_column(JSON, default=list)
    last_place: Mapped[str] = mapped_column(String(120))
    last_lat: Mapped[float] = mapped_column(Float)
    last_lng: Mapped[float] = mapped_column(Float)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)
    photo_path: Mapped[str | None] = mapped_column(String(500))
    embedding: Mapped[list | None] = mapped_column(JSON)

    # Governance: a case is only ever created by an officer, from a real police report.
    police_reference: Mapped[str] = mapped_column(String(60), default="")
    status: Mapped[str] = mapped_column(String(10), default="open")  # open | closed
    published: Mapped[bool] = mapped_column(Boolean, default=False)  # appears on the public appeals page
    public_summary: Mapped[str] = mapped_column(Text, default="")
    face_matching_authorised: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    dv_screened_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    dv_screened_at: Mapped[datetime | None] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)

    leads: Mapped[list["Lead"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class CaseAccess(Base):
    """Need-to-know: an officer sees only the cases they have been granted."""

    __tablename__ = "case_access"
    __table_args__ = (UniqueConstraint("user_id", "person_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("missing_persons.id"), index=True)
    granted_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Sighting(Base):
    """A witness report. `source` is "public" for tips from the general public."""

    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reported_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    seen_at: Mapped[datetime | None] = mapped_column(DateTime)
    text: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str | None] = mapped_column(String(500))
    extracted: Mapped[dict] = mapped_column(JSON, default=dict)
    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)
    embedding: Mapped[list | None] = mapped_column(JSON)

    source: Mapped[str] = mapped_column(String(10), default="officer")  # officer | public
    reference: Mapped[str | None] = mapped_column(String(30), unique=True)  # receipt shown to the reporter
    appeal_id: Mapped[int | None] = mapped_column(Integer)  # the appeal the reporter was looking at
    contact: Mapped[str | None] = mapped_column(String(120))
    source_hash: Mapped[str | None] = mapped_column(String(32), index=True)  # keyed hash, never the raw IP
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    flag_reason: Mapped[str] = mapped_column(String(200), default="")

    leads: Mapped[list["Lead"]] = relationship(back_populates="sighting", cascade="all, delete-orphan")


class Lead(Base):
    """One (sighting, missing person) pairing with its priority score."""

    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("missing_persons.id"), index=True)
    sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"), index=True)
    score: Mapped[float] = mapped_column(Float, index=True)
    components: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    review: Mapped[str] = mapped_column(String(12), default="unreviewed")  # unreviewed | useful | dismissed
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)

    person: Mapped[MissingPerson] = relationship(back_populates="leads")
    sighting: Mapped[Sighting] = relationship(back_populates="leads")


class AuditLog(Base):
    """Append-only, hash-chained record of who did what. See traceai.api.audit."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    user_id: Mapped[int | None] = mapped_column(Integer)
    username: Mapped[str] = mapped_column(String(80), default="")
    action: Mapped[str] = mapped_column(String(60))
    object_type: Mapped[str] = mapped_column(String(30), default="")
    object_id: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    hash: Mapped[str] = mapped_column(String(64), default="")


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        kwargs = {"connect_args": {"check_same_thread": False}} if config.DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(config.DATABASE_URL, **kwargs)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


# Arbitrary constant shared by every TraceAI process; only its equality matters.
_SCHEMA_LOCK_ID = 7_264_190_311


def init_db() -> None:
    """Create any missing tables. Safe to call from several processes starting at once.

    The public and officer APIs both call this at startup, and in Docker Compose they start together. On
    PostgreSQL two concurrent CREATE TABLE statements race and one crashes with a duplicate `pg_type` key,
    so schema creation takes a transaction-scoped advisory lock: the second process waits, then finds the
    tables already there.
    """
    eng = engine()
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": _SCHEMA_LOCK_ID})
        Base.metadata.create_all(conn)


def session():
    engine()
    return _Session()
