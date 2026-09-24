"""SQLAlchemy models. Embeddings are stored as JSON lists so SQLite and PostgreSQL both work."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
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

    leads: Mapped[list["Lead"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class Sighting(Base):
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

    person: Mapped[MissingPerson] = relationship(back_populates="leads")
    sighting: Mapped[Sighting] = relationship(back_populates="leads")


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        kwargs = {"connect_args": {"check_same_thread": False}} if config.DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(config.DATABASE_URL, **kwargs)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(engine())


def session():
    engine()
    return _Session()
