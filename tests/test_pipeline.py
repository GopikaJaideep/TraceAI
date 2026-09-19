from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from traceai import pipeline, scoring
from traceai.db import Base, MissingPerson
from traceai.nlp import textsim

LAST_SEEN = datetime(2026, 9, 18, 12, 0)
NOW = datetime(2026, 9, 19, 12, 0)


@pytest.fixture()
def session(monkeypatch):
    monkeypatch.setattr(textsim, "similarity", lambda a, b: 0.5)  # no model download in tests
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(engine)()
    s.add(MissingPerson(
        name="Test (synthetic)", age=30, description="Adult", clothing=["grey hoodie"],
        last_place="Flinders Street Station", last_lat=-37.8183, last_lng=144.9671,
        last_seen_at=LAST_SEEN,
    ))
    s.commit()
    yield s
    s.close()


def test_predating_sighting_scores_zero_without_a_location(session):
    # No place in the text, so proximity cannot be computed; chronology must still reject it.
    _, leads = pipeline.ingest_sighting(
        session, text="Definitely saw them in a grey hoodie.",
        seen_at=LAST_SEEN - timedelta(hours=5), reported_at=NOW,
    )
    assert leads[0].score == 0.0
    assert any("predates" in n for n in leads[0].notes)


def test_predating_check_helper_needs_no_coordinates():
    score, note = scoring.proximity_component(
        -37.8, 144.9, LAST_SEEN, None, None, LAST_SEEN - timedelta(hours=1)
    )
    assert score == 0.0 and "predates" in note


def test_timezone_aware_seen_at_is_normalised(session):
    # 20:00+10:00 is 10:00 UTC, i.e. 22h after LAST_SEEN; must not raise on naive/aware mixing.
    aware = datetime(2026, 9, 18, 22, 0, tzinfo=timezone(timedelta(hours=10)))
    sighting, leads = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond in a grey hoodie", seen_at=aware, reported_at=NOW,
    )
    assert sighting.seen_at == datetime(2026, 9, 18, 12, 0)
    assert sighting.seen_at.tzinfo is None
    assert leads[0].score > 0


def test_timezone_aware_reported_at_is_normalised(session):
    aware = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    sighting, _ = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond at 2pm", reported_at=aware,
    )
    assert sighting.reported_at == NOW


def test_explicit_seen_at_is_recorded_in_extraction_and_credibility(session):
    seen = LAST_SEEN + timedelta(hours=3)
    _, prose_leads = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond in a grey hoodie at 3pm", reported_at=NOW,
    )
    structured, leads = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond in a grey hoodie", seen_at=seen, reported_at=NOW,
    )
    assert structured.extracted["seen_at"] == seen.isoformat()
    # Same evidence, so credibility must not depend on whether the time came from prose or a field.
    assert leads[0].components["credibility"] == prose_leads[0].components["credibility"]


def test_coordinates_replace_the_text_place(session):
    # Text says Richmond but the supplied coordinates are at Fremantle (Perth).
    sighting, _ = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond", lat=-32.0569, lng=115.7439, reported_at=NOW,
    )
    assert sighting.extracted["place"] == "Fremantle"
    assert (sighting.lat, sighting.lng) == (-32.0569, 115.7439)


def test_coordinates_far_from_any_known_place_leave_it_unlabelled(session):
    sighting, _ = pipeline.ingest_sighting(
        session, text="Saw someone near Richmond", lat=-25.0, lng=133.0, reported_at=NOW,
    )
    assert sighting.extracted["place"] is None
