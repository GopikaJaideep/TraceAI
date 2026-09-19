from datetime import datetime, timedelta

from traceai import scoring
from traceai.geo import gazetteer
from traceai.geo.clustering import SightingPoint, cluster_sightings, corridor_text
from traceai.nlp import extractor

NOW = datetime(2026, 9, 19, 21, 0)


def test_gazetteer_prefers_longest_form():
    assert gazetteer.resolve("saw him at Flinders Street Station").name == "Flinders Street Station"
    assert gazetteer.resolve("down on flinders st").name == "Flinders Street Station"
    assert gazetteer.resolve("somewhere in the outback") is None


def test_extracts_place_clothing_time():
    ex = extractor.extract(
        "Saw an individual near Flinders Street Station wearing a grey hoodie around 7 pm.", NOW
    )
    assert ex.place_text == "Flinders Street Station"
    assert "grey hoodie" in ex.clothing
    assert ex.seen_at == datetime(2026, 9, 19, 19, 0)


def test_future_clock_time_rolls_back_a_day():
    ex = extractor.extract("Someone at Fitzroy at 11pm", datetime(2026, 9, 19, 2, 0))
    assert ex.seen_at == datetime(2026, 9, 18, 23, 0)


def test_hedging_lowers_credibility():
    sure = extractor.extract("Definitely saw them at Fitzroy in a red jacket at 5pm", NOW)
    unsure = extractor.extract("Maybe saw someone at Fitzroy, not sure, hard to tell", NOW)
    assert scoring.credibility_component(sure, False) > scoring.credibility_component(unsure, False)


def _score(lat, lng, seen_at, cosine=None):
    ex = extractor.Extraction(place_text="x", lat=lat, lng=lng, seen_at=seen_at)
    return scoring.score_lead(
        person_lat=-37.8183, person_lng=144.9671, last_seen_at=NOW - timedelta(hours=10),
        ex=ex, seen_at=seen_at, now=NOW, face_cosine=cosine, description_sim=0.5, has_photo=cosine is not None,
    )


def test_close_sighting_outranks_far_sighting():
    near = _score(-37.8241, 144.9903, NOW - timedelta(hours=2))
    far = _score(-37.6690, 144.8410, NOW - timedelta(hours=2))
    assert near.score > far.score


def test_impossible_journey_scores_zero_even_with_perfect_face():
    # Melbourne -> Sydney (~700 km) in one hour
    sydney = _score(-33.8688, 151.2093, NOW - timedelta(hours=9), cosine=0.9)
    assert sydney.score == 0.0
    assert any("plausible" in n for n in sydney.notes)


def test_missing_signals_do_not_penalise():
    text_only = _score(-37.8241, 144.9903, NOW - timedelta(hours=2))
    with_face = _score(-37.8241, 144.9903, NOW - timedelta(hours=2), cosine=0.65)
    assert text_only.components["image"] is None
    assert with_face.score > text_only.score


def test_clusters_and_corridor():
    t0 = NOW - timedelta(hours=8)
    pts = [
        SightingPoint(1, -37.8183, 144.9671, t0),
        SightingPoint(2, -37.8190, 144.9680, t0 + timedelta(minutes=30)),
        SightingPoint(3, -37.8241, 144.9903, t0 + timedelta(hours=4)),
        SightingPoint(4, -37.8245, 144.9905, t0 + timedelta(hours=4, minutes=20)),
    ]
    clusters = cluster_sightings(pts)
    assert sorted(c.size for c in clusters) == [2, 2]
    assert corridor_text(clusters) == "Flinders Street Station → Richmond"
