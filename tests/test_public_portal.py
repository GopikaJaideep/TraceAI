"""Public-side safeguards: what the public can see, what a tip returns, and abuse controls."""
import os
from datetime import timedelta

from PIL import Image

from conftest import PASSWORD, jpeg_bytes, login, make_user, open_case, public_client

TIP = "Saw someone near Fitzroy in a grey hoodie."


def _post(client, text=TIP, consent="true", **extra):
    return client.post("/public/tips", data={"text": text, "consent": consent, **extra})


def _published_case(officer_client):
    make_user("alice")
    h = login(officer_client, "alice")
    return h, open_case(officer_client, h, publish="true", public_summary="Demo appeal, last seen in Melbourne.").json()


def test_public_app_exposes_no_officer_routes():
    from traceai.api import public_app

    paths = {r.path for r in public_app.app.routes}
    assert not any("officer" in p or "analysis" in p or "lead" in p for p in paths)


def test_appeals_show_only_published_open_cases_with_coarse_location(officer_client, pub):
    h, case = _published_case(officer_client)
    open_case(officer_client, h, police_reference="SYN-9002")  # not published
    appeals = pub.get("/public/appeals").json()
    assert len(appeals) == 1
    a = appeals[0]
    assert a["last_seen_city"] == "Melbourne"
    # Whitelist: nothing else about the case ever reaches the public.
    assert set(a) == {"id", "name", "age", "summary", "clothing", "last_seen_city", "last_seen_date", "has_photo"}
    assert "Fitzroy" not in str(a) and "SYN-" not in str(a)


def test_unpublished_case_photo_is_not_public(officer_client, pub):
    h, _ = _published_case(officer_client)
    other = open_case(officer_client, h, police_reference="SYN-9003",
                      files={"photo": ("p.jpg", jpeg_bytes(), "image/jpeg")}).json()
    assert pub.get(f"/public/appeals/{other['id']}/photo").status_code == 404


def test_tip_returns_only_a_receipt_never_scores_or_matches(officer_client, pub):
    _published_case(officer_client)
    r = _post(pub)
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"reference", "message"} and body["reference"].startswith("TIP-")
    for leaked in ("score", "lead", "case", "match", "Fitzroy"):
        assert leaked not in r.text.replace(body["message"], "")


def test_tip_reaches_officers_as_an_unreviewed_public_lead(officer_client, pub):
    h, case = _published_case(officer_client)
    _post(pub)
    lead = officer_client.get(f"/officer/cases/{case['id']}/analysis", headers=h).json()["leads"][0]
    assert lead["source"] == "public" and lead["review"] == "unreviewed" and lead["reference"].startswith("TIP-")


def test_consent_is_required(pub):
    assert _post(pub, consent="false").status_code == 400


def test_text_length_limits(pub):
    assert _post(pub, text="too short").status_code == 400
    assert _post(pub, text="x" * 2001).status_code == 400


def test_non_image_upload_rejected_whatever_its_name(pub):
    r = pub.post("/public/tips", data={"text": TIP, "consent": "true"},
                 files={"image": ("cute.jpg", b"MZ\x90\x00 this is really an executable", "image/jpeg")})
    assert r.status_code == 400


def test_valid_images_in_other_formats_are_rejected(pub):
    import io

    for fmt in ("GIF", "BMP", "TIFF"):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, fmt)
        r = pub.post("/public/tips", data={"text": TIP, "consent": "true"},
                     files={"image": (f"a.{fmt.lower()}", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 400, fmt


def test_oversized_image_rejected(pub):
    big = jpeg_bytes((10, 10)) + b"0" * (5 * 1024 * 1024)
    r = pub.post("/public/tips", data={"text": TIP, "consent": "true"}, files={"image": ("a.jpg", big, "image/jpeg")})
    assert r.status_code in (400, 413)


def test_uploaded_photo_is_stripped_of_exif_and_gps(officer_client, pub, env):
    from traceai.db import Sighting

    _published_case(officer_client)
    assert Image.open(__import__("io").BytesIO(jpeg_bytes(exif=True))).getexif()  # the input does carry EXIF
    r = pub.post("/public/tips", data={"text": TIP, "consent": "true"},
                 files={"image": ("a.jpg", jpeg_bytes(exif=True), "image/jpeg")})
    assert r.status_code == 202
    s = env.session()
    path = s.query(Sighting).one().image_path
    s.close()
    with Image.open(path) as saved:
        assert len(saved.getexif()) == 0
    assert os.path.basename(path) != "a.jpg"  # the client's filename is never used


def test_honeypot_looks_successful_but_stores_nothing(pub, env):
    from traceai.db import Sighting

    r = _post(pub, website="http://spam.example")
    assert r.status_code == 202
    s = env.session()
    assert s.query(Sighting).count() == 0
    s.close()


def test_rate_limit_per_source(pub):
    codes = [_post(pub, text=f"Saw someone near Fitzroy, sighting number {i}").status_code for i in range(7)]
    assert codes[:5] == [202] * 5 and codes[5:] == [429, 429]
    # A different source is unaffected.
    assert _post(public_client("10.9.9.9")).status_code == 202


def test_repeat_sources_and_duplicate_text_are_flagged_not_blocked(officer_client, env):
    from traceai.db import Sighting

    make_user("alice")
    same_source = public_client("10.1.1.1")
    for i in range(3):
        assert _post(same_source, text=f"Saw someone near Fitzroy, wave {i}").status_code == 202
    assert _post(public_client("10.2.2.2"), text="Identical text from two people near Fitzroy").status_code == 202
    assert _post(public_client("10.3.3.3"), text="Identical text from two people near Fitzroy").status_code == 202

    s = env.session()
    tips = s.query(Sighting).order_by(Sighting.id).all()
    s.close()
    assert [t.flagged for t in tips] == [False, False, True, False, True]
    assert "3 tips from one source" in tips[2].flag_reason
    assert "identical text" in tips[4].flag_reason


def test_raw_ip_is_never_stored(pub, env):
    from traceai.db import Sighting

    _post(pub)
    s = env.session()
    tip = s.query(Sighting).one()
    s.close()
    assert "10.0.0.1" not in (tip.source_hash or "") and len(tip.source_hash) == 32


def test_public_tip_retention_keeps_only_useful_ones(officer_client, env):
    from traceai import pipeline
    from traceai.cases import purge_expired_tips
    from traceai.db import Lead, Sighting, utcnow

    h, case = _published_case(officer_client)
    old = utcnow() - timedelta(days=200)
    for i in range(2):
        pipeline.ingest_sighting(env.session(), text=f"Saw someone near Fitzroy in a grey hoodie {i}",
                                 reported_at=old, source="public", extra=dict(reference=f"TIP-OLD{i}"))
    s = env.session()
    keep = s.query(Lead).join(Sighting).filter(Sighting.reference == "TIP-OLD0").one()
    keep.review = "useful"
    s.commit()

    assert purge_expired_tips(s, days=90) == 1
    assert [t.reference for t in s.query(Sighting).all()] == ["TIP-OLD0"]
    s.close()
