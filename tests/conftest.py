"""Test environment. Must set config *before* any traceai import, hence at module top level."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="traceai-tests-")
os.environ["TRACEAI_DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["TRACEAI_FACE_BACKEND"] = "none"
os.environ["TRACEAI_SECRET_KEY"] = "test-secret-key-0123456789-0123456789-abcdef"

import io  # noqa: E402
from datetime import timedelta  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

PASSWORD = "correct horse battery"


@pytest.fixture()
def env(monkeypatch):
    """Fresh database, cleared rate limits, and no ML model downloads."""
    from traceai import db
    from traceai.api import officer_app, public_app
    from traceai.db import Base
    from traceai.nlp import textsim

    monkeypatch.setattr(textsim, "similarity", lambda a, b: 0.5)
    Base.metadata.drop_all(db.engine())
    Base.metadata.create_all(db.engine())
    for limiter in (officer_app.LOGIN_WINDOW, public_app.SHORT_WINDOW, public_app.DAY_WINDOW):
        limiter.reset()
    return db


@pytest.fixture()
def officer_client(env):
    from traceai.api import officer_app

    return TestClient(officer_app.app)


def public_client(ip="10.0.0.1"):
    from traceai.api import public_app

    return TestClient(public_app.app, client=(ip, 50000))


@pytest.fixture()
def pub(env):
    return public_client()


def make_user(username, role="officer", password=PASSWORD):
    from traceai import db
    from traceai.api import security
    from traceai.db import User

    s = db.session()
    user = User(username=username, password_hash=security.hash_password(password), role=role, agency="Test")
    s.add(user)
    s.commit()
    uid = user.id
    s.close()
    return uid


def login(client, username, password=PASSWORD):
    r = client.post("/officer/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


CASE_FORM = dict(
    name="Test Person (synthetic)", police_reference="SYN-9001", family_violence_screened="true",
    last_place="Fitzroy", age="30", description="Adult", clothing="grey hoodie, blue jeans",
    publish="false", public_summary="",
)


def open_case(client, headers, files=None, **overrides):
    from traceai.db import utcnow

    data = dict(CASE_FORM, last_seen_at=(utcnow() - timedelta(hours=10)).isoformat(), **overrides)
    return client.post("/officer/cases", data=data, files=files, headers=headers)


def jpeg_bytes(size=(64, 64), exif=False):
    img = Image.new("RGB", size, (120, 80, 40))
    buf = io.BytesIO()
    if exif:
        e = Image.Exif()
        e[0x010F] = "SecretCameraMaker"  # Make
        e[0x8825] = {1: "S", 2: (37.0, 48.0, 0.0), 3: "E", 4: (144.0, 58.0, 0.0)}  # GPS block
        img.save(buf, "JPEG", exif=e)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()
