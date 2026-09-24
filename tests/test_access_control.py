"""Officer-side safeguards: authentication, need-to-know, mandatory attestations, audit, closure."""
import os

import numpy as np

from conftest import PASSWORD, login, make_user, open_case, public_client

TIP = "Saw someone near Fitzroy in a grey hoodie."


def _tip(text=TIP, ip="10.0.0.1", **form):
    return public_client(ip).post("/public/tips", data={"text": text, "consent": "true", **form})


def test_officer_routes_require_a_valid_token(officer_client):
    for method, path in [
        ("get", "/officer/cases"), ("get", "/officer/cases/1/analysis"), ("get", "/officer/me"),
        ("get", "/officer/admin/audit"), ("get", "/officer/admin/users"),
        ("get", "/officer/cases/1/photo"), ("get", "/officer/leads/1/image"),
    ]:
        assert getattr(officer_client, method)(path).status_code == 401, path
    r = officer_client.get("/officer/cases", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_login_failures_are_indistinguishable(officer_client):
    make_user("alice")
    wrong_pw = officer_client.post("/officer/login", json={"username": "alice", "password": "nope" * 5})
    no_user = officer_client.post("/officer/login", json={"username": "ghost", "password": "nope" * 5})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_account_locks_after_repeated_failures(officer_client):
    make_user("alice")
    for _ in range(5):
        officer_client.post("/officer/login", json={"username": "alice", "password": "wrong-wrong-wrong"})
    r = officer_client.post("/officer/login", json={"username": "alice", "password": PASSWORD})
    assert r.status_code == 401  # even the right password is refused while locked


def test_login_is_rate_limited(officer_client):
    codes = [officer_client.post("/officer/login", json={"username": "x", "password": "y"}).status_code
             for _ in range(12)]
    assert 429 in codes


def test_admin_endpoints_need_admin_role(officer_client):
    make_user("alice")
    r = officer_client.get("/officer/admin/users", headers=login(officer_client, "alice"))
    assert r.status_code == 403


def test_case_requires_reference_and_family_violence_screening(officer_client):
    make_user("alice")
    h = login(officer_client, "alice")
    r = open_case(officer_client, h, police_reference="  ")
    assert r.status_code == 422 and "police reference" in r.json()["detail"].lower()
    r = open_case(officer_client, h, family_violence_screened="false")
    assert r.status_code == 422 and "family-violence" in r.json()["detail"].lower()
    assert open_case(officer_client, h).status_code == 201


def test_need_to_know_between_officers_and_admins(officer_client):
    make_user("alice"), make_user("bob"), make_user("root", role="admin")
    a, b, root = (login(officer_client, u) for u in ("alice", "bob", "root"))
    case_id = open_case(officer_client, a).json()["id"]

    assert officer_client.get(f"/officer/cases/{case_id}/analysis", headers=a).status_code == 200
    # Another officer, and even an admin, get 404: the case's existence is not revealed either.
    for h in (b, root):
        assert officer_client.get(f"/officer/cases/{case_id}/analysis", headers=h).status_code == 404
        assert officer_client.get(f"/officer/cases/{case_id}/photo", headers=h).status_code == 404
    assert officer_client.get("/officer/cases", headers=b).json() == []
    assert officer_client.get(f"/officer/cases/{case_id + 99}/analysis", headers=b).status_code == 404

    bob_id = next(u["id"] for u in officer_client.get("/officer/admin/users", headers=root).json() if u["username"] == "bob")
    officer_client.post(f"/officer/admin/cases/{case_id}/access", json={"user_id": bob_id}, headers=root)
    assert officer_client.get(f"/officer/cases/{case_id}/analysis", headers=b).status_code == 200


def test_officer_cannot_review_a_lead_on_someone_elses_case(officer_client):
    make_user("alice"), make_user("bob")
    a, b = login(officer_client, "alice"), login(officer_client, "bob")
    case_id = open_case(officer_client, a).json()["id"]
    assert _tip().status_code == 202
    lead_id = officer_client.get(f"/officer/cases/{case_id}/analysis", headers=a).json()["leads"][0]["id"]
    assert officer_client.patch(f"/officer/leads/{lead_id}", json={"review": "dismissed"}, headers=b).status_code == 404
    assert officer_client.patch(f"/officer/leads/{lead_id}", json={"review": "useful"}, headers=a).status_code == 200


def test_dismissed_leads_disappear_from_the_analysis(officer_client):
    make_user("alice")
    a = login(officer_client, "alice")
    case_id = open_case(officer_client, a).json()["id"]
    _tip()
    lead = officer_client.get(f"/officer/cases/{case_id}/analysis", headers=a).json()["leads"][0]
    officer_client.patch(f"/officer/leads/{lead['id']}", json={"review": "dismissed"}, headers=a)
    assert officer_client.get(f"/officer/cases/{case_id}/analysis", headers=a).json()["leads"] == []


def test_short_password_rejected_for_new_users(officer_client):
    make_user("root", role="admin")
    h = login(officer_client, "root")
    r = officer_client.post("/officer/admin/users", json={"username": "x", "password": "short"}, headers=h)
    assert r.status_code == 422


def test_audit_log_records_access_and_detects_tampering(officer_client, env):
    from traceai.api import audit
    from traceai.db import AuditLog

    make_user("alice"), make_user("root", role="admin")
    a, root = login(officer_client, "alice"), login(officer_client, "root")
    case_id = open_case(officer_client, a).json()["id"]
    officer_client.get(f"/officer/cases/{case_id}/analysis", headers=a)

    rows = officer_client.get("/officer/admin/audit", headers=root).json()
    actions = [r["action"] for r in rows]
    assert {"login", "case.create", "case.view"} <= set(actions)
    assert officer_client.get("/officer/admin/audit/verify", headers=root).json() == {"intact": True, "first_bad_id": None}

    s = env.session()
    victim = s.query(AuditLog).filter(AuditLog.action == "case.view").one()
    victim.username = "someone-else"  # rewrite history
    s.commit()
    ok, bad = audit.verify_chain(s)
    s.close()
    assert not ok and bad == victim.id


def test_reading_the_audit_log_is_itself_audited(officer_client):
    make_user("root", role="admin")
    h = login(officer_client, "root")
    officer_client.get("/officer/admin/audit", headers=h)
    rows = officer_client.get("/officer/admin/audit", headers=h).json()
    assert any(r["action"] == "audit.view" and r["username"] == "root" for r in rows)


def test_failed_logins_are_audited(officer_client):
    make_user("alice"), make_user("root", role="admin")
    officer_client.post("/officer/login", json={"username": "alice", "password": "wrong-wrong-wrong"})
    rows = officer_client.get("/officer/admin/audit", headers=login(officer_client, "root")).json()
    assert any(r["action"] == "login.failed" and r["username"] == "alice" for r in rows)


def test_face_matching_needs_a_lawfully_obtained_photo_attestation(officer_client, monkeypatch):
    from traceai.vision import embedder

    monkeypatch.setattr(embedder, "embed_file", lambda path: np.ones(512, dtype=np.float32) / 22.6)
    make_user("alice")
    h = login(officer_client, "alice")
    from conftest import jpeg_bytes

    def files():
        return {"photo": ("p.jpg", jpeg_bytes(), "image/jpeg")}

    r = open_case(officer_client, h, files=files(), photo_lawfully_obtained="false")
    assert r.json()["face_matching_authorised"] is False
    r = open_case(officer_client, h, files=files(), photo_lawfully_obtained="true")
    assert r.json()["face_matching_authorised"] is True


def test_closing_a_case_purges_biometrics_leads_and_public_listing(officer_client, env, monkeypatch):
    from traceai.db import Lead, MissingPerson
    from traceai.vision import embedder
    from conftest import jpeg_bytes

    monkeypatch.setattr(embedder, "embed_file", lambda path: np.ones(512, dtype=np.float32) / 22.6)
    make_user("alice")
    h = login(officer_client, "alice")
    case = open_case(
        officer_client, h, files={"photo": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        photo_lawfully_obtained="true", publish="true", public_summary="Demo appeal.",
    ).json()
    _tip()
    s = env.session()
    photo = s.get(MissingPerson, case["id"]).photo_path
    assert os.path.exists(photo) and s.query(Lead).count() == 1
    assert len(public_client().get("/public/appeals").json()) == 1

    assert officer_client.post(f"/officer/cases/{case['id']}/close", headers=h).status_code == 200
    s.expire_all()
    p = s.get(MissingPerson, case["id"])
    assert p.status == "closed" and p.embedding is None and p.photo_path is None
    assert not p.face_matching_authorised and not p.published
    assert s.query(Lead).count() == 0 and not os.path.exists(photo)
    s.close()
    assert public_client().get("/public/appeals").json() == []
    assert officer_client.post(f"/officer/cases/{case['id']}/close", headers=h).status_code == 409


def test_closed_cases_receive_no_new_leads(officer_client):
    make_user("alice")
    h = login(officer_client, "alice")
    case_id = open_case(officer_client, h).json()["id"]
    officer_client.post(f"/officer/cases/{case_id}/close", headers=h)
    _tip()
    assert officer_client.get(f"/officer/cases/{case_id}/analysis", headers=h).json()["leads"] == []
