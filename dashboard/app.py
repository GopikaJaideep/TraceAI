"""Officer console (Streamlit): streamlit run dashboard/app.py

Talks only to the officer API. Nothing here is reachable without logging in, and the API enforces
access again on every call, so this UI is a convenience, not a security boundary.
"""
import os
from datetime import datetime, time, timedelta

import folium
import httpx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

API = os.getenv("TRACEAI_API_URL", "http://localhost:8010")

st.set_page_config(page_title="TraceAI officer console", page_icon="🧭", layout="wide")


def call(method: str, path: str, **kw):
    """Authenticated request. A 401 means the session expired: drop it and ask for a fresh login."""
    headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
    r = httpx.request(method, f"{API}{path}", headers=headers, timeout=60, **kw)
    if r.status_code == 401:
        st.session_state.clear()
        st.rerun()
    return r


def logout():
    st.session_state.clear()


def login_screen():
    st.title("🧭 TraceAI officer console")
    st.caption("Authorised law-enforcement and partner-agency users only. All access is logged.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            r = httpx.post(f"{API}/officer/login", json={"username": username, "password": password}, timeout=30)
            if r.is_success:
                st.session_state.update(token=r.json()["token"], user=r.json())
                st.rerun()
            elif r.status_code == 429:
                st.error("Too many attempts. Wait a minute and try again.")
            else:
                st.error("Incorrect username or password.")


def score_colour(score: float) -> str:
    return "red" if score >= 0.6 else "orange" if score >= 0.4 else "blue"


def new_case_form():
    with st.form("new_case", clear_on_submit=True):
        st.caption("Cases are opened from a real police report only.")
        name = st.text_input("Name")
        ref = st.text_input("Police reference number")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 0, 120, 30)
        place = c2.text_input("Last-known place", help="e.g. Flinders Street Station")
        d, t = st.columns(2)
        seen_date = d.date_input("Last seen (date)")
        seen_time = t.time_input("Last seen (time)", time(12, 0))
        clothing = st.text_input("Clothing (comma-separated)", "grey hoodie, blue jeans")
        description = st.text_area("Description")
        photo = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
        lawful = st.checkbox("This photo was lawfully obtained; enable face matching for this case", disabled=False)
        fv = st.checkbox("Family-violence and protection-order screening has been completed")
        publish = st.checkbox("Publish a public appeal")
        summary = st.text_area("Public summary (shown to the public if published)")
        if st.form_submit_button("Open case"):
            data = dict(
                name=name, police_reference=ref, age=int(age), last_place=place, description=description,
                last_seen_at=datetime.combine(seen_date, seen_time).isoformat(), clothing=clothing,
                family_violence_screened=str(fv).lower(), photo_lawfully_obtained=str(lawful).lower(),
                publish=str(publish).lower(), public_summary=summary,
            )
            files = {"photo": (photo.name, photo.getvalue(), photo.type)} if photo else None
            r = call("POST", "/officer/cases", data=data, files=files)
            if r.is_success:
                st.success(f"Case {r.json()['id']} opened.")
                st.rerun()
            else:
                st.error(r.json().get("detail", r.text))


def case_view(case):
    analysis = call("GET", f"/officer/cases/{case['id']}/analysis", params={"min_score": st.session_state.get("min_score", 0.25)})
    if not analysis.is_success:
        st.error("Could not load this case.")
        return
    a = analysis.json()
    leads, clusters = a["leads"], a["clusters"]

    if case["status"] == "closed":
        st.info("This case is closed. Its photo, face data and leads have been deleted.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Leads", len(leads))
    m2.metric("From the public", sum(l["source"] == "public" for l in leads))
    m3.metric("Flagged", sum(l["flagged"] for l in leads))
    m4.metric("Top score", f"{leads[0]['score']:.2f}" if leads else "-")
    if a["corridor"]:
        st.success(f"Inferred movement corridor: **{a['corridor']}**")
    st.caption("Leads are prompts for human review, not identifications. Confirm every lead before acting on it.")

    map_col, list_col = st.columns([3, 2])
    with map_col:
        m = folium.Map(location=[case["last_lat"], case["last_lng"]], zoom_start=12)
        folium.Marker([case["last_lat"], case["last_lng"]], tooltip="Last known location",
                      icon=folium.Icon(color="black", icon="star")).add_to(m)
        for l in leads:
            if l["lat"] is None:
                continue
            folium.CircleMarker(
                [l["lat"], l["lng"]], radius=6 + 10 * l["score"], color="gray" if l["flagged"] else score_colour(l["score"]),
                fill=True, fill_opacity=0.6, tooltip=f"{l['score']:.2f} · {l['place'] or 'unresolved'}", popup=l["text"],
            ).add_to(m)
        for c in clusters:
            if c["size"] >= 2:
                folium.Circle([c["lat"], c["lng"]], radius=1500, color="purple", fill=False).add_to(m)
        st_folium(m, height=480, use_container_width=True, returned_objects=[])

    with list_col:
        st.subheader("Leads to review")
        for l in leads[:15]:
            badge = "🌐 public" if l["source"] == "public" else "👮 officer"
            flag = " ⚠️" if l["flagged"] else ""
            with st.expander(f"{l['score']:.2f} · {l['place'] or 'unknown place'} · {badge}{flag} · {l['review']}"):
                if l["flagged"]:
                    st.warning(f"Treat with extra care: {l['flag_reason']}")
                st.write(l["text"])
                if l["reference"]:
                    st.caption(f"Reference {l['reference']}" + (f" · contact: {l['contact']}" if l["contact"] else ""))
                comp = pd.Series({k: v for k, v in l["components"].items() if v is not None}, name="score")
                st.bar_chart(comp, height=120)
                for note in l["notes"]:
                    st.caption(note)
                if l["has_image"]:
                    img = call("GET", f"/officer/leads/{l['id']}/image")
                    if img.is_success:
                        st.image(img.content, width=140)
                b1, b2 = st.columns(2)
                if b1.button("Useful", key=f"u{l['id']}"):
                    call("PATCH", f"/officer/leads/{l['id']}", json={"review": "useful"})
                    st.rerun()
                if b2.button("Dismiss", key=f"d{l['id']}"):
                    call("PATCH", f"/officer/leads/{l['id']}", json={"review": "dismissed"})
                    st.rerun()

    st.divider()
    st.subheader("Case actions")
    pub = case["published"]
    c1, c2 = st.columns(2)
    with c1:
        if pub and st.button("Withdraw public appeal"):
            call("POST", f"/officer/cases/{case['id']}/publish", json={"published": False})
            st.rerun()
        if not pub:
            summary = st.text_input("Public summary", key="pubsum")
            if st.button("Publish appeal"):
                r = call("POST", f"/officer/cases/{case['id']}/publish", json={"published": True, "public_summary": summary})
                st.error(r.json()["detail"]) if not r.is_success else st.rerun()
    with c2:
        confirm = st.checkbox("The person has been located and is safe (or the case is otherwise resolved)")
        if st.button("Close case and delete its data", disabled=not confirm, type="primary"):
            call("POST", f"/officer/cases/{case['id']}/close")
            st.rerun()

    with st.expander("Record a phone-in or walk-in report"):
        with st.form("log"):
            text = st.text_area("Report")
            if st.form_submit_button("Analyse") and text.strip():
                r = call("POST", "/officer/sightings", data={"text": text})
                st.write(f"{len(r.json())} of your cases matched something." if r.is_success else r.text)


def admin_view():
    st.subheader("Users")
    users = call("GET", "/officer/admin/users").json()
    st.dataframe(pd.DataFrame(users), hide_index=True)
    with st.expander("Add a user"):
        with st.form("adduser", clear_on_submit=True):
            u = st.text_input("Username")
            p = st.text_input("Temporary password (12+ characters)", type="password")
            role = st.selectbox("Role", ["officer", "admin"])
            agency = st.text_input("Agency")
            if st.form_submit_button("Create"):
                r = call("POST", "/officer/admin/users", json={"username": u, "password": p, "role": role, "agency": agency})
                st.success("Created") if r.is_success else st.error(r.json()["detail"])
    with st.expander("Grant a user access to a case"):
        with st.form("grant"):
            case_id = st.number_input("Case ID", 1, step=1)
            uid = st.selectbox("User", [u["id"] for u in users], format_func=lambda i: next(x["username"] for x in users if x["id"] == i))
            if st.form_submit_button("Grant"):
                r = call("POST", f"/officer/admin/cases/{int(case_id)}/access", json={"user_id": uid})
                st.success("Granted") if r.is_success else st.error(r.json()["detail"])
    st.subheader("Audit log")
    v = call("GET", "/officer/admin/audit/verify").json()
    (st.success if v["intact"] else st.error)(
        "Audit chain verified: no tampering detected." if v["intact"] else f"Audit chain broken at entry {v['first_bad_id']}."
    )
    st.dataframe(pd.DataFrame(call("GET", "/officer/admin/audit", params={"limit": 200}).json()), hide_index=True)


# ------------------------------------------------------------------------------------------ main --
if "token" not in st.session_state:
    login_screen()
    st.stop()

user = st.session_state["user"]
with st.sidebar:
    st.write(f"Signed in as **{user['username']}** ({user['role']})")
    st.button("Sign out", on_click=logout)
    st.session_state["min_score"] = st.slider("Minimum lead score", 0.0, 1.0, 0.25, 0.05)

st.title("🧭 TraceAI officer console")
st.warning("Research prototype using synthetic data only. Leads need human review; this is not an identification system.")
tabs = st.tabs(["Cases", "Admin"] if user["role"] == "admin" else ["Cases"])

with tabs[0]:
    cases = call("GET", "/officer/cases").json()
    with st.sidebar:
        with st.expander("Open a new case"):
            new_case_form()
        if cases:
            chosen = st.selectbox("Case", cases, format_func=lambda c: f"{c['police_reference']} · {c['name']}" + (" (closed)" if c["status"] == "closed" else ""))
    if not cases:
        st.info("You have no cases yet. Open one from the sidebar" + (", or grant yourself access under Admin." if user["role"] == "admin" else "."))
    else:
        if chosen["has_photo"] and chosen["status"] == "open":
            photo = call("GET", f"/officer/cases/{chosen['id']}/photo")
            if photo.is_success:
                st.sidebar.image(photo.content, width=160)
        st.sidebar.write(f"**Last seen:** {chosen['last_place']}")
        st.sidebar.write(f"**Clothing:** {', '.join(chosen['clothing'])}")
        st.sidebar.write(f"**Face matching:** {'on' if chosen['face_matching_authorised'] else 'off'}")
        case_view(chosen)

if user["role"] == "admin":
    with tabs[1]:
        admin_view()
