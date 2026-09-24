"""Streamlit dashboard: streamlit run dashboard/app.py"""
import os

import folium
import httpx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

API = os.getenv("TRACEAI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="TraceAI", page_icon="🧭", layout="wide")


@st.cache_data(ttl=5)
def api_get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def score_colour(score: float) -> str:
    return "red" if score >= 0.6 else "orange" if score >= 0.4 else "blue"


st.title("🧭 TraceAI")
try:
    health = api_get("/health")
except Exception as exc:
    st.error(f"Cannot reach the TraceAI API at {API}: {exc}")
    st.stop()

st.warning(health["disclaimer"])
if not health["face_backend"]:
    st.info("Face similarity is off (InsightFace unavailable). Leads use text, location, recency and credibility only.")

persons = api_get("/persons")
if not persons:
    st.info("No profiles yet. Run `python -m scripts.seed` to load synthetic data.")
    st.stop()

with st.sidebar:
    st.header("Profile")
    by_label = {f"{p['name']}": p for p in persons}
    person = by_label[st.selectbox("Synthetic missing-person profile", list(by_label))]
    if person["has_photo"]:
        st.image(f"{API}/persons/{person['id']}/photo", width=180)
    st.write(f"**Age:** {person['age']}")
    st.write(f"**Last seen:** {person['last_place']} at {person['last_seen_at'][:16].replace('T', ' ')}")
    st.write(f"**Clothing:** {', '.join(person['clothing'])}")
    min_score = st.slider("Minimum lead score", 0.0, 1.0, 0.25, 0.05)

analysis = api_get(f"/persons/{person['id']}/analysis", min_score=min_score)
leads, clusters = analysis["leads"], analysis["clusters"]

c1, c2, c3 = st.columns(3)
c1.metric("Leads above threshold", len(leads))
c2.metric("Spatial clusters", len(clusters))
c3.metric("Top lead score", f"{leads[0]['score']:.2f}" if leads else "—")
if analysis["corridor"]:
    st.success(f"Inferred movement corridor: **{analysis['corridor']}**")

map_col, list_col = st.columns([3, 2])
with map_col:
    m = folium.Map(location=[person["last_lat"], person["last_lng"]], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker([person["last_lat"], person["last_lng"]], tooltip="Last known location",
                  icon=folium.Icon(color="black", icon="star")).add_to(m)
    for lead in leads:
        if lead["lat"] is None:
            continue
        folium.CircleMarker(
            [lead["lat"], lead["lng"]], radius=6 + 10 * lead["score"], color=score_colour(lead["score"]),
            fill=True, fill_opacity=0.6,
            tooltip=f"score {lead['score']:.2f} · {lead['place'] or 'unresolved place'}",
            popup=lead["text"],
        ).add_to(m)
    for c in clusters:
        if c["size"] >= 2:
            folium.Circle([c["lat"], c["lng"]], radius=1500, color="purple", fill=False,
                          tooltip=f"cluster near {c['place']} ({c['size']} sightings)").add_to(m)
    path = sorted((c for c in clusters if c["first_seen"]), key=lambda c: c["first_seen"])
    if len(path) > 1:
        folium.PolyLine([[c["lat"], c["lng"]] for c in path], color="purple", weight=3, dash_array="6").add_to(m)
    st_folium(m, height=520, use_container_width=True, returned_objects=[])

with list_col:
    st.subheader("Prioritised leads")
    if not leads:
        st.write("No leads above the threshold.")
    for lead in leads[:12]:
        with st.expander(f"{lead['score']:.2f} · {lead['place'] or 'unknown place'}"):
            st.write(lead["text"])
            comp = pd.Series({k: v for k, v in lead["components"].items() if v is not None}, name="score")
            st.bar_chart(comp, height=140)
            for note in lead["notes"]:
                st.caption(note)
            if lead["has_image"]:
                st.image(f"{API}/sightings/{lead['sighting_id']}/image", width=140)

st.divider()
st.subheader("Submit a new sighting")
with st.form("sighting"):
    text = st.text_area("Witness report", placeholder="Saw someone near Flinders Street Station in a grey hoodie around 7pm.")
    image = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Analyse")
if submitted and text.strip():
    files = {"image": (image.name, image.getvalue(), image.type)} if image else None
    r = httpx.post(f"{API}/sightings", data={"text": text}, files=files, timeout=120)
    if r.is_success:
        st.cache_data.clear()
        top = sorted(r.json(), key=lambda l: -l["score"])[:3]
        st.write("Best-matching profiles for this report:")
        names = {p["id"]: p["name"] for p in persons}
        for l in top:
            st.write(f"- **{names.get(l['person_id'], l['person_id'])}** — score {l['score']:.2f} ({'; '.join(l['notes']) or 'no notes'})")
    else:
        st.error(r.text)
