"""Public tip portal (Streamlit): streamlit run public_portal/app.py --server.port 8502

Read-only appeals plus a submit-only tip form. It only ever talks to the public API, which never
returns scores, matches or case details beyond what an officer published.
"""
import os

import httpx
import streamlit as st

API = os.getenv("TRACEAI_PUBLIC_API_URL", "http://localhost:8020")

st.set_page_config(page_title="Report a sighting", page_icon="🧭", layout="centered")


@st.cache_data(ttl=30)
def get(path: str):
    r = httpx.get(f"{API}{path}", timeout=20)
    r.raise_for_status()
    return r.json()


st.title("🧭 Report a sighting")
try:
    notice = get("/public/notice")
except Exception:
    st.error("The reporting service is unavailable right now. If someone is in danger, call 000.")
    st.stop()

st.info(notice["notice"])
st.caption(notice["disclaimer"])

st.header("Current appeals")
appeals = get("/public/appeals")
if not appeals:
    st.write("There are no public appeals at the moment.")
for a in appeals:
    with st.container(border=True):
        cols = st.columns([1, 3])
        if a["has_photo"]:
            cols[0].image(f"{API}/public/appeals/{a['id']}/photo", width=120)
        cols[1].subheader(a["name"])
        cols[1].write(f"Age {a['age']} · last seen {a['last_seen_date']} in {a['last_seen_city']}")
        cols[1].write(a["summary"])
        if a["clothing"]:
            cols[1].caption("Believed to be wearing: " + ", ".join(a["clothing"]))

st.header("Tell police what you saw")
with st.form("tip", clear_on_submit=True):
    text = st.text_area(
        "What did you see, where, and when?", max_chars=2000, height=150,
        help="Include the place, the time and what the person was wearing. Only report what you saw yourself.",
    )
    image = st.file_uploader("Photo (optional, JPG or PNG)", type=["jpg", "jpeg", "png"])
    contact = st.text_input("Your contact details (optional)", max_chars=120,
                            help="Only used if an officer needs to follow up. You can stay anonymous.")
    consent = st.checkbox(
        "I understand my report will be reviewed by police, that photos are stored securely and "
        "deleted after a retention period, and that knowingly making a false report is an offence."
    )
    if st.form_submit_button("Send report"):
        if not consent:
            st.error("Please tick the box to confirm you understand.")
        else:
            files = {"image": (image.name, image.getvalue(), image.type)} if image else None
            r = httpx.post(f"{API}/public/tips", data={"text": text, "consent": "true", "contact": contact},
                           files=files, timeout=60)
            if r.status_code == 202:
                st.success(f"{r.json()['message']}\n\nYour reference: **{r.json()['reference']}**")
            elif r.status_code == 429:
                st.error("You have sent several reports recently. Please wait before sending another.")
            else:
                st.error(r.json().get("detail", "Something went wrong."))
    st.caption("You will not be told whether your report matched anything: that keeps the process safe for everyone.")
