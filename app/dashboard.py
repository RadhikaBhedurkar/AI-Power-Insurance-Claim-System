import os, requests, streamlit as st
API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Claim Review", layout="wide")
tab1, tab2 = st.tabs(["Submit claim", "Officer review"])

with tab1:
    desc = st.text_area("Describe the accident")
    age = st.number_input("Vehicle age (years)", 0, 30, 5)
    imgs = st.file_uploader("Damage photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    docs = st.file_uploader("Claim form / estimate (PDF or image)", type=["pdf", "jpg", "png"], accept_multiple_files=True)
    if st.button("Analyze") and imgs:
        files = [("images", (f.name, f.getvalue())) for f in imgs] + [("documents", (f.name, f.getvalue())) for f in docs]
        with st.spinner("Running models..."):
            r = requests.post(f"{API}/claims", data={"description": desc, "vehicle_age": age}, files=files)
        st.success(f"Submitted: {r.json().get('claim_id')}") if r.ok else st.error(r.text)

with tab2:
    claims = requests.get(f"{API}/claims").json()
    if claims:
        cid = st.selectbox("Claim", [c["id"] for c in claims])
        c = requests.get(f"{API}/claims/{cid}").json(); r = c["result"]
        st.write("Status:", c["status"])
        a, b, d = st.columns(3)
        a.metric("Severity", r["damage"]["severity"], f"conf {r['damage']['severity_conf']}")
        b.metric("Est. repair (INR)", f"{r['cost_range_inr'][0]:,} - {r['cost_range_inr'][1]:,}")
        d.metric("Accident type", r["accident"]["category"], f"conf {r['accident']['confidence']}")
        for fl in r["flags"]: st.warning(fl)
        st.caption("Estimate is model output, not a verified quote.")
        st.write("Detections", r["damage"]["detections"]); st.write("Extracted fields", r["entities"])
        with st.expander("OCR text"): st.text(r["ocr_text"])
        for f in [x for x in c["files"] if x.lower().endswith((".jpg", ".jpeg", ".png"))]:
            if os.path.exists(f): st.image(f, width=300)
        note = st.text_input("Reviewer note", c["reviewer_note"])
        cols = st.columns(3)
        for col, s in zip(cols, ("APPROVED", "REJECTED", "NEEDS_INFO")):
            if col.button(s):
                requests.patch(f"{API}/claims/{cid}/decision", data={"status": s, "note": note}); st.rerun()
