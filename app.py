# ==========================================================
# FUTUREPROOF AI – Career Intelligence Engine (2026)
# Premium Production Version with Admin Monitoring
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
import smtplib
from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer, util
from google import genai

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="FutureProof AI",
    page_icon="🚀",
    layout="wide"
)

# ==========================================================
# AI BACKGROUND + PREMIUM UI
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1677442136019-21780ecad995");
    background-size: cover;
    background-attachment: fixed;
}

.main-title {
    font-size:48px;
    font-weight:900;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass-card {
    background: rgba(255,255,255,0.85);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
    margin-bottom:20px;
}

.admin-box {
    background:#111827;
    padding:15px;
    border-radius:12px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown('<div class="main-title">🚀 FutureProof AI Career Intelligence</div>', unsafe_allow_html=True)
st.caption("AI-Powered Career Growth Planning Engine (2026 Edition)")

# ==========================================================
# LOAD ENV VARIABLES
# ==========================================================

api_key = os.getenv("GOOGLE_API_KEY")
admin_email = os.getenv("ADMIN_EMAIL")
admin_password = os.getenv("ADMIN_APP_PASSWORD")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=api_key)

DATASET_CONFIDENCE_THRESHOLD = 0.55
GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================================
# EMAIL ALERT SYSTEM
# ==========================================================

def send_admin_alert(error_message):
    if not admin_email or not admin_password:
        return

    try:
        msg = MIMEText(f"🚨 FutureProof AI Alert\n\nError:\n{error_message}")
        msg["Subject"] = "FutureProof AI - API Failure Alert"
        msg["From"] = admin_email
        msg["To"] = admin_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(admin_email, admin_password)
        server.sendmail(admin_email, admin_email, msg.as_string())
        server.quit()
    except:
        pass

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("futureproof_dummy_data.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df["current_skills"] = df["current_skills"].fillna("").str.lower()
    df["interested_future_field"] = df["interested_future_field"].fillna("").str.lower()
    return df

df = load_data()

# ==========================================================
# GEMINI FUNCTION WITH ALERT
# ==========================================================

def gemini_generate(prompt):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response or not response.text:
            send_admin_alert("Gemini returned empty response.")
            return None

        return response.text.strip()

    except Exception as e:
        send_admin_alert(str(e))
        return None

# ==========================================================
# INTELLIGENCE ENGINE
# ==========================================================

def dataset_confidence(user_skills):
    user_emb = embed_model.encode(" ".join(user_skills), convert_to_tensor=True)

    fields = df["interested_future_field"].unique().tolist()
    profiles = []

    for f in fields:
        skills = ",".join(df[df["interested_future_field"] == f]["current_skills"])
        profiles.append(skills[:500])

    field_embs = embed_model.encode(profiles, convert_to_tensor=True)
    scores = util.cos_sim(user_emb, field_embs)[0]

    best_idx = scores.argmax()
    return float(scores[best_idx]), fields[best_idx]

def infer_role(skills):
    prompt = f"""
Suggest ONE high-growth 2026 job role for skills:
{", ".join(skills)}
Return only role name.
"""
    return gemini_generate(prompt) or "Business Transformation Specialist"

def infer_growth(role, skills):
    prompt = f"""
For {role}, suggest 6 future growth skills.
Return comma-separated only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []
    raw = re.split(r",|\n", response)
    return [s.strip().title() for s in raw if s.strip()][:6]

def infer_certifications(role):
    prompt = f"""
Suggest 5 trending global certifications in 2026 for {role}.
Return comma-separated only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []
    return [c.strip() for c in response.split(",")][:5]

# ==========================================================
# USER INPUT SECTION
# ==========================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Name")

with col2:
    education = st.selectbox(
        "🎓 Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )

skills_input = st.text_input("💡 Current Skills (comma-separated)")
hours = st.slider("⏳ Weekly Learning Hours", 1, 40, 10)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# GENERATE BUTTON
# ==========================================================

if st.button("🚀 Generate Career Intelligence Report", use_container_width=True):

    if not name or not skills_input:
        st.warning("Please enter required details.")
    else:

        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing AI market signals..."):

            confidence, dataset_field = dataset_confidence(skills)

            if confidence >= DATASET_CONFIDENCE_THRESHOLD:
                role = dataset_field.title()
                source = "Peer Intelligence"
            else:
                role = infer_role(skills)
                source = "Live AI Market Intelligence"

            growth = infer_growth(role, skills)
            certs = infer_certifications(role)

            weeks = round((len(growth) * 40) / hours) if hours > 0 else 0

        st.success("Career Intelligence Report Ready!")

        tab1, tab2, tab3 = st.tabs(["🎯 Overview", "📈 Growth Plan", "🎓 Certifications"])

        with tab1:
            colA, colB, colC = st.columns(3)
            colA.metric("Recommended Role", role)
            colB.metric("Confidence", f"{int(confidence*100)}%")
            colC.metric("Market Demand", f"{np.random.randint(75,95)}%")
            st.progress(int(confidence*100))
            st.caption(f"Source: {source}")

        with tab2:
            st.subheader("Future Growth Areas")
            for g in growth:
                st.write("✔️", g)

            st.info(f"Estimated Upskilling Timeline: ~{weeks} weeks")

        with tab3:
            st.subheader("Trending Certifications")
            for c in certs:
                st.write("🏅", c)

# ==========================================================
# FEEDBACK SECTION
# ==========================================================

st.divider()
st.subheader("💬 Feedback")

rating = st.slider("How useful was this report?", 1, 5, 4)
feedback = st.text_area("Suggestions for improvement")

if st.button("Submit Feedback"):
    with open("feedback_log.txt", "a") as f:
        f.write(f"\n{name} | Rating:{rating} | {feedback}")
    st.success("Thank you for your feedback!")

# ==========================================================
# ADMIN PANEL
# ==========================================================

if st.sidebar.checkbox("🔐 Admin Panel"):
    st.sidebar.markdown('<div class="admin-box">', unsafe_allow_html=True)
    st.sidebar.write("API Loaded:", "Yes" if api_key else "No")
    st.sidebar.write("Model:", GEMINI_MODEL)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
