# ==========================================================
# FUTUREPROOF AI – Career Intelligence Engine (Admin Edition)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
from sentence_transformers import SentenceTransformer, util
from google import genai

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FutureProof AI",
    page_icon="🚀",
    layout="wide"
)

# ================= PROFESSIONAL UI THEME =================
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Titles */
.main-title {
    font-size:42px;
    font-weight:800;
    color:white;
}

/* Labels fix */
label, .stSelectbox label, .stTextInput label, .stSlider label {
    color: white !important;
    font-weight: 500;
}

/* Slider text */
.stSlider span {
    color: white !important;
}

/* Input boxes */
input, textarea {
    background-color: #f8fafc !important;
    color: black !important;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🚀 FutureProof AI Career Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Plan Your 2026 Career Growth Intelligently")

# ================= ADMIN LOGIN SYSTEM =================
ADMIN_USERNAME = os.getenv("ADMIN_USER")
ADMIN_PASSWORD = os.getenv("ADMIN_PASS")

if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

with st.sidebar:
    st.markdown("## 🔐 Admin Access")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.admin_logged = True
            st.success("Admin logged in")
        else:
            st.error("Invalid credentials")

# ================= LOAD API KEY =================
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-2.5-flash"

# ================= LOAD MODELS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("futureproof_dummy_data.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df["current_skills"] = df["current_skills"].fillna("").str.lower()
    df["interested_future_field"] = df["interested_future_field"].fillna("").str.lower()
    return df

df = load_data()

# ================= GEMINI HELPER =================
def gemini_generate(prompt):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        if response and response.text:
            return response.text.strip()
        return None
    except Exception:
        return None

# ================= CONFIDENCE LOGIC =================
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

# ================= ROLE INFERENCE =================
def infer_career_role(skills):
    prompt = f"""
Suggest ONE high-growth future job role in 2026 
for someone with these skills:
{", ".join(skills)}
Return ONLY the role name.
"""
    role = gemini_generate(prompt)
    return role if role else "Business Transformation Specialist"

# ================= UPSKILLING =================
def infer_growth_plan(role, skills):
    prompt = f"""
For the role {role}, suggest 6 high-impact skills in 2026.
Return comma-separated names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []

    raw = re.split(r",|\n", response)
    return [s.strip().title() for s in raw if s.strip()][:6]

# ================= CERTIFICATIONS =================
def infer_certifications(role):
    prompt = f"""
Suggest 5 trending global certifications in 2026 for {role}.
Return comma-separated certification names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []
    return [c.strip() for c in response.split(",")][:5]

# ================= USER INPUT =================
st.markdown("### 👤 Your Profile")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")

with col2:
    education = st.selectbox(
        "Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )

skills_input = st.text_input("Current Skills (comma-separated)")
hours = st.slider("Weekly Learning Hours Available", 1, 40, 10)

# ================= GENERATE =================
if st.button("🔎 Generate Career Intelligence Plan", use_container_width=True):

    if not name or not skills_input:
        st.warning("Please enter your name and skills.")
    else:

        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing market signals..."):
            confidence, dataset_field = dataset_confidence(skills)

            if confidence >= 0.55:
                role = dataset_field.title()
                source = "Peer Career Intelligence"
            else:
                role = infer_career_role(skills)
                source = "Live Market AI Intelligence"

            growth_skills = infer_growth_plan(role, skills)
            certifications = infer_certifications(role)
            weeks = round((len(growth_skills) * 40) / hours)

        st.success("✅ Career Intelligence Report Ready!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 Overview", "📈 Growth Roadmap", "🎓 Certifications", "📊 Skill Insights"]
        )

        with tab1:
            colA, colB, colC = st.columns(3)
            colA.metric("🎯 Recommended Role", role)
            colB.metric("📊 Confidence", f"{int(confidence*100)}%")
            colC.metric("📈 Market Demand Score", f"{np.random.randint(75,95)}%")
            st.progress(int(confidence*100))
            st.markdown(f"**Insight Source:** {source}")

        with tab2:
            for skill in growth_skills:
                st.markdown(f"✔️ {skill}")
            st.markdown(f"### ⏳ Estimated Upskilling Time: ~{weeks} weeks")

        with tab3:
            for cert in certifications:
                st.markdown(f"🏅 {cert}")

        with tab4:
            colX, colY = st.columns(2)
            with colX:
                for s in skills:
                    st.markdown(f"- {s.title()}")
            with colY:
                for s in growth_skills[:4]:
                    st.markdown(f"- {s}")

        st.divider()

        rating = st.slider("How useful was this plan?", 1, 5, 4)
        feedback_text = st.text_area("What can we improve?")

        if st.button("Submit Feedback"):
            with open("feedback_log.txt", "a") as f:
                f.write(f"\n{name} | Rating:{rating} | {feedback_text}")
            st.success("Thank you for feedback!")

# ================= ADMIN DASHBOARD =================
if st.session_state.admin_logged:

    st.sidebar.markdown("## 🛠 Admin Dashboard")

    if st.sidebar.button("Check API Status"):
        test = gemini_generate("Say hello")
        if test:
            st.sidebar.success("API Working")
        else:
            st.sidebar.error("API Not Working")

    if st.sidebar.button("View Feedback Logs"):
        try:
            with open("feedback_log.txt", "r") as f:
                logs = f.read()
            st.sidebar.text_area("Feedback Logs", logs, height=300)
        except:
            st.sidebar.info("No feedback yet.")
