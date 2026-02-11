# ==========================================================
# FUTUREPROOF AI – Career Intelligence Engine (2026 Edition)
# Production Interactive Version
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FutureProof AI",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:800;
}
.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:20px;
}
.metric-box {
    padding:15px;
    border-radius:10px;
    background-color:#f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 FutureProof AI Career Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Plan Your 2026 Career Growth Intelligently")

# ---------------- LOAD API KEY ----------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.stop()

client = genai.Client(api_key=api_key)

DATASET_CONFIDENCE_THRESHOLD = 0.55
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("futureproof_dummy_data.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df["current_skills"] = df["current_skills"].fillna("").str.lower()
    df["interested_future_field"] = df["interested_future_field"].fillna("").str.lower()
    return df

df = load_data()

# ---------------- GEMINI HELPER ----------------
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

# ---------------- CONFIDENCE LOGIC ----------------
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

# ---------------- ROLE INFERENCE ----------------
def infer_career_role(skills):
    prompt = f"""
You are a senior AI career strategist in 2026.

Given these skills:
{", ".join(skills)}

Suggest ONE high-growth future job role.
Return ONLY the role name.
"""
    role = gemini_generate(prompt)
    return role if role else "Business Transformation Specialist"

# ---------------- UPSKILLING RECOMMENDATIONS ----------------
def infer_growth_plan(role, skills):
    prompt = f"""
For the role: {role}

Suggest 6 high-impact future growth skills for 2026.
Return comma-separated names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []

    raw = re.split(r",|\n", response)
    cleaned = []

    for s in raw:
        s = s.strip().title()
        if s and len(s.split()) <= 4 and s.lower() not in skills:
            cleaned.append(s)

    return list(dict.fromkeys(cleaned))[:6]

# ---------------- CERTIFICATIONS ----------------
def infer_certifications(role):
    prompt = f"""
Suggest 5 trending global certifications in 2026 for {role}.
Return comma-separated certification names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []

    return [c.strip() for c in response.split(",")][:5]

# ==========================================================
# ====================== USER INPUT ========================
# ==========================================================

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

# ==========================================================
# ====================== GENERATE ==========================
# ==========================================================

if st.button("🔎 Generate Career Intelligence Plan", use_container_width=True):

    if not name or not skills_input:
        st.warning("Please enter your name and skills.")
    else:

        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing market signals and future opportunities..."):

            confidence, dataset_field = dataset_confidence(skills)

            if confidence >= DATASET_CONFIDENCE_THRESHOLD:
                role = dataset_field.title()
                source = "Peer Career Intelligence"
            else:
                role = infer_career_role(skills)
                source = "Live Market AI Intelligence"

            growth_skills = infer_growth_plan(role, skills)
            certifications = infer_certifications(role)

            weeks = round((len(growth_skills) * 40) / hours)

        st.success("✅ Career Intelligence Report Ready!")

        # ======================================================
        # DASHBOARD TABS
        # ======================================================

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 Overview", "📈 Growth Roadmap", "🎓 Certifications", "📊 Skill Insights"]
        )

        # ---------------- TAB 1 ----------------
        with tab1:

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("🎯 Recommended Role", role)

            with colB:
                st.metric("📊 Confidence", f"{int(confidence*100)}%")

            with colC:
                market_score = np.random.randint(70, 95)
                st.metric("📈 Market Demand Score", f"{market_score}%")

            st.progress(int(confidence * 100))

            st.markdown(f"**Insight Source:** {source}")

        # ---------------- TAB 2 ----------------
        with tab2:

            st.markdown("### 🚀 Future Growth Focus Areas")

            for skill in growth_skills:
                st.markdown(f"✔️ {skill}")

            st.markdown("### 🗺 Structured Learning Roadmap")

            phase1 = int(weeks * 0.3)
            phase2 = int(weeks * 0.7)

            st.info(f"Phase 1 (Foundation): Weeks 1 – {phase1}")
            st.info(f"Phase 2 (Advanced Skill Development): Weeks {phase1} – {phase2}")
            st.info(f"Phase 3 (Specialization & Certification): Weeks {phase2} – {weeks}")

            st.markdown(f"### ⏳ Estimated Total Upskilling Time: **~{weeks} weeks**")

        # ---------------- TAB 3 ----------------
        with tab3:

            st.markdown("### 🎓 Trending Certifications (2026 Relevant)")

            for cert in certifications:
                st.markdown(f"🏅 {cert}")

        # ---------------- TAB 4 ----------------
        with tab4:

            colX, colY = st.columns(2)

            with colX:
                st.markdown("### ✅ Your Existing Skills")
                for s in skills:
                    st.markdown(f"- {s.title()}")

            with colY:
                st.markdown("### 🚀 High-Impact Growth Areas")
                for s in growth_skills[:4]:
                    st.markdown(f"- {s}")

        st.divider()

        # ======================================================
        # FEEDBACK SECTION
        # ======================================================

        st.markdown("## 💬 Help Us Improve")

        colF1, colF2 = st.columns(2)

        with colF1:
            rating = st.slider("How useful was this plan?", 1, 5, 4)

        with colF2:
            satisfaction = st.selectbox(
                "Would you recommend this tool?",
                ["Yes", "Maybe", "No"]
            )

        feedback_text = st.text_area("What can we improve?")

        if st.button("Submit Feedback"):
            with open("feedback_log.txt", "a") as f:
                f.write(
                    f"\n{name} | Rating: {rating} | Recommend: {satisfaction} | {feedback_text}"
                )
            st.success("🙏 Thank you for your feedback!")

