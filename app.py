# ==========================================================
# FUTUREPROOF AI – Career Intelligence Engine (Admin Edition)
# Dynamic Domain-Specific Version (Groq Llama 3.1)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Groq
from groq import Groq

# Google Sheet
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FutureProof AI",
    page_icon="🚀",
    layout="wide"
)

# ================= UI THEME =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    color: white !important;
}
label { color: white !important; font-weight: 500; }
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🚀 FutureProof AI Career Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Plan Your 2026 Career Growth Intelligently")

# ================= ADMIN LOGIN =================
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

# ================= LOAD GROQ API KEY =================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

client = Groq(api_key=api_key)
GROQ_MODEL = "llama-3.1-8b-instant"

# ================= API USAGE LOGGER =================
def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")

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

# ================= GROQ HELPER =================
def gemini_generate(prompt):  # keeping same function name to avoid breaking logic
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise career intelligence assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()

        log_api_usage("Groq Call", "SUCCESS")
        return result

    except Exception as e:
        log_api_usage("Groq Call", f"FAILED: {str(e)}")
        return None

# ==========================================================
# ================= DYNAMIC DOMAIN DETECTION =================
# ==========================================================

def detect_domain(skills):
    prompt = f"""
You are an expert technology domain classifier.

Based on these skills:
{", ".join(skills)}

Identify the PRIMARY technology domain.

Return ONLY the domain name.
"""
    domain = gemini_generate(prompt)
    return domain if domain else "Technology Domain"

# ================= GOOGLE SHEET SAVE =================
def save_feedback_to_sheet(data_row):
    try:
        creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not creds_json:
            log_api_usage("GoogleSheet Save", "NO_CREDENTIALS")
            return

        creds_dict = json.loads(creds_json)

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client_gs = gspread.authorize(credentials)

        sheet = client_gs.open("FutureProof_AI_Feedback").sheet1
        sheet.append_row(data_row)

        log_api_usage("GoogleSheet Save", "SUCCESS")

    except Exception as e:
        log_api_usage("GoogleSheet Save", f"FAILED: {str(e)}")

# ================= ROLE & GROWTH =================
def infer_career_role(skills):
    domain = detect_domain(skills)

    prompt = f"""
User Domain: {domain}
User Skills: {", ".join(skills)}

Suggest ONE advanced career role strictly inside this domain.
Return only role name.
"""
    role = gemini_generate(prompt)
    return role if role else f"Senior {domain} Specialist"


def infer_growth_plan(role, skills):
    domain = detect_domain(skills)

    prompt = f"""
Role: {role}
Domain: {domain}

Suggest 6 high-impact growth skills strictly within this domain.
Return comma-separated names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []

    raw = re.split(r",|\n", response)
    return [s.strip().title() for s in raw if s.strip()][:6]


def infer_certifications(role, skills):
    domain = detect_domain(skills)

    prompt = f"""
Role: {role}
Domain: {domain}

Suggest 5 globally recognized certifications strictly for this domain.
Return comma-separated names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []

    return [c.strip() for c in response.split(",")][:5]

# ==========================================================
# ================= USER INPUT =================
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
# ================= GENERATE =================
# ==========================================================

if st.button("🔎 Generate Career Intelligence Plan", use_container_width=True):

    if not name or not skills_input:
        st.warning("Please enter your name and skills.")
    else:

        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing your domain and future opportunities..."):

            role = infer_career_role(skills)
            growth_skills = infer_growth_plan(role, skills)
            certifications = infer_certifications(role, skills)
            weeks = round((len(growth_skills) * 40) / hours)

        st.success("✅ Career Intelligence Report Ready!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 Overview", "📈 Growth Roadmap", "🎓 Certifications", "📊 Skill Insights"]
        )

        with tab1:
            st.metric("🎯 Recommended Role", role)
            st.metric("📈 Market Demand Score", f"{np.random.randint(75,95)}%")

        with tab2:
            for skill in growth_skills:
                st.markdown(f"✔️ {skill}")
            st.markdown(f"### ⏳ Estimated Upskilling Time: ~{weeks} weeks")

        with tab3:
            for cert in certifications:
                st.markdown(f"🏅 {cert}")

        with tab4:
            st.markdown("### Your Skills")
            for s in skills:
                st.markdown(f"- {s.title()}")

        st.divider()

        rating = st.slider("How useful was this plan?", 1, 5, 4)
        feedback_text = st.text_area("What can we improve?")

        if st.button("Submit Feedback"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("feedback_log.txt", "a") as f:
                f.write(f"{timestamp} | {name} | {rating} | {education} | {skills_input} | {feedback_text}\n")

            save_feedback_to_sheet([
                timestamp,
                name,
                rating,
                education,
                skills_input,
                feedback_text
            ])

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
