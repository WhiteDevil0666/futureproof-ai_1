# ==========================================================
# FUTUREPROOF AI – Skill Intelligence & Market Insight Engine
# Production Version (With Certification Tab Restored)
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

from groq import Groq
import gspread
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FutureProof Skill Intelligence",
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

st.markdown('<div class="main-title">🚀 FutureProof Skill Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Analyze Your Skills. Understand Your Domain. Evaluate Market Reality.")

# ================= ADMIN =================
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

# ================= GROQ =================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

client = Groq(api_key=api_key)
GROQ_MODEL = "llama-3.1-8b-instant"

def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")

def gemini_generate(prompt):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a neutral professional career intelligence analyst."},
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

# ================= DOMAIN =================
def detect_domain(skills):
    prompt = f"""
Based strictly on these skills:
{", ".join(skills)}

Identify the TRUE professional domain.
Return only domain name.
"""
    return gemini_generate(prompt) or "General Domain"

# ================= ROLE =================
def infer_career_role(skills):
    domain = detect_domain(skills)
    prompt = f"""
Skills: {", ".join(skills)}
Domain: {domain}

Suggest one realistic professional role strictly aligned.
Return only role name.
"""
    return gemini_generate(prompt) or "Specialist"

# ================= GROWTH =================
def infer_growth_plan(role, skills):
    domain = detect_domain(skills)
    prompt = f"""
Role: {role}
Domain: {domain}

Suggest 6 skills that increase competitiveness.
Return comma-separated only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []
    raw = re.split(r",|\n", response)
    return [s.strip().title() for s in raw if s.strip()][:6]

# ================= CERTIFICATIONS =================
def infer_certifications(role, skills):
    domain = detect_domain(skills)
    prompt = f"""
Role: {role}
Domain: {domain}

Suggest 6 globally recognized certifications.
Return comma-separated names only.
"""
    response = gemini_generate(prompt)
    if not response:
        return []
    return [c.strip() for c in response.split(",")][:6]

# ================= MARKET =================
def generate_market_summary(role, skills):
    domain = detect_domain(skills)
    prompt = f"""
Role: {role}
Domain: {domain}

Explain:
- Demand level
- Hiring scale
- 3-5 year outlook
- Job availability
Keep it market-focused.
"""
    return gemini_generate(prompt) or "Market data unavailable."

# ================= CONFIDENCE =================
def generate_confidence_and_risk(role, skills):
    domain = detect_domain(skills)
    prompt = f"""
Role: {role}
Domain: {domain}

Provide:
Confidence: X%
Risk: Low/Medium/High
Summary: Short explanation
Based on career demand.
"""
    return gemini_generate(prompt) or "Confidence: 70%\nRisk: Medium\nSummary: Moderate outlook."

# ================= GOOGLE SHEET =================
def save_feedback_to_sheet(data_row):
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds_dict = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client_gs = gspread.authorize(credentials)
        sheet = client_gs.open("FutureProof_Feedback").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        st.error(f"Google Sheet Error: {str(e)}")

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
if st.button("🔎 Analyze Skill Intelligence", use_container_width=True):

    skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

    role = infer_career_role(skills)
    growth_skills = infer_growth_plan(role, skills)
    certifications = infer_certifications(role, skills)
    market_summary = generate_market_summary(role, skills)
    confidence_risk = generate_confidence_and_risk(role, skills)

    weeks = round((len(growth_skills) * 40) / hours)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎯 Role Alignment",
         "📈 Competitiveness Plan",
         "🎓 Certifications",
         "🌍 Market Outlook",
         "📊 Skill Summary"]
    )

    with tab1:
        st.markdown(f"<h1 style='color:#60a5fa'>{role}</h1>", unsafe_allow_html=True)
        st.markdown(f"🧭 Detected Domain: `{detect_domain(skills)}`")
        st.markdown(f"```\n{confidence_risk}\n```")

    with tab2:
        for skill in growth_skills:
            st.markdown(f"✔️ {skill}")
        st.markdown(f"⏳ Estimated Timeline: ~{weeks} weeks")

    with tab3:
        st.markdown("### 🎓 Recommended Certifications")
        for cert in certifications:
            st.markdown(f"- {cert}")

    with tab4:
        st.markdown(market_summary)

    with tab5:
        for s in skills:
            st.markdown(f"- {s.title()}")

    st.divider()

    rating = st.slider("How useful was this analysis?", 1, 5, 4)
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

        st.success("✅ Feedback saved successfully!")

# ==========================================================
# ================= MOCK TEST MODULE (ADDED ONLY) ==========
# ==========================================================

st.divider()
st.markdown("## 🎓 Skill-Based Mock Assessment")

if "mock_questions" not in st.session_state:
    st.session_state.mock_questions = []

if "mock_generated" not in st.session_state:
    st.session_state.mock_generated = False

difficulty = st.selectbox(
    "Select Difficulty Level",
    ["Beginner", "Intermediate", "Expert"]
)

if st.button("Generate 10 MCQs"):

    skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

    prompt = f"""
Create 10 MCQs based strictly on these skills:
{", ".join(skills)}

Difficulty: {difficulty}

Return strict JSON:

[
 {{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "answer": "Correct Option Text"
 }}
]
No explanation.
"""

    response = gemini_generate(prompt)

    try:
        questions = json.loads(response)
        st.session_state.mock_questions = questions
        st.session_state.mock_generated = True
    except:
        st.error("Failed to generate structured questions. Try again.")

if st.session_state.mock_generated:

    user_answers = []

    for idx, q in enumerate(st.session_state.mock_questions):
        st.markdown(f"**Q{idx+1}. {q['question']}**")
        selected = st.radio(
            "",
            q["options"],
            key=f"mock_{idx}"
        )
        user_answers.append(selected)

    if st.button("Submit Mock Test"):

        score = 0
        for i, q in enumerate(st.session_state.mock_questions):
            if user_answers[i] == q["answer"]:
                score += 1

        percentage = (score / 10) * 100

        st.markdown("### 📊 Result")
        st.markdown(f"Score: {score}/10")
        st.markdown(f"Percentage: {percentage}%")

        if percentage >= 80:
            st.success("✅ Qualified (80%+)")
        else:
            st.error("❌ Not Qualified (Below 80%)")
