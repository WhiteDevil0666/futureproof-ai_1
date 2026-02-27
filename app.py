# ==========================================================
# FUTUREPROOF AI – Production Optimized Version
# Full Logic Preserved | Mock Separated | Production Safe
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import time
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

# ================= ENV CONFIG =================
ADMIN_USERNAME = os.getenv("ADMIN_USER")
ADMIN_PASSWORD = os.getenv("ADMIN_PASS")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

client = Groq(api_key=api_key)

MAIN_MODEL = "llama-3.1-8b-instant"
MCQ_MODEL = "llama-3.3-70b-versatile"

# ================= UTILITIES =================

def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")

def safe_llm_call(model, messages, temperature=0.3, retries=2):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            log_api_usage(model, "SUCCESS")
            return response.choices[0].message.content.strip()
        except Exception:
            time.sleep(2)
    log_api_usage(model, "FAILED")
    return None

def safe_json_load(text):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return None

def normalize_skills(skills_input):
    skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
    return list(set(skills))

# ================= CACHED FUNCTIONS =================

@st.cache_data(ttl=3600)
def detect_domain_cached(skills_tuple):
    prompt = f"""
Based strictly on these skills:
{", ".join(skills_tuple)}
Return only the professional domain name.
"""
    return safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return only domain name."},
        {"role": "user", "content": prompt}
    ]) or "General Domain"

@st.cache_data(ttl=3600)
def infer_role_cached(skills_tuple, domain):
    prompt = f"""
Skills: {", ".join(skills_tuple)}
Domain: {domain}
Return only one realistic professional role.
"""
    return safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return only role name."},
        {"role": "user", "content": prompt}
    ]) or "Specialist"

# ================= CORE FUNCTIONS =================

def generate_growth(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Suggest 6 competitive skills. Return comma-separated.
"""
    response = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}])
    if not response:
        return []
    return [s.strip().title() for s in re.split(r",|\n", response) if s.strip()][:6]

def generate_certifications(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Suggest 6 globally recognized certifications. Return comma-separated.
"""
    response = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}])
    if not response:
        return []
    return [c.strip() for c in response.split(",")][:6]

def generate_market(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Explain demand, hiring scale, 3-5 year outlook, job availability.
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or "Market unavailable."

def generate_confidence(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Provide:
Confidence: X%
Risk: Low/Medium/High
Summary:
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or \
           "Confidence: 70%\nRisk: Medium\nSummary: Moderate outlook."

def generate_platforms(role, domain, skills):
    prompt = f"""
Role: {role}
Domain: {domain}
Skills: {", ".join(skills)}

Return ONLY valid JSON:
{{
 "free": [{{"name":"","url":""}}],
 "paid": [{{"name":"","url":""}}]
}}
"""
    response = safe_llm_call(
        MAIN_MODEL,
        [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )
    return safe_json_load(response) or {"free": [], "paid": []}

def generate_mcqs(skills, difficulty):
    prompt = f"""
Create 10 advanced multiple choice questions.

Skills: {", ".join(skills)}
Difficulty: {difficulty}

Rules:
- Avoid basic general knowledge.
- Questions must test applied understanding.
- Include scenario-based and analytical thinking.
- Options must be plausible and competitive.
- Do NOT make easy textbook questions.

Return ONLY valid JSON:
[
 {{
   "question": "...",
   "options": ["A", "B", "C", "D"],
   "answer": "Correct Option Text"
 }}
]
"""
    response = safe_llm_call(
        MCQ_MODEL,
        [
            {"role": "system", "content": "Return only valid JSON array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return safe_json_load(response)

# ================= GOOGLE SHEET =================

def save_feedback(data_row):
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["GOOGLE_SERVICE_ACCOUNT"],
            scopes=scopes
        )
        client_gs = gspread.authorize(creds)
        sheet = client_gs.open("FutureProof_Feedback").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        st.error(f"Google Sheet Error: {str(e)}")

def save_mock_result(data_row):
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["GOOGLE_SERVICE_ACCOUNT"],
            scopes=scopes
        )
        client_gs = gspread.authorize(creds)

        try:
            sheet = client_gs.open("FutureProof_Mock_Results").sheet1
        except:
            sheet = client_gs.create("FutureProof_Mock_Results").sheet1

        sheet.append_row(data_row)
    except Exception as e:
        st.error(f"Mock Sheet Error: {str(e)}")

# ================= SIDEBAR NAVIGATION =================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["🔎 Skill Intelligence", "🎓 Mock Assessment", "🔐 Admin"]
)

# ==========================================================
# ================= SKILL INTELLIGENCE =====================
# ==========================================================

if page == "🔎 Skill Intelligence":

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
    with col2:
        education = st.selectbox(
            "Education Level",
            ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
        )

    skills_input = st.text_input("Current Skills (comma-separated)")
    hours = st.slider("Weekly Learning Hours", 1, 40, 10)

    if st.button("🔎 Analyze Skill Intelligence", use_container_width=True):

        skills = normalize_skills(skills_input)
        skills_tuple = tuple(skills)

        domain = detect_domain_cached(skills_tuple)
        role = infer_role_cached(skills_tuple, domain)

        with ThreadPoolExecutor() as executor:
            growth = executor.submit(generate_growth, role, domain).result()
            certs = executor.submit(generate_certifications, role, domain).result()
            market = executor.submit(generate_market, role, domain).result()
            confidence = executor.submit(generate_confidence, role, domain).result()
            platforms = executor.submit(generate_platforms, role, domain, skills).result()

        weeks = round((len(growth) * 40) / hours) if hours else 0

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Role Alignment",
            "📈 Competitiveness Plan",
            "🎓 Certifications",
            "🌍 Market Outlook"
        ])

        with tab1:
            st.header(role)
            st.markdown(f"Detected Domain: `{domain}`")
            st.code(confidence)

        with tab2:
            for g in growth:
                st.markdown(f"✔ {g}")
            st.markdown(f"Estimated Timeline: ~{weeks} weeks")

        with tab3:
            for c in certs:
                st.markdown(f"- {c}")

        with tab4:
            st.markdown(market)

# ==========================================================
# ================= MOCK ASSESSMENT ========================
# ==========================================================

elif page == "🎓 Mock Assessment":

    st.header("🎓 Skill-Based Mock Assessment")

    candidate_name = st.text_input("Full Name")
    candidate_email = st.text_input("Email")
    candidate_education = st.selectbox(
        "Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )
    skills_input = st.text_input("Skills (comma-separated)")
    difficulty = st.selectbox(
        "Select Difficulty",
        ["Beginner", "Intermediate", "Expert"]
    )

    if "mock_questions" not in st.session_state:
        st.session_state.mock_questions = []

    if st.button("Generate Test"):
        skills = normalize_skills(skills_input)
        questions = generate_mcqs(skills, difficulty)
        if questions:
            st.session_state.mock_questions = questions

    if st.session_state.mock_questions:

        user_answers = []
        score = 0

        for i, q in enumerate(st.session_state.mock_questions):
            st.markdown(f"### Q{i+1}. {q['question']}")
            selected = st.radio("", q["options"], index=None, key=f"mock_{i}")
            user_answers.append(selected)

        if st.button("Submit Test"):

            for i, q in enumerate(st.session_state.mock_questions):
                if user_answers[i] == q["answer"]:
                    score += 1

            percent = (score / 10) * 100

            st.markdown(f"### Score: {score}/10")
            st.markdown(f"### Percentage: {percent:.2f}%")

            save_mock_result([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                candidate_name,
                candidate_email,
                candidate_education,
                skills_input,
                difficulty,
                score,
                percent
            ])

# ==========================================================
# ================= ADMIN ================================
# ==========================================================

elif page == "🔐 Admin":

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.success("Admin logged in")
        else:
            st.error("Invalid credentials")
