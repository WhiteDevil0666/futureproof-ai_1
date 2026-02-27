# ==========================================================
# FUTUREPROOF AI – Production Optimized Version
# All Features Preserved | Optimized | Stable | Scalable
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
        except Exception as e:
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
    response = safe_llm_call(MAIN_MODEL, [
        {"role": "user", "content": prompt}
    ])
    if not response:
        return []
    return [s.strip().title() for s in re.split(r",|\n", response) if s.strip()][:6]

def generate_certifications(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Suggest 6 globally recognized certifications. Return comma-separated.
"""
    response = safe_llm_call(MAIN_MODEL, [
        {"role": "user", "content": prompt}
    ])
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
    response = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt}
    ])
    return safe_json_load(response) or {"free": [], "paid": []}

def generate_mcqs(skills, difficulty):
    prompt = f"""
Create 10 MCQs.
Skills: {", ".join(skills)}
Difficulty: {difficulty}

Return JSON array format.
"""
    response = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return only valid JSON array."},
        {"role": "user", "content": prompt}
    ], temperature=0.2)

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

# ================= ADMIN =================

if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

with st.sidebar:
    st.markdown("## 🔐 Admin Access")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            st.session_state.admin_logged = True
            st.success("Admin logged in")
        else:
            st.error("Invalid credentials")

# ================= USER INPUT =================

st.title("🚀 FutureProof Skill Intelligence Engine")

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

# ================= ANALYSIS =================

if st.button("🔎 Analyze Skill Intelligence", use_container_width=True):

    skills = normalize_skills(skills_input)
    skills_tuple = tuple(skills)

    domain = detect_domain_cached(skills_tuple)
    role = infer_role_cached(skills_tuple, domain)

    with ThreadPoolExecutor() as executor:
        growth_future = executor.submit(generate_growth, role, domain)
        cert_future = executor.submit(generate_certifications, role, domain)
        market_future = executor.submit(generate_market, role, domain)
        conf_future = executor.submit(generate_confidence, role, domain)
        platform_future = executor.submit(generate_platforms, role, domain, skills)

        growth = growth_future.result()
        certs = cert_future.result()
        market = market_future.result()
        confidence = conf_future.result()
        platforms = platform_future.result()

    weeks = round((len(growth) * 40) / hours) if hours else 0

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Role Alignment",
        "📈 Competitiveness Plan",
        "🎓 Certifications",
        "🌍 Market Outlook",
        "📊 Skill Summary"
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

        st.markdown("### 🆓 Free Platforms")
        for item in platforms.get("free", []):
            st.markdown(f"- [{item['name']}]({item['url']})")

        st.markdown("### 💼 Paid Platforms")
        for item in platforms.get("paid", []):
            st.markdown(f"- [{item['name']}]({item['url']})")

    with tab4:
        st.markdown(market)

    with tab5:
        for s in skills:
            st.markdown(f"- {s.title()}")

    # Feedback
    st.divider()
    rating = st.slider("How useful was this analysis?", 1, 5, 4)
    feedback_text = st.text_area("What can we improve?")

    if st.button("Submit Feedback"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("feedback_log.txt", "a") as f:
            f.write(f"{timestamp}|{name}|{rating}|{education}|{skills_input}|{feedback_text}\n")

        save_feedback([
            timestamp, name, rating,
            education, skills_input, feedback_text
        ])

        st.success("✅ Feedback saved successfully!")

# ================= MOCK TEST =================

st.divider()
st.header("🎓 Skill-Based Mock Assessment")

difficulty = st.selectbox(
    "Select Difficulty",
    ["Beginner", "Intermediate", "Expert"]
)

if "mock_questions" not in st.session_state:
    st.session_state.mock_questions = []

if st.button("Generate 10 MCQs"):
    skills = normalize_skills(skills_input)
    questions = generate_mcqs(skills, difficulty)

    if questions:
        st.session_state.mock_questions = questions
    else:
        st.error("Failed to generate MCQs.")

if st.session_state.mock_questions:
    score = 0
    user_answers = []

    for i, q in enumerate(st.session_state.mock_questions):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        ans = st.radio("", q["options"], key=f"q{i}")
        user_answers.append(ans)

    if st.button("Submit Mock Test"):
        for i, q in enumerate(st.session_state.mock_questions):
            if user_answers[i] == q["answer"]:
                score += 1

        percent = (score / 10) * 100

        st.markdown(f"Score: {score}/10")
        st.markdown(f"Percentage: {percent}%")

        if percent >= 80:
            st.success("✅ Qualified (80%+)")
        else:
            st.error("❌ Not Qualified (Below 80%)")
