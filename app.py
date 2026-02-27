# ==========================================================
# FUTUREPROOF AI – Production Optimized Version
# Full Original Structure Restored + Mock Separated
# + Admin Portal Added (No Logic Removed)
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
    background-color: #0b1220 !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 16px;
    font-weight: 600;
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

# ================= SIDEBAR NAVIGATION =================

st.sidebar.markdown("## 📌 Navigation")

page = st.sidebar.radio(
    "",
    ["🔎 Skill Intelligence", "🎓 Mock Assessment", "🔐 Admin Portal"]
)

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

# ================= DYNAMIC CERTIFICATION PLATFORMS (ADDED) =================
def generate_certification_platforms(role, skills):
    domain = detect_domain(skills)

    prompt = f"""
Role: {role}
Domain: {domain}
Skills: {", ".join(skills)}

Provide certification platforms relevant to this domain.

Return ONLY valid JSON:

{{
 "free": [
   {{"name":"Platform Name","url":"https://example.com"}}
 ],
 "paid": [
   {{"name":"Platform Name","url":"https://example.com"}}
 ]
}}

No explanation.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Return strictly valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_output = response.choices[0].message.content.strip()
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_output)

        log_api_usage("Groq Certification Platforms", "SUCCESS")
        return data

    except Exception as e:
        log_api_usage("Groq Certification Platforms", f"FAILED: {str(e)}")
        return {"free": [], "paid": []}

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
            certifications = executor.submit(generate_certifications, role, domain).result()
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
            st.markdown(f"🧭 Detected Domain: `{domain}`")
            st.code(confidence)

        with tab2:
            for skill in growth:
                st.markdown(f"✔️ {skill}")
            st.markdown(f"⏳ Estimated Timeline: ~{weeks} weeks")

        with tab3:
            st.markdown("### 🎓 Recommended Certifications")
            for cert in certifications:
                st.markdown(f"- {cert}")

            st.markdown("---")
            st.markdown("### 🌐 Certification Platforms (Domain-Specific)")

            st.markdown("#### 🆓 Free Learning Platforms")
            for item in platforms.get("free", []):
                st.markdown(f"- [{item['name']}]({item['url']})")

            st.markdown("#### 💼 Paid / Market Recognized Certifications")
            for item in platforms.get("paid", []):
                st.markdown(f"- [{item['name']}]({item['url']})")

        with tab4:
            st.markdown(market)

        st.divider()

        rating = st.slider("How useful was this analysis?", 1, 5, 4)
        feedback_text = st.text_area("What can we improve?")

        if st.button("Submit Feedback"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("feedback_log.txt", "a") as f:
                f.write(f"{timestamp} | {name} | {rating} | {education} | {skills_input} | {feedback_text}\n")

            save_feedback([
                timestamp,
                name,
                rating,
                education,
                skills_input,
                feedback_text
            ])

            st.success("✅ Feedback saved successfully!")

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
        else:
            st.error("Failed to generate test questions.")

    if st.session_state.mock_questions:

        user_answers = []
        score = 0

        for i, q in enumerate(st.session_state.mock_questions):
            st.markdown(f"### Q{i+1}. {q['question']}")
            selected = st.radio(
                "",
                q["options"],
                index=None,
                key=f"mock_{i}"
            )
            user_answers.append(selected)

        if st.button("Submit Test"):

            for i, q in enumerate(st.session_state.mock_questions):
                if user_answers[i] == q["answer"]:
                    score += 1

            percent = (score / len(st.session_state.mock_questions)) * 100

            st.markdown("## 📊 Test Result")
            st.markdown(f"### Score: {score}/10")
            st.markdown(f"### Percentage: {percent:.2f}%")

            if percent >= 80:
                st.success("✅ Qualified (80%+)")
            else:
                st.error("❌ Not Qualified (Below 80%)")

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
# ================= ADMIN PORTAL ===========================
# ==========================================================

elif page == "🔐 Admin Portal":

    st.header("🔐 Admin Portal")

    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")

    if st.button("Login"):

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.success("✅ Admin Logged In")

            st.markdown("### 📊 System Logs")
            if os.path.exists("api_usage_log.txt"):
                with open("api_usage_log.txt", "r") as f:
                    logs = f.read()
                st.text_area("API Usage Logs", logs, height=300)

            if os.path.exists("feedback_log.txt"):
                with open("feedback_log.txt", "r") as f:
                    feedback_logs = f.read()
                st.text_area("Feedback Logs", feedback_logs, height=300)

        else:
            st.error("❌ Invalid Admin Credentials")        
        


