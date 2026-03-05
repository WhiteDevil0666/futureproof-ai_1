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
import PyPDF2
from PIL import Image
import io

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FutureProof Skill Intelligence",
    page_icon="🚀",
    layout="wide"
)

# ================= UI STYLE =================
st.markdown("""
<style>

/* ================= MAIN APP BACKGROUND ================= */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #ffffff;
}

/* Fix main content area background */
section.main > div {
    background-color: transparent !important;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background-color: #0b1220 !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ================= RADIO OPTIONS ================= */
/* Force radio label text white */
div[data-testid="stRadio"] label p {
    color: #ffffff !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* Force radio container text white */
div[data-testid="stRadio"] label {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Fix disabled/low-opacity issue */
div[data-testid="stRadio"] div {
    opacity: 1 !important;
}

/* Make radio circle border visible */
div[data-testid="stRadio"] span {
    border-color: #ffffff !important;
}

/* ================= INPUT FIELDS ================= */
/* Target Streamlit form labels specifically */
div[data-testid="stForm"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSlider"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Fix small caption-style labels */
label[data-testid="stWidgetLabel"] {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Remove faded effect */
.css-1cpxqw2, .css-1offfwp {
    opacity: 1 !important;
}

/* ================= BUTTON ================= */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white !important;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    border: none;
}

/* ================= HEADINGS ================= */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* ================= TAB VISIBILITY FIX ================= */

button[data-baseweb="tab"] {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

button[data-baseweb="tab"]:hover {
    color: #60a5fa !important;
}

button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #3b82f6 !important;
}

/* Remove faded inactive look */
button[data-baseweb="tab"] span {
    opacity: 1 !important;
}

/* ================= METRIC VISIBILITY FINAL FIX ================= */

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08) !important;
    padding: 22px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Target ALL possible metric label layers */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] span,
div[data-testid="stMetric"] div:first-child {
    color: #ffffff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    font-size: 16px !important;
}

/* Force metric value bright */
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 32px !important;
    opacity: 1 !important;
}
/* Extra safety for Streamlit tab text */
div[data-testid="stTabs"] button {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #3b82f6 !important;
    color: #ffffff !important;
}
/* ================= CHAT VISIBILITY FIX ================= */

/* Chat message container */
div[data-testid="stChatMessage"] {
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Chat text color */
div[data-testid="stChatMessage"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* User message bubble */
div[data-testid="stChatMessage"][data-testid*="user"] {
    background: linear-gradient(90deg, #6366f1, #3b82f6) !important;
}

/* Assistant message bubble */
div[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(255,255,255,0.08) !important;
}
/* ================= CODE BLOCK VISIBILITY FIX ================= */

/* Inline code (like SELECT) */
code {
    background-color: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* Code block container */
pre {
    background-color: #1e293b !important;
    color: #ffffff !important;
    padding: 16px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    overflow-x: auto !important;
}

/* Code inside pre */
pre code {
    background: none !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Streamlit specific markdown code block */
div[data-testid="stMarkdownContainer"] pre {
    background-color: #1e293b !important;
    color: #ffffff !important;
}

/* Force syntax highlight text white */
div[data-testid="stMarkdownContainer"] code {
    color: #ffffff !important;
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
# ================= MODEL PRICING CONFIG =================

MODEL_PRICING = {
    "llama-3.1-8b-instant": 0.0002,
    "llama-3.3-70b-versatile": 0.0006
}

# ================= REQUEST CONTROL =================

MAX_REQUESTS_PER_SESSION = 60
REQUEST_COOLDOWN = 3

# ================= SIDEBAR NAVIGATION =================

st.sidebar.markdown("## 📌 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🔎 Skill Intelligence",
        "🎓 Mock Assessment",
        "📚 Guided Study Chat",
        "💼 AI Job Finder (Premium)",
        "🔐 Admin Portal"
    ]
)

# ================= AI REQUEST USAGE =================

remaining = MAX_REQUESTS_PER_SESSION - st.session_state.get("request_count", 0)
st.sidebar.caption(f"🤖 AI Requests Remaining: {remaining}")

# Reset mock session when switching pages
if page != "🎓 Mock Assessment":
    if "mock_questions" in st.session_state:
        del st.session_state.mock_questions
    if "start_time" in st.session_state:
        del st.session_state.start_time
    if "time_limit" in st.session_state:
        del st.session_state.time_limit
    if "exam_submitted" in st.session_state:
        del st.session_state.exam_submitted

if page != "📚 Guided Study Chat":
    if "study_chat_started" in st.session_state:
        del st.session_state.study_chat_started
    if "study_messages" in st.session_state:
        del st.session_state.study_messages
    if "study_context" in st.session_state:
        del st.session_state.study_context

if page != "💼 AI Job Finder (Premium)":
    if "job_analysis_result" in st.session_state:
        del st.session_state.job_analysis_result


# =====================================================
# ✅ ADD THIS BLOCK RIGHT HERE (FIRST CHANGE)
# =====================================================

if "current_user" not in st.session_state:
    st.session_state.current_user = "System"

if "current_feature" not in st.session_state:
    st.session_state.current_feature = "General"


# ================= REQUEST CONTROL =================

# MAX_REQUESTS_PER_SESSION = 25
# REQUEST_COOLDOWN = 3  # seconds


def check_request_limit():

    now = time.time()

    if "request_count" not in st.session_state:
        st.session_state.request_count = 0

    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    # Cooldown protection
    if now - st.session_state.last_request_time < REQUEST_COOLDOWN:
        st.warning("⏳ Please wait a few seconds before sending another request.")
        st.stop()

    # Session limit protection
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error("⚠️ Session request limit reached. Please refresh the page.")
        st.stop()

    st.session_state.last_request_time = now
    st.session_state.request_count += 1

# ================= UTILITIES =================

def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")


def safe_llm_call(model, messages, temperature=0.3, retries=2):

    # 🔥 Read current tracking info safely
    user = st.session_state.get("current_user", "System")
    feature = st.session_state.get("current_feature", "General")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )

            content = response.choices[0].message.content.strip()

            # ================= TOKEN USAGE =================
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

            # ================= COST CALCULATION =================
            price_per_1k = MODEL_PRICING.get(model, 0.0005)
            estimated_cost = (total_tokens / 1000) * price_per_1k

            # ================= SAVE TO GOOGLE SHEET =================
            try:
                save_api_usage([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user,
                    feature,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    round(estimated_cost, 6)
                ])
            except Exception as sheet_error:
                print("Sheet Logging Error:", sheet_error)

            log_api_usage(model, "SUCCESS")

            return content

        except Exception as e:
            print(f"LLM Attempt {attempt+1} Failed:", e)
            time.sleep(2)

    log_api_usage(model, "FAILED")
    return None


# ================= SAFE JSON PARSER =================

def safe_json_load(text):
    try:
        if not text:
            return None

        cleaned = text.replace("```json", "").replace("```", "").strip()

        # Extract JSON block safely
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1

        if start != -1 and end != -1:
            cleaned = cleaned[start:end]

        return json.loads(cleaned)

    except Exception as e:
        print("JSON Parse Error:", e)
        return None

# ================= SKILL NORMALIZER =================

def normalize_skills(skills_input):
    if not skills_input:
        return []

    skills = [
        s.strip().lower()
        for s in skills_input.split(",")
        if s.strip()
    ]

    return list(set(skills))
# ================= CACHED FUNCTIONS =================

@st.cache_data(ttl=3600)
def detect_domain_cached(skills_tuple):
    prompt = f"""
You are a career classification engine.

Given these professional skills:
{", ".join(skills_tuple)}

Identify the professional career field (e.g., Data Analytics, Software Engineering, Cybersecurity, Finance, Marketing).

Return ONLY the domain name.
"""
    return safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Classify career domain only."},
        {"role": "user", "content": prompt}
    ]) or "General Domain"

@st.cache_data(ttl=3600)
def infer_role_cached(skills_tuple, domain):
    prompt = f"""
Skills: {", ".join(skills_tuple)}
Domain: {domain}

Suggest one realistic professional role aligned with this domain.
Return only role name.
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
Suggest 6 skills that increase competitiveness.
Return comma-separated only.
"""
    response = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}])
    if not response:
        return []
    raw = re.split(r",|\n", response)
    return [s.strip().title() for s in raw if s.strip()][:6]

def generate_certifications(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}

Suggest 6 globally recognized certifications relevant to this role.
Return comma-separated names only.
No explanations.
No numbering.
"""

    response = safe_llm_call(
        MAIN_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.3
    )

    if not response:
        return []

    raw = re.split(r",|\n", response)
    return [c.strip() for c in raw if c.strip()][:6]

def generate_platforms(role, domain, skills):

    prompt = f"""
Role: {role}
Domain: {domain}
Skills: {", ".join(skills)}

Provide certification platforms relevant to this domain.

STRICT RULES:
- Return ONLY pure JSON
- No explanation
- No markdown
- No extra text
- No triple backticks

Format exactly like this:

{{
  "free": [
    {{"name": "Platform Name", "url": "https://example.com"}}
  ],
  "paid": [
    {{"name": "Platform Name", "url": "https://example.com"}}
  ]
}}
"""

    response = safe_llm_call(
        MAIN_MODEL,
        [
            {"role": "system", "content": "Return ONLY raw JSON. No text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    if not response:
        return {"free": [], "paid": []}

    # --------- IMPROVED JSON CLEANING ---------
    try:
        cleaned = response.strip()

        # Remove markdown if model adds it
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Extract JSON block if extra text exists
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end != -1:
            cleaned = cleaned[start:end]

        return json.loads(cleaned)

    except Exception as e:
        print("Platform JSON parse error:", response)
        return {"free": [], "paid": []}
def generate_market(role, domain):
    prompt = f"""
Role: {role}
Domain: {domain}
Explain demand level, hiring scale, 3-5 year outlook, job availability.
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or "Market data unavailable."

def generate_confidence(role, domain):

    prompt = f"""
Role: {role}
Domain: {domain}

You are a career evaluation engine.

Provide ONLY:

Confidence: X% (numeric realistic hiring confidence)
Risk: Low/Medium/High
Summary: 2-3 lines about job market demand and career stability.

DO NOT:
- Propose projects
- Suggest implementation plans
- Provide technical design
- Provide budget
- Provide roadmap
- Provide company strategy

Keep response short and strictly career-focused.
"""

    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or \
           "Confidence: 70%\nRisk: Medium\nSummary: Moderate outlook."

def generate_mcqs(skills, difficulty, test_mode):

    if test_mode == "Theoretical Knowledge":
        mode_instruction = """
        Generate theory-based conceptual multiple choice questions.
        Focus on definitions, comparisons, and best practices.
        Do NOT include code snippets.
        """

    elif test_mode == "Logical Thinking":
        mode_instruction = """
        Generate logical reasoning and scenario-based questions
        related to the provided skills.
        Focus on analytical and problem-solving ability.
        """

    elif test_mode == "Coding Based":
        mode_instruction = """
        Generate practical coding-based questions.
        Include code snippets, debugging, output prediction,
        and algorithmic thinking.
        """

    else:
        mode_instruction = "Generate skill-based multiple choice questions."

    prompt = f"""
    Create 10 multiple choice questions.

    Skills: {", ".join(skills)}
    Difficulty: {difficulty}

    {mode_instruction}

    STRICT FORMAT:
    Return ONLY valid JSON array like this:

    [
        {{
            "question": "Question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0
        }}
    ]

    IMPORTANT:
    - Exactly 4 options per question
    - answer must be index (0,1,2,3)
    - No explanations
    - No markdown
    """

    response = safe_llm_call(
        MCQ_MODEL,
        [
            {"role": "system", "content": "Return ONLY valid JSON array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    data = safe_json_load(response)

    if isinstance(data, list):
        return data

    return None

def generate_explanation(question, correct_answer):

    prompt = f"""
Question:
{question}

Correct Answer:
{correct_answer}

Explain briefly (2-4 lines) why this answer is correct.
Keep it educational and clear.
Do NOT repeat the question.
Do NOT add extra formatting.
"""

    return safe_llm_call(
        MAIN_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.3
    ) or "Explanation unavailable."

# ================= CAREER READINESS SCORE =================

def calculate_career_readiness(skills, growth, confidence_value):

    skill_score = min(len(skills) * 10, 40)
    growth_score = min(len(growth) * 5, 30)
    market_score = min(confidence_value, 30)

    total_score = skill_score + growth_score + market_score

    skill_msg = "Strong skill foundation." if skill_score > 25 else "You need to build more core skills."

    growth_msg = "Only minor improvements needed." if growth_score > 20 else "Focus on recommended growth skills."

    market_msg = "Market demand is strong." if market_score > 20 else "This role has moderate demand."

    return {
        "total": total_score,
        "skill_score": skill_score,
        "growth_score": growth_score,
        "market_score": market_score,
        "skill_msg": skill_msg,
        "growth_msg": growth_msg,
        "market_msg": market_msg
    }


# ================= TIMER CONFIG =================
def get_time_limit(difficulty):
    if difficulty == "Beginner":
        return 2 * 60   # 2 minutes
    elif difficulty == "Intermediate":
        return 4 * 60   # 4 minutes
    elif difficulty == "Expert":
        return 6 * 60   # 6 minutes
    return 2 * 60

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


def save_api_usage(data_row):
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
            sheet = client_gs.open("FutureProof_API_Usage").sheet1
        except:
            sheet = client_gs.create("FutureProof_API_Usage").sheet1

        sheet.append_row(data_row)

    except Exception as e:
        print("API Usage Logging Error:", str(e))

# ================= ADMIN ANALYTICS =================

@st.cache_data(ttl=300)
def load_mock_results():
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

        sheet = client_gs.open("FutureProof_Mock_Results").sheet1

        data = sheet.get_all_records()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        return df

    except Exception as e:
        st.error(f"Analytics Load Error: {str(e)}")
        return pd.DataFrame()

# ==========================================================
# ================= AI MENTOR ENGINE =======================
# ==========================================================

def analyze_user_trend(name):
    df = load_mock_results()

    if df.empty:
        return None

    df.columns = df.columns.str.strip().str.lower()

    if "candidate_name" not in df.columns or "percent" not in df.columns:
        return None

    user_df = df[df["candidate_name"].str.lower() == name.lower()]

    if user_df.empty:
        return None

    user_df["percent"] = pd.to_numeric(user_df["percent"], errors="coerce")

    scores = user_df["percent"].dropna().tolist()

    if not scores:
        return None

    trend = "stable"

    if len(scores) >= 2:
        if scores[-1] > scores[-2]:
            trend = "improving"
        elif scores[-1] < scores[-2]:
            trend = "declining"

    return {
        "latest": scores[-1],
        "best": max(scores),
        "average": sum(scores)/len(scores),
        "total_tests": len(scores),
        "trend": trend
    }


def recommend_difficulty(performance_data):

    if not performance_data:
        return "Beginner"

    latest = performance_data["latest"]

    if latest >= 85:
        return "Expert"
    elif latest >= 60:
        return "Intermediate"
    else:
        return "Beginner"


def generate_mentor_response(name, performance_data):

    prompt = f"""
You are an intelligent AI Career Mentor.

User Name: {name}
Performance Data: {performance_data}

Your job:
- Greet personally
- Analyze progress trend
- Encourage improvement
- Suggest next step
- Be motivating but professional
- Keep under 8 lines
"""

    return safe_llm_call(
        "llama-3.3-70b-versatile",
        [{"role": "user", "content": prompt}],
        temperature=0.5
    )

# ==========================================================
# ================= STUDY MEMORY ENGINE ====================
# ==========================================================

def save_study_history(data_row):
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
            sheet = client_gs.open("FutureProof_Study_History").sheet1
        except:
            sheet = client_gs.create("FutureProof_Study_History").sheet1

        sheet.append_row(data_row)

    except Exception as e:
        st.error(f"Study History Error: {str(e)}")


def check_study_history(name, education, topic, level):
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

        sheet = client_gs.open("FutureProof_Study_History").sheet1
        data = sheet.get_all_records()

        if not data:
            return False

        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower()

        match = df[
            (df["name"].str.lower() == name.lower()) &
            (df["education"].str.lower() == education.lower()) &
            (df["topic"].str.lower() == topic.lower()) &
            (df["level"].str.lower() == level.lower())
        ]

        return not match.empty

    except:
        return False



# ==========================================================
# ================= SKILL INTELLIGENCE =====================
# ==========================================================

if page == "🔎 Skill Intelligence":

    # 🔥 SET FEATURE IMMEDIATELY (SAFE)
    st.session_state.current_feature = "Skill_Intelligence"

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")

        # 🔥 SET USER IMMEDIATELY AFTER INPUT
        st.session_state.current_user = name.strip() if name else "Guest"

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

        domain = detect_domain_cached(skills_tuple) or "General Domain"
        role = infer_role_cached(skills_tuple, domain) or "Specialist"

        growth = generate_growth(role, domain) or []
        certifications = generate_certifications(role, domain) or []
        market = generate_market(role, domain) or "Market data unavailable."
        confidence = generate_confidence(role, domain)
        platforms = generate_platforms(role, domain, skills) or {"free": [], "paid": []}
        weeks = round((len(growth) * 40) / hours) if hours else 0

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Role Alignment",
            "📈 Competitiveness Plan",
            "🎓 Certifications",
            "🌍 Market Outlook"
        ])

        # ======================================================
        # ================= ROLE ALIGNMENT =====================
        # ======================================================
        with tab1:

            st.header(role)
            st.markdown(f"🧭 Detected Domain: `{domain}`")

            # ================= SAFE CONFIDENCE HANDLING =================
            confidence_value = 70
            risk_value = "Medium"
            summary_value = "Moderate job outlook."

            if isinstance(confidence, dict):
                confidence_value = confidence.get("confidence", 70)
                risk_value = confidence.get("risk", "Medium")
                summary_value = confidence.get("summary", "Moderate job outlook.")

            elif isinstance(confidence, str):

                conf_match = re.search(r"(\d+)%", confidence)
                risk_match = re.search(r"Risk:\s*(Low|Medium|High)", confidence)
                summary_match = re.search(r"Summary:\s*(.*)", confidence)

                if conf_match:
                    confidence_value = int(conf_match.group(1))

                if risk_match:
                    risk_value = risk_match.group(1)

                if summary_match:
                    summary_value = summary_match.group(1)

            readiness = calculate_career_readiness(
                skills,
                growth,
                confidence_value
            )

            colA, colB = st.columns(2)

            with colA:
                st.metric("Hiring Confidence", f"{confidence_value}%")

            with colB:
                st.metric("Market Risk", risk_value)

            st.markdown("### 📌 Career Outlook")
            st.markdown(summary_value)

            st.markdown("### 🚀 Career Readiness Score")

            st.metric(
                "Overall Career Readiness",
                f"{readiness['total']}%"
            )

            colR1, colR2, colR3 = st.columns(3)

            with colR1:
                st.metric("Skill Strength", f"{readiness['skill_score']}/40")

            with colR2:
                st.metric("Growth Potential", f"{readiness['growth_score']}/30")

            with colR3:
                st.metric("Market Alignment", f"{readiness['market_score']}/30")

            st.markdown("### 📌 Score Explanation")

            st.info(f"**Skill Strength:** {readiness['skill_msg']}")
            st.info(f"**Growth Potential:** {readiness['growth_msg']}")
            st.info(f"**Market Alignment:** {readiness['market_msg']}")    
    # ======================================================
    # ================= COMPETITIVENESS PLAN ===============
    # ======================================================
        with tab2:

            if growth:
                for skill in growth:
                    st.markdown(f"✔️ {skill}")
                st.markdown(f"⏳ Estimated Timeline: ~{weeks} weeks")
            else:
                st.info("No skill recommendations available.")

        # ======================================================
        # ================= CERTIFICATIONS =====================
        # ======================================================
        with tab3:

            st.markdown("### 🎓 Recommended Certifications")

            if certifications:
                for cert in certifications:
                    st.markdown(f"- {cert}")
            else:
                st.info("No certifications available.")

            st.markdown("---")
            st.markdown("### 🌐 Certification Platforms (Domain-Specific)")

            st.markdown("#### 🆓 Free Learning Platforms")
            for item in platforms.get("free", []):
                st.markdown(f"- [{item['name']}]({item['url']})")

            st.markdown("#### 💼 Paid / Market Recognized Certifications")
            for item in platforms.get("paid", []):
                st.markdown(f"- [{item['name']}]({item['url']})")

        # ======================================================
        # ================= MARKET OUTLOOK =====================
        # ======================================================
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

    # 🔥 ADD THIS
    st.session_state.current_user = candidate_name if candidate_name else "Guest"
    st.session_state.current_feature = "Mock_Assessment"
    # ================= AI MENTOR GREETING =================
    if candidate_name:

        performance_data = analyze_user_trend(candidate_name)

        if performance_data:

            mentor_message = generate_mentor_response(
                candidate_name,
                performance_data
            )

            st.success(mentor_message)

            recommended_level = recommend_difficulty(performance_data)

            st.info(
                f"🎯 Recommended Difficulty Based On Performance: {recommended_level}"
            )

        else:
            st.info(
                f"👋 Welcome {candidate_name}! Let's start building your career momentum 🚀"
            )

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

    test_mode = st.selectbox(
        "Select Test Mode",
        ["Theoretical Knowledge", "Logical Thinking", "Coding Based"]
    )

    if "mock_questions" not in st.session_state:
        st.session_state.mock_questions = []

    # ================= GENERATE TEST =================
    if st.button("Generate Test"):

        skills = normalize_skills(skills_input)
        questions = generate_mcqs(skills, difficulty, test_mode)

        if questions and isinstance(questions, list):

            st.session_state.mock_questions = questions
            st.session_state.start_time = time.time()
            st.session_state.time_limit = get_time_limit(difficulty)

            st.session_state.exam_submitted = False
            st.session_state.explanations = {}
            st.session_state.result_saved = False

        else:
            st.error("Failed to generate test questions. Try again.")
    # ================= DISPLAY QUESTIONS =================
    if st.session_state.get("mock_questions"):

        auto_submit = False

        # ================= TIMER =================
        if "start_time" in st.session_state:

            elapsed = int(time.time() - st.session_state.start_time)
            remaining = st.session_state.time_limit - elapsed

            if remaining <= 0:
                auto_submit = True
                st.warning("⏰ Time is up! Auto-submitting test...")
            else:
                minutes = remaining // 60
                seconds = remaining % 60
                st.markdown(f"### ⏳ Time Remaining: {minutes:02d}:{seconds:02d}")

        # ================= HANDLE SUBMISSION =================
        if (st.button("Submit Test") or auto_submit) and not st.session_state.get("exam_submitted"):

            st.session_state.exam_submitted = True
            score = 0

            for i, q in enumerate(st.session_state.mock_questions):

                selected = st.session_state.get(f"mock_{i}")
                correct_answer = q.get("answer")
                correct_option = None

                # ---------- SAFE ANSWER HANDLING ----------
                if isinstance(correct_answer, int):
                    if 0 <= correct_answer < len(q["options"]):
                        correct_option = q["options"][correct_answer]

                elif isinstance(correct_answer, str):
                    correct_answer = correct_answer.strip()

                    if correct_answer.isdigit():
                        idx = int(correct_answer)
                        if 0 <= idx < len(q["options"]):
                            correct_option = q["options"][idx]

                    elif correct_answer in q["options"]:
                        correct_option = correct_answer

                    elif correct_answer in ["A", "B", "C", "D"]:
                        index_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                        idx = index_map[correct_answer]
                        if idx < len(q["options"]):
                            correct_option = q["options"][idx]

                # ---------- SAFE COMPARISON ----------
                if selected and correct_option:
                    if selected.strip().lower() == correct_option.strip().lower():
                        score += 1

                # ---------- GENERATE EXPLANATION ----------
                if i not in st.session_state.explanations:
                    explanation = generate_explanation(
                        q["question"],
                        correct_option
                    )
                    st.session_state.explanations[i] = explanation

            # ---------- FINAL RESULT CALCULATION ----------
            percent = (score / len(st.session_state.mock_questions)) * 100

            st.session_state.final_score = score
            st.session_state.final_percent = percent

            st.session_state.pop("start_time", None)
            st.session_state.pop("time_limit", None)

        # ================= QUESTION LOOP =================
        for i, q in enumerate(st.session_state.mock_questions):

            st.markdown(f"### Q{i+1}. {q['question']}")

            selected = st.radio(
                "",
                q["options"],
                index=None,
                key=f"mock_{i}",
                disabled=st.session_state.get("exam_submitted", False)
            )

            if st.session_state.get("exam_submitted"):

                correct_answer = q.get("answer")
                correct_option = None

                if isinstance(correct_answer, int):
                    if correct_answer < len(q["options"]):
                        correct_option = q["options"][correct_answer]

                elif isinstance(correct_answer, str):

                    if correct_answer in q["options"]:
                        correct_option = correct_answer

                    elif correct_answer in ["A", "B", "C", "D"]:
                        index_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                        idx = index_map[correct_answer]
                        if idx < len(q["options"]):
                            correct_option = q["options"][idx]

                if selected == correct_option:
                    st.success(f"✅ Correct Answer: {correct_option}")
                else:
                    st.error(f"❌ Your Answer: {selected}")
                    st.info(f"✔ Correct Answer: {correct_option}")

                st.markdown("📘 **Explanation:**")
                st.info(st.session_state.explanations.get(i, "No explanation available."))

        # ================= SHOW FINAL RESULT =================
        if st.session_state.get("exam_submitted"):

            st.markdown("## 📊 Test Result")
            st.markdown(f"### Score: {st.session_state.get('final_score', 0)}/10")
            st.markdown(f"### Percentage: {st.session_state.get('final_percent', 0):.2f}%")

            if st.session_state.get("final_percent", 0) >= 80:
                st.success("✅ Qualified (80%+)")
            else:
                st.error("❌ Not Qualified (Below 80%)")

            if not st.session_state.get("result_saved"):

                save_mock_result([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    candidate_name,
                    candidate_email,
                    candidate_education,
                    skills_input,
                    difficulty,
                    st.session_state.get("final_score", 0),
                    st.session_state.get("final_percent", 0)
                ])

                st.session_state.result_saved = True



# ==========================================================
# ================= GUIDED STUDY CHAT ======================
# ==========================================================

elif page == "📚 Guided Study Chat":

    st.header("📚 AI Guided Study Chat")

    # ================= USER DETAILS =================
    candidate_name = st.text_input("Full Name")

    # 🔥 Only update tracking when name is entered
    if candidate_name:
        st.session_state.current_user = candidate_name
    else:
        st.session_state.current_user = "Guest"

    st.session_state.current_feature = "Guided_Study_Chat"

    # ================= AI MENTOR GREETING =================
    if candidate_name:

        performance_data = analyze_user_trend(candidate_name)

        if performance_data:

            mentor_message = generate_mentor_response(
                candidate_name,
                performance_data
            )

            st.success(mentor_message)

        else:
            st.info(f"👋 Hi {candidate_name}! Let's start building your expertise 🚀")

    education = st.selectbox(
        "Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )

    topic = st.text_input("Topic You Want To Study")

    level = st.selectbox(
        "Skill Level",
        ["Beginner", "Intermediate", "Expert"]
    )

    learning_goal = st.text_input("Learning Goal (Optional)")

    if "study_chat_started" not in st.session_state:
        st.session_state.study_chat_started = False

    if "study_messages" not in st.session_state:
        st.session_state.study_messages = []

    # ================= START LEARNING BUTTON =================
    if st.button("Start Learning"):

        if topic and candidate_name:

            # 🔥 NEW: Check Study History
            already_studied = check_study_history(
                candidate_name,
                education,
                topic,
                level
            )

            if already_studied:

                st.warning(
                    f"You have already studied **{topic} ({level})**."
                )

                action = st.selectbox(
                    "What would you like to do?",
                    ["Revise", "Study in Detailed Mode", "Test Yourself"]
                )

                if action == "Test Yourself":

                    questions = generate_mcqs(
                        [topic],
                        level,
                        "Theoretical Knowledge"
                    )

                    if questions:
                        st.session_state.quick_test = questions[:5]

                elif action == "Revise":
                    st.session_state.study_chat_started = True

                elif action == "Study in Detailed Mode":
                    level = "Expert"
                    st.session_state.study_chat_started = True

            else:
                # First time studying
                save_study_history([
                    candidate_name,
                    education,
                    topic,
                    level,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

                st.session_state.study_chat_started = True

            # Study context setup (kept from your original code)
            st.session_state.study_context = f"""
You are an expert tutor and AI career mentor.

Teach Topic: {topic}
Difficulty Level: {level}
Student Education: {education}
Student Goal: {learning_goal}

If user has past performance:
- Encourage improvement
- Align explanations with career growth

RULES:
- Stay strictly within the selected topic.
- Adjust explanation depth based on difficulty.
- Provide structured explanations.
- Use examples when helpful.
- If question is outside topic, politely redirect.
"""

            st.session_state.study_messages = []

        else:
            st.warning("Please fill required details.")

    # ================= CHAT AREA =================
    if st.session_state.study_chat_started:

        st.markdown("---")
        st.subheader(f"📘 Learning Topic: {topic}")

        user_input = st.chat_input("Ask your question about this topic...")

        if user_input:

            st.session_state.study_messages.append(
                {"role": "user", "content": user_input}
            )

            messages = [
                {"role": "system", "content": st.session_state.study_context}
            ] + st.session_state.study_messages

            response = safe_llm_call(
                MAIN_MODEL,
                messages,
                temperature=0.4
            )

            st.session_state.study_messages.append(
                {"role": "assistant", "content": response}
            )

        for msg in st.session_state.study_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # ================= QUICK SELF TEST =================
    if "quick_test" in st.session_state:

        st.markdown("### 📝 Quick Self-Test")

        score = 0

        for i, q in enumerate(st.session_state.quick_test):

            selected = st.radio(
                q["question"],
                q["options"],
                key=f"quick_{i}"
            )

            if selected == q["options"][q["answer"]]:
                score += 1

        if st.button("Submit Quick Test"):

            percent = (score / len(st.session_state.quick_test)) * 100

            if percent >= 80:
                st.success(
                    "🔥 Excellent! You are ready for higher level or next topic!"
                )
            else:
                st.info(
                    "You may revise this level once more before upgrading."
                )

            del st.session_state.quick_test



# ==========================================================
# ================= AI JOB FINDER (PREMIUM) ================
# ==========================================================

elif page == "💼 AI Job Finder (Premium)":

    st.header("💼 AI Career Job Finder")
    st.success("🎯 AI-Powered Smart Job Matching Activated")

    name = st.text_input("Full Name")

    # 🔥 Safe user tracking
    if name:
        st.session_state.current_user = name
    else:
        st.session_state.current_user = "Guest"

    st.session_state.current_feature = "Job_Finder"

    age = st.number_input("Age", min_value=16, max_value=65, step=1)

    education = st.selectbox(
        "Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )

    skills_input = st.text_input("Skills (comma-separated)")

    experience = st.selectbox(
        "Years of Experience",
        ["Fresher", "1-2 Years", "3-5 Years", "5+ Years"]
    )

    current_field = st.text_input("Current Field / Industry")

    target_role = st.text_input("Target Job Profile You Are Looking For")

    resume_file = st.file_uploader(
        "Upload Resume (PDF, JPG, PNG supported)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if st.button("🔍 Find Suitable Jobs"):

        if name and skills_input and target_role:

            with st.spinner("Analyzing profile and matching jobs..."):

                skills = normalize_skills(skills_input)

                resume_text = ""

                # ---------------- PDF Extraction ----------------
                if resume_file is not None:

                    if resume_file.type == "application/pdf":
                        try:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(resume_file)
                            pages = [page.extract_text() for page in pdf_reader.pages]
                            resume_text = "\n".join(pages)
                        except:
                            resume_text = "PDF uploaded (text extraction failed)."

                    # ---------------- IMAGE Upload ----------------
                    elif resume_file.type in ["image/png", "image/jpeg"]:
                        from PIL import Image
                        image = Image.open(resume_file)
                        st.image(image, caption="Uploaded Resume", use_column_width=True)
                        resume_text = "Resume uploaded as image."

                prompt = f"""
You are an AI Career Placement Advisor.

Candidate Profile:
Name: {name}
Age: {age}
Education: {education}
Skills: {", ".join(skills)}
Experience: {experience}
Current Field: {current_field}
Target Job Role: {target_role}
Resume Content: {resume_text}

Analyze compatibility between the candidate and the target role.

Provide structured output:

1. Job Fit Score (0-100%)
2. Strengths
3. Skill Gaps
4. Resume Improvements
5. 3 Alternative Roles
6. Suggested Industries
7. Recommended Job Search Keywords
"""

                response = safe_llm_call(
                    "llama-3.3-70b-versatile",
                    [{"role": "user", "content": prompt}],
                    temperature=0.4
                )

            st.markdown("## 🎯 AI Career Recommendations")
            st.markdown(response)

            # ================= SMART JOB LINKS =================
            encoded_role = target_role.replace(" ", "%20")

            st.markdown("### 🔗 Direct Job Search Links")

            st.markdown(
                f"🔹 [LinkedIn Jobs](https://www.linkedin.com/jobs/search/?keywords={encoded_role})"
            )
            st.markdown(
                f"🔹 [Indeed Jobs](https://www.indeed.com/jobs?q={encoded_role})"
            )
            st.markdown(
                f"🔹 [Naukri Jobs](https://www.naukri.com/{target_role.replace(' ', '-')}-jobs)"
            )
            st.markdown(
                f"🔹 [Glassdoor Jobs](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={encoded_role})"
            )

        else:
            st.warning("Please fill required fields (Name, Skills, Target Role).")
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

            df = load_mock_results()

            if df.empty:
                st.warning("No mock test data available yet.")
                st.stop()

            # ================= CLEAN & STANDARDIZE COLUMNS =================
            df.columns = df.columns.str.strip().str.lower()

            df = df.rename(columns={
                "percentage": "percent",
                "marks": "score",
                "level of exam": "difficulty",
                "education level": "education",
                "name": "candidate_name",
                "email": "candidate_email",
                "timestamp": "timestamp",
                "skills": "skills"
            })

            required_columns = ["percent", "difficulty", "score"]

            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Missing column: {col}")
                    st.write("Available columns:", df.columns)
                    st.stop()

            df["percent"] = pd.to_numeric(df["percent"], errors="coerce")
            df["score"] = pd.to_numeric(df["score"], errors="coerce")

            # ================= SUMMARY METRICS =================
            st.markdown("## 📊 Platform Overview")

            total_tests = len(df)
            avg_score = df["percent"].mean()
            pass_rate = (df["percent"] >= 80).mean() * 100

            col1, col2, col3 = st.columns(3)

            def metric_card(title, value):
                st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        padding: 20px;
                        border-radius: 12px;
                        text-align: center;
                        border: 1px solid rgba(255,255,255,0.1);
                    ">
                        <h4 style="color:#94a3b8; margin-bottom:10px;">{title}</h4>
                        <h2 style="color:white; font-weight:700;">{value}</h2>
                    </div>
                """, unsafe_allow_html=True)

            with col1:
                metric_card("Total Tests", total_tests)

            with col2:
                metric_card("Average Score", f"{avg_score:.2f}%")

            with col3:
                metric_card("Pass Rate", f"{pass_rate:.2f}%")

            st.divider()

            # ================= DIFFICULTY ANALYSIS =================
            st.markdown("## 📈 Difficulty Breakdown")
            difficulty_counts = df["difficulty"].value_counts()
            st.bar_chart(difficulty_counts)

            st.divider()

            # ================= SCORE DISTRIBUTION =================
            st.markdown("## 📊 Score Distribution")
            st.bar_chart(df["percent"])

            st.divider()

            # ================= TOP PERFORMERS =================
            st.markdown("## 🏆 Top Performers")
            top_5 = df.sort_values("percent", ascending=False).head(5)

            st.dataframe(top_5[[
                "candidate_name",
                "candidate_email",
                "difficulty",
                "percent"
            ]])

            st.divider()

            # ================= FULL DATASET =================
            st.markdown("## 📂 Full Dataset")
            st.dataframe(df)

            # ======================================================
            # ================= API COST ANALYTICS =================
            # ======================================================

            st.divider()

            @st.cache_data(ttl=300)
            def load_api_usage():
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
                    sheet = client_gs.open("FutureProof_API_Usage").sheet1
                    data = sheet.get_all_records()

                    if not data:
                        return pd.DataFrame()

                    return pd.DataFrame(data)

                except:
                    return pd.DataFrame()

                       api_df = load_api_usage()

            if not api_df.empty:

                api_df.columns = api_df.columns.str.strip().str.lower()

                # ================= CLEAN COST COLUMN =================
                if "estimated_cost" in api_df.columns:
                    api_df["estimated_cost"] = pd.to_numeric(api_df["estimated_cost"], errors="coerce")
                else:
                    api_df["estimated_cost"] = pd.to_numeric(api_df.iloc[:, -1], errors="coerce")

                st.markdown("## 💰 API Cost Analytics")

                total_cost = api_df["estimated_cost"].sum()
                st.metric("Total Platform API Cost", f"${total_cost:.4f}")

                # ================= COST PER USER =================
                st.markdown("### 💵 Cost Per User")

                if "user" in api_df.columns:
                    user_cost = api_df.groupby("user")["estimated_cost"].sum().reset_index()
                else:
                    user_cost = api_df.groupby(api_df.columns[1])["estimated_cost"].sum().reset_index()

                st.dataframe(user_cost)

                st.divider()

                # ======================================================
                # ================= AI USAGE BY FEATURE =================
                # ======================================================

                if "feature" in api_df.columns:

                    st.markdown("## 📊 AI Usage by Feature")

                    feature_usage = api_df["feature"].value_counts()

                    st.bar_chart(feature_usage)

                st.divider()

                # ======================================================
                # ================= COST BY MODEL =======================
                # ======================================================

                if "model" in api_df.columns:

                    st.markdown("## 💰 Cost by AI Model")

                    model_cost = api_df.groupby("model")["estimated_cost"].sum()

                    st.bar_chart(model_cost)

                st.divider()

                # ======================================================
                # ================= MOST ACTIVE USERS ===================
                # ======================================================

                if "user" in api_df.columns:

                    st.markdown("## 🔥 Most Active Users")

                    top_users = api_df["user"].value_counts().head(10)

                    st.bar_chart(top_users)

                st.divider()

                # ======================================================
                # ================= TOKEN USAGE TREND ===================
                # ======================================================

                if "total_tokens" in api_df.columns:

                    st.markdown("## 📈 Token Usage Trend")

                    api_df["total_tokens"] = pd.to_numeric(api_df["total_tokens"], errors="coerce")

                    token_usage = api_df["total_tokens"]

                    st.line_chart(token_usage)

            else:
                st.info("No API usage data available yet.")






































