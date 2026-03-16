# ==========================================================
# FUTUREPROOF AI – Production Optimized Version
# Full Original Structure Restored + Mock Separated
# + Admin Portal Added (No Logic Removed)
# + FIX 3: check_request_limit returns bool (no st.stop inside)
# + FIX 4: Resume image OCR via pytesseract
# + FIX 5: CSS in apply_custom_css() function
# + FIX 6: normalize_skills with validation & sanitization
# + FIX 7: Removed unused ThreadPoolExecutor import
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
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
import PyPDF2
from PIL import Image
import io

# FIX 4: OCR import with graceful fallback
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FutureProof Skill Intelligence",
    page_icon="🚀",
    layout="wide"
)

# ================= FIX 5: CSS IN FUNCTION =================

def apply_custom_css():
    css = """
    <style>

    /* ================= MAIN APP BACKGROUND ================= */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #ffffff;
    }

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
    div[data-testid="stRadio"] label p {
        color: #ffffff !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    div[data-testid="stRadio"] label {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    div[data-testid="stRadio"] div {
        opacity: 1 !important;
    }

    div[data-testid="stRadio"] span {
        border-color: #ffffff !important;
    }

    /* ================= INPUT FIELDS ================= */
    div[data-testid="stForm"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSlider"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    label[data-testid="stWidgetLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

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

    button[data-baseweb="tab"] span {
        opacity: 1 !important;
    }

    /* ================= METRIC VISIBILITY ================= */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08) !important;
        padding: 22px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] span,
    div[data-testid="stMetric"] div:first-child {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        font-size: 16px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 32px !important;
        opacity: 1 !important;
    }

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
    div[data-testid="stChatMessage"] {
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    div[data-testid="stChatMessage"] * {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(90deg, #6366f1, #3b82f6) !important;
    }

    div[data-testid="stChatMessage"][data-testid*="assistant"] {
        background: rgba(255,255,255,0.08) !important;
    }

    /* ================= CODE BLOCK VISIBILITY ================= */
    code {
        background-color: rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    pre {
        background-color: #1e293b !important;
        color: #ffffff !important;
        padding: 16px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        overflow-x: auto !important;
    }

    pre code {
        background: none !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMarkdownContainer"] pre {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    div[data-testid="stMarkdownContainer"] code {
        color: #ffffff !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply CSS once at startup
apply_custom_css()

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

# ================= REQUEST CONTROL CONFIG =================
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

# ================= SESSION TRACKING =================
if "current_user" not in st.session_state:
    st.session_state.current_user = "System"

if "current_feature" not in st.session_state:
    st.session_state.current_feature = "General"


# ================= FIX 3: REQUEST LIMIT — RETURNS BOOL, NO st.stop() INSIDE =================

def check_request_limit() -> bool:
    now = time.time()

    if "request_count" not in st.session_state:
        st.session_state.request_count = 0

    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    if now - st.session_state.last_request_time < REQUEST_COOLDOWN:
        st.warning("⏳ Please wait a few seconds before sending another request.")
        return False

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error("⚠️ Session request limit reached. Please refresh the page.")
        return False

    st.session_state.last_request_time = now
    st.session_state.request_count += 1
    return True


# ================= UTILITIES =================

def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")


def safe_llm_call(model, messages, temperature=0.3, retries=2):
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

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

            price_per_1k = MODEL_PRICING.get(model, 0.0005)
            estimated_cost = (total_tokens / 1000) * price_per_1k

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

        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1

        if start != -1 and end != -1:
            cleaned = cleaned[start:end]

        return json.loads(cleaned)

    except Exception as e:
        print("JSON Parse Error:", e)
        return None


# ================= FIX 6: SKILL NORMALIZER WITH VALIDATION =================

MAX_SKILLS = 20
MAX_SKILL_LENGTH = 50

def normalize_skills(skills_input: str) -> list:
    if not skills_input:
        return []

    # Keep letters, numbers, spaces, and common tech characters (+, #, ., -)
    # This allows C++, C#, Node.js, etc. while blocking injection attempts
    sanitized = re.sub(r"[^\w\s+#.\-]", "", skills_input)

    skills = [s.strip().lower() for s in sanitized.split(",") if s.strip()]

    # Drop any single skill that's suspiciously long
    skills = [s for s in skills if len(s) <= MAX_SKILL_LENGTH]

    # Deduplicate preserving order, then cap total count
    seen = set()
    unique_skills = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique_skills.append(s)

    return unique_skills[:MAX_SKILLS]


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
    response = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.3)
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

    try:
        cleaned = response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
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


def generate_mcqs(skills, difficulty, test_mode, mcq_count=10):
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


# ==========================================================
# ===== ADD THESE TWO FUNCTIONS TO YOUR MAIN app.py  =======
# ===== Place them alongside generate_mcqs()         =======
# ==========================================================


def generate_coding_written_questions(skills, difficulty, count):
    """
    Generates written/subjective coding questions.
    Returns a list of dicts with 'question' and 'hints' keys.
    """
    prompt = f"""
You are a senior software engineering interviewer.

Create {count} written coding questions.

Skills: {", ".join(skills)}
Difficulty: {difficulty}

STRICT FORMAT:
Return ONLY a valid JSON array like this:

[
    {{
        "question": "Write a function that takes a list of integers and returns the two numbers that add up to a target sum. Explain your approach and write the code.",
        "hints": "Think about hash maps for O(n) solution."
    }}
]

RULES:
- Each question must require the user to actually WRITE code, not just pick an answer
- Questions should test real implementation ability
- Include a short hint to guide thinking
- Difficulty must match: {difficulty}
  - Beginner: basic loops, conditionals, simple functions
  - Intermediate: data structures, algorithms, OOP
  - Expert: system design logic, optimization, complex algorithms
- No multiple choice options
- No answers provided
- Return ONLY the JSON array, no extra text, no markdown
"""

    response = safe_llm_call(
        MCQ_MODEL,
        [
            {"role": "system", "content": "Return ONLY valid JSON array. No text outside the array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    data = safe_json_load(response)

    if isinstance(data, list):
        # Attach type tag
        for q in data:
            q["type"] = "written"
        return data

    return []


def evaluate_written_answer(question, user_answer, difficulty):
    """
    AI evaluates a written coding answer.
    Returns dict with score (0-10), feedback, and a model answer snippet.
    """

    if not user_answer or not user_answer.strip():
        return {
            "score": 0,
            "feedback": "No answer provided.",
            "model_answer": "N/A"
        }

    prompt = f"""
You are a strict but fair coding interview evaluator.

Question:
{question}

Candidate's Answer:
{user_answer}

Difficulty Level: {difficulty}

Evaluate the answer on these criteria:
1. Correctness — Does the logic/code actually solve the problem?
2. Efficiency — Is the approach reasonably optimal?
3. Clarity — Is the code readable and well-structured?

Return ONLY this JSON (no markdown, no extra text):

{{
    "score": <integer 0 to 10>,
    "feedback": "<2-3 lines: what was good, what was wrong or missing>",
    "model_answer": "<a short correct code snippet or approach in 3-6 lines>"
}}

Scoring guide:
- 9-10: Fully correct, efficient, clean
- 7-8: Correct but minor issues
- 5-6: Partially correct, right direction
- 3-4: Wrong but shows some understanding
- 0-2: Off-track or blank
"""

    response = safe_llm_call(
        MCQ_MODEL,
        [
            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    if not response:
        return {
            "score": 0,
            "feedback": "Evaluation failed. Please try again.",
            "model_answer": "N/A"
        }

    try:
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end != -1:
            cleaned = cleaned[start:end]
        result = json.loads(cleaned)
        return result
    except Exception as e:
        print("Written eval parse error:", e)
        return {
            "score": 0,
            "feedback": "Could not parse evaluation. Raw response logged.",
            "model_answer": "N/A"
        }

# ================= TIMER CONFIG =================

def get_time_limit(difficulty, mcq_count=10, test_mode="Theoretical Knowledge"):
    time_per_q = {"Beginner": 12, "Intermediate": 24, "Expert": 36}
    base = time_per_q.get(difficulty, 12)
    if test_mode == "Coding Based":
        mcq_half = mcq_count // 2
        written_half = mcq_count - mcq_half
        return (mcq_half * base) + (written_half * base * 3)
    return base * mcq_count


# ================= GOOGLE SHEET FUNCTIONS =================

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
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Analytics Load Error: {str(e)}")
        return pd.DataFrame()


# ================= AI MENTOR ENGINE =================

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
        "average": sum(scores) / len(scores),
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


# ================= STUDY MEMORY ENGINE =================

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

    st.session_state.current_feature = "Skill_Intelligence"

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        st.session_state.current_user = name.strip() if name else "Guest"

    with col2:
        education = st.selectbox(
            "Education Level",
            ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
        )

    skills_input = st.text_input("Current Skills (comma-separated)")
    hours = st.slider("Weekly Learning Hours", 1, 40, 10)

    if st.button("🔎 Analyze Skill Intelligence", use_container_width=True):

        # FIX 3: check_request_limit returns bool, st.stop() is explicit here
        if not check_request_limit():
            st.stop()

        skills = normalize_skills(skills_input)

        # FIX 6: Validate skills before proceeding
        if not skills and skills_input.strip():
            st.warning("⚠️ No valid skills detected. Use comma-separated names (e.g. Python, SQL, Excel).")
            st.stop()

        if not skills:
            st.warning("⚠️ Please enter at least one skill.")
            st.stop()

        if len(skills) == MAX_SKILLS:
            st.info(f"ℹ️ Analysis is based on the first {MAX_SKILLS} skills entered.")

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

        with tab1:
            st.header(role)
            st.markdown(f"🧭 Detected Domain: `{domain}`")

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

            readiness = calculate_career_readiness(skills, growth, confidence_value)

            colA, colB = st.columns(2)
            with colA:
                st.metric("Hiring Confidence", f"{confidence_value}%")
            with colB:
                st.metric("Market Risk", risk_value)

            st.markdown("### 📌 Career Outlook")
            st.markdown(summary_value)

            st.markdown("### 🚀 Career Readiness Score")
            st.metric("Overall Career Readiness", f"{readiness['total']}%")

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

        with tab2:
            if growth:
                for skill in growth:
                    st.markdown(f"✔️ {skill}")
                st.markdown(f"⏳ Estimated Timeline: ~{weeks} weeks")
            else:
                st.info("No skill recommendations available.")

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

        with tab4:
            st.markdown(market)

        st.divider()

        rating = st.slider("How useful was this analysis?", 1, 5, 4)
        feedback_text = st.text_area("What can we improve?")

        if st.button("Submit Feedback"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("feedback_log.txt", "a") as f:
                f.write(f"{timestamp} | {name} | {rating} | {education} | {skills_input} | {feedback_text}\n")
            save_feedback([timestamp, name, rating, education, skills_input, feedback_text])
            st.success("✅ Feedback saved successfully!")


# ==========================================================
# ================= MOCK ASSESSMENT ========================
# ==========================================================

elif page == "🎓 Mock Assessment":

    st.header("🎓 Skill-Based Mock Assessment")

    candidate_name = st.text_input("Full Name")
    st.session_state.current_user = candidate_name if candidate_name else "Guest"
    st.session_state.current_feature = "Mock_Assessment"

    if candidate_name:
        performance_data = analyze_user_trend(candidate_name)
        if performance_data:
            mentor_message = generate_mentor_response(candidate_name, performance_data)
            st.success(mentor_message)
            recommended_level = recommend_difficulty(performance_data)
            st.info(f"🎯 Recommended Difficulty Based On Performance: {recommended_level}")
        else:
            st.info(f"👋 Welcome {candidate_name}! Let's start building your career momentum 🚀")

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

    # ================= CODING MODE INFO BANNER =================
    if test_mode == "Coding Based":
        st.info(
            "💡 **Coding Based mode** gives you a mixed test:\n\n"
            "- 🔘 **50% MCQs** — output prediction, debugging, concept questions\n"
            "- ✍️ **50% Written** — real code writing, evaluated by AI"
        )

    # ================= MCQ COUNT SELECTOR =================
    mcq_count = st.select_slider(
        "📝 Number of Questions",
        options=[5, 10, 15, 20],
        value=10,
        help="Total questions in your test. For Coding Based, this is split 50/50."
    )

    # Dynamic time estimate
    time_per_q = {
        "Beginner": 12,
        "Intermediate": 24,
        "Expert": 36
    }
    # Written questions take longer
    if test_mode == "Coding Based":
        mcq_half = mcq_count // 2
        written_half = mcq_count - mcq_half
        estimated_seconds = (mcq_half * time_per_q[difficulty]) + (written_half * time_per_q[difficulty] * 3)
    else:
        estimated_seconds = time_per_q[difficulty] * mcq_count

    estimated_minutes = round(estimated_seconds / 60, 1)
    st.caption(f"⏱️ Estimated time: ~{estimated_minutes} min")
    # ===========================================================

    if "mock_questions" not in st.session_state:
        st.session_state.mock_questions = []

    if st.button("Generate Test"):

        if not check_request_limit():
            st.stop()

        skills = normalize_skills(skills_input)

        if not skills and skills_input.strip():
            st.warning("⚠️ No valid skills detected. Use comma-separated names (e.g. Python, SQL, Excel).")
            st.stop()

        if not skills:
            st.warning("⚠️ Please enter at least one skill.")
            st.stop()

        # ================= CODING BASED: SPLIT 50/50 =================
        if test_mode == "Coding Based":

            mcq_half = mcq_count // 2
            written_half = mcq_count - mcq_half  # handles odd numbers (e.g. 5 → 2 MCQ + 3 written)

            with st.spinner(f"Generating {mcq_half} coding MCQs..."):
                mcq_questions = generate_mcqs(skills, difficulty, "Coding Based", mcq_half)

            with st.spinner(f"Generating {written_half} written coding questions..."):
                written_questions = generate_coding_written_questions(skills, difficulty, written_half)

            # Tag MCQ questions with type
            if mcq_questions:
                for q in mcq_questions:
                    q["type"] = "mcq"

            # Interleave: written and mcq questions alternated for better UX
            # Pattern: W, M, W, M ... so user doesn't get all written at end
            combined = []
            w_idx, m_idx = 0, 0
            written_list = written_questions or []
            mcq_list = mcq_questions or []

            while w_idx < len(written_list) or m_idx < len(mcq_list):
                if w_idx < len(written_list):
                    combined.append(written_list[w_idx])
                    w_idx += 1
                if m_idx < len(mcq_list):
                    combined.append(mcq_list[m_idx])
                    m_idx += 1

            if combined:
                st.session_state.mock_questions = combined
                st.session_state.written_evaluations = {}   # stores AI eval results
            else:
                st.error("Failed to generate coding questions. Try again.")
                st.stop()

        # ================= OTHER MODES: ALL MCQ =================
        else:
            with st.spinner("Generating questions..."):
                questions = generate_mcqs(skills, difficulty, test_mode, mcq_count)

            if questions and isinstance(questions, list):
                for q in questions:
                    q["type"] = "mcq"
                st.session_state.mock_questions = questions
            else:
                st.error("Failed to generate test questions. Try again.")
                st.stop()

        # Common setup after generation
        st.session_state.start_time = time.time()
        st.session_state.time_limit = get_time_limit(difficulty, mcq_count, test_mode)
        st.session_state.exam_submitted = False
        st.session_state.explanations = {}
        st.session_state.result_saved = False
        st.session_state.mcq_count = mcq_count

    # ================= DISPLAY TEST =================
    if st.session_state.get("mock_questions"):

        auto_submit = False
        total_questions = len(st.session_state.mock_questions)

        # Timer
        if "start_time" in st.session_state:
            elapsed = int(time.time() - st.session_state.start_time)
            remaining_time = st.session_state.time_limit - elapsed

            if remaining_time <= 0:
                auto_submit = True
                st.warning("⏰ Time is up! Auto-submitting test...")
            else:
                minutes = remaining_time // 60
                seconds = remaining_time % 60
                st.markdown(f"### ⏳ Time Remaining: {minutes:02d}:{seconds:02d}")

        # ================= QUESTION LOOP =================
        for i, q in enumerate(st.session_state.mock_questions):

            q_type = q.get("type", "mcq")

            # -------- WRITTEN CODING QUESTION --------
            if q_type == "written":

                st.markdown(f"### ✍️ Q{i+1}. {q['question']}")

                if q.get("hints"):
                    st.caption(f"💡 Hint: {q['hints']}")

                # Show text area only if exam not submitted
                if not st.session_state.get("exam_submitted", False):
                    st.text_area(
                        "Write your code / answer here:",
                        key=f"written_{i}",
                        height=200,
                        placeholder="# Write your solution here...\ndef solution():\n    pass"
                    )
                else:
                    # After submission: show what user wrote
                    user_ans = st.session_state.get(f"written_{i}", "")
                    st.code(user_ans if user_ans else "No answer provided.", language="python")

                    # Show AI evaluation result if available
                    eval_result = st.session_state.get("written_evaluations", {}).get(i)
                    if eval_result:
                        score_val = eval_result.get("score", 0)
                        feedback = eval_result.get("feedback", "")
                        model_ans = eval_result.get("model_answer", "")

                        # Color code the score
                        if score_val >= 8:
                            st.success(f"✅ AI Score: {score_val}/10")
                        elif score_val >= 5:
                            st.warning(f"⚠️ AI Score: {score_val}/10")
                        else:
                            st.error(f"❌ AI Score: {score_val}/10")

                        st.markdown("📘 **Feedback:**")
                        st.info(feedback)

                        st.markdown("📗 **Model Answer / Approach:**")
                        st.code(model_ans, language="python")

                    else:
                        st.info("⏳ Evaluating your answer...")

            # -------- MCQ QUESTION --------
            else:

                st.markdown(f"### 🔘 Q{i+1}. {q['question']}")

                st.radio(
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

                    selected = st.session_state.get(f"mock_{i}")

                    if selected == correct_option:
                        st.success(f"✅ Correct Answer: {correct_option}")
                    else:
                        st.error(f"❌ Your Answer: {selected}")
                        st.info(f"✔ Correct Answer: {correct_option}")

                    st.markdown("📘 **Explanation:**")
                    st.info(st.session_state.explanations.get(i, "No explanation available."))

            st.divider()

        # ================= SUBMIT BUTTON =================
        if (st.button("Submit Test") or auto_submit) and not st.session_state.get("exam_submitted"):

            st.session_state.exam_submitted = True

            mcq_score = 0
            mcq_total = 0
            written_score_total = 0
            written_total = 0

            for i, q in enumerate(st.session_state.mock_questions):

                q_type = q.get("type", "mcq")

                # ---- Score MCQ ----
                if q_type == "mcq":
                    mcq_total += 1
                    selected = st.session_state.get(f"mock_{i}")
                    correct_answer = q.get("answer")
                    correct_option = None

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

                    if selected and correct_option:
                        if selected.strip().lower() == correct_option.strip().lower():
                            mcq_score += 1

                    # Generate MCQ explanation
                    if i not in st.session_state.explanations:
                        explanation = generate_explanation(q["question"], correct_option)
                        st.session_state.explanations[i] = explanation

                # ---- Evaluate Written ----
                elif q_type == "written":
                    written_total += 1
                    user_answer = st.session_state.get(f"written_{i}", "")

                    with st.spinner(f"🤖 AI evaluating Q{i+1}..."):
                        eval_result = evaluate_written_answer(
                            q["question"],
                            user_answer,
                            difficulty
                        )

                    if "written_evaluations" not in st.session_state:
                        st.session_state.written_evaluations = {}

                    st.session_state.written_evaluations[i] = eval_result
                    written_score_total += eval_result.get("score", 0)

            # ================= FINAL SCORE CALCULATION =================
            # MCQ: each correct = 10 points (normalized to 10)
            # Written: each question scored 0-10 by AI
            # Combined: weighted average

            if mcq_total > 0:
                mcq_percent = (mcq_score / mcq_total) * 100
            else:
                mcq_percent = 0

            if written_total > 0:
                written_percent = (written_score_total / (written_total * 10)) * 100
            else:
                written_percent = 0

            # Overall: equal weight between MCQ and written halves
            if mcq_total > 0 and written_total > 0:
                overall_percent = (mcq_percent + written_percent) / 2
            elif mcq_total > 0:
                overall_percent = mcq_percent
            else:
                overall_percent = written_percent

            st.session_state.final_score = mcq_score
            st.session_state.final_percent = overall_percent
            st.session_state.mcq_percent = mcq_percent
            st.session_state.written_percent = written_percent
            st.session_state.mcq_total = mcq_total
            st.session_state.written_total = written_total
            st.session_state.written_score_total = written_score_total
            st.session_state.pop("start_time", None)
            st.session_state.pop("time_limit", None)

        # ================= RESULT DISPLAY =================
        if st.session_state.get("exam_submitted"):

            st.markdown("## 📊 Test Result")

            mcq_total = st.session_state.get("mcq_total", 0)
            written_total = st.session_state.get("written_total", 0)

            # Show breakdown only for coding mode (has both types)
            if mcq_total > 0 and written_total > 0:

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "🔘 MCQ Score",
                        f"{st.session_state.get('final_score', 0)}/{mcq_total}",
                        f"{st.session_state.get('mcq_percent', 0):.1f}%"
                    )

                with col2:
                    written_pts = st.session_state.get("written_score_total", 0)
                    st.metric(
                        "✍️ Written Score",
                        f"{written_pts}/{written_total * 10} pts",
                        f"{st.session_state.get('written_percent', 0):.1f}%"
                    )

                with col3:
                    overall = st.session_state.get("final_percent", 0)
                    st.metric(
                        "🏆 Overall Score",
                        f"{overall:.1f}%"
                    )

            else:
                # Pure MCQ test
                st.markdown(f"### Score: {st.session_state.get('final_score', 0)}/{total_questions}")
                st.markdown(f"### Percentage: {st.session_state.get('final_percent', 0):.2f}%")

            # Pass/Fail
            if st.session_state.get("final_percent", 0) >= 80:
                st.success("✅ Qualified (80%+)")
            elif st.session_state.get("final_percent", 0) >= 60:
                st.warning("⚠️ Average Performance (60-79%) — Keep practicing!")
            else:
                st.error("❌ Not Qualified (Below 60%) — Review fundamentals and try again.")

            if not st.session_state.get("result_saved"):
                save_mock_result([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    candidate_name,
                    candidate_email,
                    candidate_education,
                    skills_input,
                    difficulty,
                    test_mode,
                    st.session_state.get("final_score", 0),
                    round(st.session_state.get("final_percent", 0), 2),
                    total_questions,
                    mcq_total,
                    written_total
                ])
                st.session_state.result_saved = True

# ==========================================================
# ================= GUIDED STUDY CHAT ======================
# ==========================================================

elif page == "📚 Guided Study Chat":

    st.header("📚 AI Guided Study Chat")

    candidate_name = st.text_input("Full Name")

    if candidate_name:
        st.session_state.current_user = candidate_name
    else:
        st.session_state.current_user = "Guest"

    st.session_state.current_feature = "Guided_Study_Chat"

    if candidate_name:
        performance_data = analyze_user_trend(candidate_name)
        if performance_data:
            mentor_message = generate_mentor_response(candidate_name, performance_data)
            st.success(mentor_message)
        else:
            st.info(f"👋 Hi {candidate_name}! Let's start building your expertise 🚀")

    education = st.selectbox(
        "Education Level",
        ["High School", "Diploma", "Graduation", "Post Graduation", "Other"]
    )

    topic = st.text_input("Topic You Want To Study")
    level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Expert"])
    learning_goal = st.text_input("Learning Goal (Optional)")

    if "study_chat_started" not in st.session_state:
        st.session_state.study_chat_started = False

    if "study_messages" not in st.session_state:
        st.session_state.study_messages = []

    if st.button("Start Learning"):

        if topic and candidate_name:

            already_studied = check_study_history(candidate_name, education, topic, level)

            if already_studied:

                st.warning(f"You have already studied **{topic} ({level})**.")

                action = st.selectbox(
                    "What would you like to do?",
                    ["Revise", "Study in Detailed Mode", "Test Yourself"]
                )

                if action == "Test Yourself":
                    questions = generate_mcqs([topic], level, "Theoretical Knowledge")
                    if questions:
                        st.session_state.quick_test = questions[:5]

                elif action == "Revise":
                    st.session_state.study_chat_started = True

                elif action == "Study in Detailed Mode":
                    level = "Expert"
                    st.session_state.study_chat_started = True

            else:
                save_study_history([
                    candidate_name,
                    education,
                    topic,
                    level,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
                st.session_state.study_chat_started = True

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

    if st.session_state.study_chat_started:

        st.markdown("---")
        st.subheader(f"📘 Learning Topic: {topic}")

        user_input = st.chat_input("Ask your question about this topic...")

        if user_input:

            # FIX 3: Explicit check
            if not check_request_limit():
                st.stop()

            st.session_state.study_messages.append({"role": "user", "content": user_input})

            messages = [
                {"role": "system", "content": st.session_state.study_context}
            ] + st.session_state.study_messages

            response = safe_llm_call(MAIN_MODEL, messages, temperature=0.4)

            st.session_state.study_messages.append({"role": "assistant", "content": response})

        for msg in st.session_state.study_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if "quick_test" in st.session_state:

        st.markdown("### 📝 Quick Self-Test")
        score = 0

        for i, q in enumerate(st.session_state.quick_test):
            selected = st.radio(q["question"], q["options"], key=f"quick_{i}")
            if selected == q["options"][q["answer"]]:
                score += 1

        if st.button("Submit Quick Test"):
            percent = (score / len(st.session_state.quick_test)) * 100
            if percent >= 80:
                st.success("🔥 Excellent! You are ready for higher level or next topic!")
            else:
                st.info("You may revise this level once more before upgrading.")
            del st.session_state.quick_test


# ==========================================================
# ================= AI JOB FINDER (PREMIUM) ================
# ==========================================================

elif page == "💼 AI Job Finder (Premium)":

    st.header("💼 AI Career Job Finder")
    st.success("🎯 AI-Powered Smart Job Matching Activated")

    name = st.text_input("Full Name")

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
    experience = st.selectbox("Years of Experience", ["Fresher", "1-2 Years", "3-5 Years", "5+ Years"])
    current_field = st.text_input("Current Field / Industry")
    target_role = st.text_input("Target Job Profile You Are Looking For")

    resume_file = st.file_uploader(
        "Upload Resume (PDF, JPG, PNG supported)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if st.button("🔍 Find Suitable Jobs"):

        if name and skills_input and target_role:

            # FIX 3: Explicit check
            if not check_request_limit():
                st.stop()

            skills = normalize_skills(skills_input)

            # FIX 6: Validate skills
            if not skills and skills_input.strip():
                st.warning("⚠️ No valid skills detected. Use comma-separated names (e.g. Python, SQL, Excel).")
                st.stop()

            with st.spinner("Analyzing profile and matching jobs..."):

                resume_text = ""

                if resume_file is not None:

                    # PDF extraction
                    if resume_file.type == "application/pdf":
                        try:
                            pdf_reader = PyPDF2.PdfReader(resume_file)
                            pages = [page.extract_text() for page in pdf_reader.pages]
                            resume_text = "\n".join(p for p in pages if p)
                            if not resume_text.strip():
                                resume_text = "PDF uploaded but no text could be extracted."
                            else:
                                st.success(f"✅ Extracted {len(resume_text.split())} words from PDF resume.")
                        except Exception as e:
                            resume_text = "PDF uploaded (text extraction failed)."
                            st.warning(f"⚠️ PDF extraction issue: {e}")

                    # FIX 4: Image OCR with pytesseract
                    elif resume_file.type in ["image/png", "image/jpeg"]:
                        image = Image.open(resume_file)
                        st.image(image, caption="Uploaded Resume", use_column_width=True)

                        if OCR_AVAILABLE:
                            try:
                                resume_text = pytesseract.image_to_string(image).strip()
                                if not resume_text:
                                    resume_text = "Image uploaded but no readable text was found."
                                    st.warning("⚠️ No text detected in the image. Try uploading a clearer scan or a PDF.")
                                else:
                                    st.success(f"✅ Extracted {len(resume_text.split())} words from resume image.")
                            except Exception as e:
                                resume_text = "OCR extraction failed."
                                st.warning(f"⚠️ Could not extract text from image: {e}")
                        else:
                            resume_text = "Resume uploaded as image (OCR not available — install pytesseract for text extraction)."
                            st.info("💡 For best results, upload a PDF. Image OCR requires pytesseract + tesseract-ocr.")

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

            encoded_role = target_role.replace(" ", "%20")
            st.markdown("### 🔗 Direct Job Search Links")
            st.markdown(f"🔹 [LinkedIn Jobs](https://www.linkedin.com/jobs/search/?keywords={encoded_role})")
            st.markdown(f"🔹 [Indeed Jobs](https://www.indeed.com/jobs?q={encoded_role})")
            st.markdown(f"🔹 [Naukri Jobs](https://www.naukri.com/{target_role.replace(' ', '-')}-jobs)")
            st.markdown(f"🔹 [Glassdoor Jobs](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={encoded_role})")

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

            st.markdown("## 📈 Difficulty Breakdown")
            difficulty_counts = df["difficulty"].value_counts()
            st.bar_chart(difficulty_counts)

            st.divider()

            st.markdown("## 📊 Score Distribution")
            st.bar_chart(df["percent"])

            st.divider()

            st.markdown("## 🏆 Top Performers")
            top_5 = df.sort_values("percent", ascending=False).head(5)
            st.dataframe(top_5[["candidate_name", "candidate_email", "difficulty", "percent"]])

            st.divider()

            st.markdown("## 📂 Full Dataset")
            st.dataframe(df)

            # ================= API COST ANALYTICS =================
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

                if "estimated_cost" in api_df.columns:
                    api_df["estimated_cost"] = pd.to_numeric(api_df["estimated_cost"], errors="coerce")

                if "total_tokens" in api_df.columns:
                    api_df["total_tokens"] = pd.to_numeric(api_df["total_tokens"], errors="coerce")

                if "timestamp" in api_df.columns:
                    api_df["timestamp"] = pd.to_datetime(api_df["timestamp"], errors="coerce")
                    today_start = pd.Timestamp.now().normalize()
                    today_end = today_start + pd.Timedelta(days=1)

                    today_df = api_df[
                        (api_df["timestamp"] >= today_start) &
                        (api_df["timestamp"] < today_end)
                    ]

                    requests_today = len(today_df)
                    active_users = today_df["user"].nunique() if "user" in today_df.columns else 0
                    today_cost = today_df["estimated_cost"].sum() if "estimated_cost" in today_df.columns else 0
                    avg_tokens = today_df["total_tokens"].mean() if "total_tokens" in today_df.columns else 0

                    st.markdown("## 🧠 Platform Health (Today)")

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric("📊 Requests Today", requests_today)
                    with c2:
                        st.metric("⚡ Avg Tokens / Request", int(avg_tokens) if avg_tokens else 0)
                    with c3:
                        st.metric("💰 Today's AI Cost", f"${today_cost:.4f}")
                    with c4:
                        st.metric("👥 Active Users Today", active_users)

                    st.divider()

                if "estimated_cost" in api_df.columns:
                    api_df["estimated_cost"] = pd.to_numeric(api_df["estimated_cost"], errors="coerce")
                else:
                    api_df["estimated_cost"] = pd.to_numeric(api_df.iloc[:, -1], errors="coerce")

                st.markdown("## 💰 API Cost Analytics")
                total_cost = api_df["estimated_cost"].sum()
                st.metric("Total Platform API Cost", f"${total_cost:.4f}")

                st.markdown("### 💵 Cost Per User")
                if "user" in api_df.columns:
                    user_cost = api_df.groupby("user")["estimated_cost"].sum().reset_index()
                else:
                    user_cost = api_df.groupby(api_df.columns[1])["estimated_cost"].sum().reset_index()
                st.dataframe(user_cost)

                st.divider()

                if "feature" in api_df.columns:
                    st.markdown("## 📊 AI Usage by Feature")
                    feature_usage = api_df["feature"].value_counts()
                    st.bar_chart(feature_usage)

                st.divider()

                if "model" in api_df.columns:
                    st.markdown("## 💰 Cost by AI Model")
                    model_cost = api_df.groupby("model")["estimated_cost"].sum()
                    st.bar_chart(model_cost)

                st.divider()

                if "user" in api_df.columns:
                    st.markdown("## 🔥 Most Active Users")
                    top_users = api_df["user"].value_counts().head(10)
                    st.bar_chart(top_users)

                st.divider()

                if "total_tokens" in api_df.columns:
                    st.markdown("## 📈 Token Usage Trend")
                    api_df["total_tokens"] = pd.to_numeric(api_df["total_tokens"], errors="coerce")
                    st.line_chart(api_df["total_tokens"])

            else:
                st.info("No API usage data available yet.")

        else:
            st.error("❌ Invalid Admin Credentials")
