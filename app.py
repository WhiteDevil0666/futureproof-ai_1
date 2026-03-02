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

        domain = detect_domain_cached(skills_tuple) or "General Domain"
        role = infer_role_cached(skills_tuple, domain) or "Specialist"

        with ThreadPoolExecutor() as executor:
            growth = executor.submit(generate_growth, role, domain).result() or []
            certifications = executor.submit(generate_certifications, role, domain).result() or []
            market = executor.submit(generate_market, role, domain).result() or "Market data unavailable."
            confidence = executor.submit(generate_confidence, role, domain).result()
            platforms = executor.submit(generate_platforms, role, domain, skills).result() or {"free": [], "paid": []}

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

                import re

                conf_match = re.search(r"(\d+)%", confidence)
                risk_match = re.search(r"Risk:\s*(Low|Medium|High)", confidence)
                summary_match = re.search(r"Summary:\s*(.*)", confidence)

                if conf_match:
                    confidence_value = int(conf_match.group(1))

                if risk_match:
                    risk_value = risk_match.group(1)

                if summary_match:
                    summary_value = summary_match.group(1)

            colA, colB = st.columns(2)
            with colA:
                st.metric("Hiring Confidence", f"{confidence_value}%")
            with colB:
                st.metric("Market Risk", risk_value)

            st.markdown("### 📌 Career Outlook")
            st.markdown(summary_value)

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

    # ✅ NEW: Test Mode Selector
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

                # ---------- SAFE ANSWER HANDLING ----------
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
                    score += 1

                # ---------- GENERATE EXPLANATION ----------
                if i not in st.session_state.explanations:
                    explanation = generate_explanation(
                        q["question"],
                        correct_option
                    )
                    st.session_state.explanations[i] = explanation

            percent = (score / len(st.session_state.mock_questions)) * 100

            st.session_state.final_score = score
            st.session_state.final_percent = percent

            # Stop timer
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

                # ---------- SAFE ANSWER HANDLING ----------
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

            # Save once
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

            # Define metric card INSIDE login block
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

        else:
            st.error("❌ Invalid Admin Credentials")
















