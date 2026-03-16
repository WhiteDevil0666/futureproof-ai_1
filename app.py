# ==========================================================
# FUTUREPROOF AI – v5.0 Production Build
# ──────────────────────────────────────────────────────────
# v5 UPGRADES:
#   UP-9   AI Career Copilot — persistent personal AI mentor
#          connects ALL modules into one career OS:
#          dashboard · weekly plan · gap analysis · chat
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import time
import uuid
import warnings
from datetime import datetime
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
import PyPDF2
from PIL import Image
import io

# ── OCR with path env support + version check ─────────────
try:
    import pytesseract
    _tess_path = os.getenv("TESSERACT_PATH")
    if _tess_path:
        pytesseract.pytesseract.tesseract_cmd = _tess_path
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ── Vector memory (ChromaDB persistent, FAISS fallback) ───
VECTOR_MEMORY_AVAILABLE = False
CHROMA_AVAILABLE        = False
FAISS_AVAILABLE         = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

if _ST_AVAILABLE:
    try:
        import chromadb
        CHROMA_AVAILABLE        = True
        VECTOR_MEMORY_AVAILABLE = True
    except ImportError:
        pass
    if not CHROMA_AVAILABLE:
        try:
            import faiss
            FAISS_AVAILABLE         = True
            VECTOR_MEMORY_AVAILABLE = True
        except ImportError:
            pass

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SkillForge – Skill Intelligence Platform",
    page_icon="⚙️",
    layout="wide"
)

# ================= CSS =================
def apply_custom_css():
    css = """
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #ffffff; }
    section.main > div { background-color: transparent !important; }
    section[data-testid="stSidebar"] { background-color: #0b1220 !important; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    div[data-testid="stRadio"] label p { color: #ffffff !important; font-weight: 500 !important; opacity: 1 !important; }
    div[data-testid="stRadio"] label { color: #ffffff !important; opacity: 1 !important; }
    div[data-testid="stRadio"] div { opacity: 1 !important; }
    div[data-testid="stRadio"] span { border-color: #ffffff !important; }
    div[data-testid="stForm"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSlider"] label { color: #ffffff !important; font-weight: 600 !important; opacity: 1 !important; }
    label[data-testid="stWidgetLabel"] { color: #ffffff !important; font-weight: 600 !important; opacity: 1 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        color: white !important; border-radius: 10px;
        height: 3em; font-weight: 600; border: none;
    }
    h1, h2, h3, h4 { color: #ffffff !important; }
    button[data-baseweb="tab"] { color: #ffffff !important; font-weight: 600 !important; opacity: 1 !important; }
    button[data-baseweb="tab"]:hover { color: #60a5fa !important; }
    button[aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #3b82f6 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08) !important; padding: 22px !important;
        border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.08) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] span { color: #ffffff !important; font-weight: 700 !important; opacity: 1 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 900 !important; font-size: 32px !important; opacity: 1 !important; }
    div[data-testid="stTabs"] button { color: #ffffff !important; opacity: 1 !important; font-weight: 600 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 3px solid #3b82f6 !important; color: #ffffff !important; }
    div[data-testid="stChatMessage"] { background-color: rgba(255,255,255,0.05) !important; border-radius: 12px !important; padding: 12px !important; }
    div[data-testid="stChatMessage"] * { color: #ffffff !important; opacity: 1 !important; }
    code { background-color: rgba(255,255,255,0.15) !important; color: #ffffff !important; padding: 4px 8px !important; border-radius: 6px !important; font-weight: 600 !important; }
    pre { background-color: #1e293b !important; color: #ffffff !important; padding: 16px !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    pre code { background: none !important; color: #ffffff !important; font-weight: 500 !important; }

    /* ── Copilot-specific styles ── */
    .copilot-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(59,130,246,0.10));
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 16px; padding: 20px 24px; margin-bottom: 14px;
    }
    .copilot-card h4 { color: #a5b4fc !important; margin: 0 0 8px 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.08em; }
    .copilot-card p  { color: #f1f5f9 !important; margin: 0; font-size: 1.05em; font-weight: 500; }
    .task-card {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin-bottom: 10px;
    }
    .task-card.study    { border-left-color: #6366f1; }
    .task-card.practice { border-left-color: #f59e0b; }
    .task-card.build    { border-left-color: #22c55e; }
    .task-card.apply    { border-left-color: #ec4899; }
    .task-day  { color: #94a3b8; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.06em; }
    .task-text { color: #f1f5f9; font-weight: 600; font-size: 0.95em; margin: 2px 0; }
    .task-why  { color: #64748b; font-size: 0.82em; }
    .task-dur  { color: #3b82f6; font-size: 0.78em; font-weight: 700; }
    .milestone-box {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.08));
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 14px; padding: 18px 22px;
    }
    .milestone-box h4 { color: #86efac !important; margin: 0 0 6px 0; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.08em; }
    .milestone-box p  { color: #f0fdf4 !important; font-weight: 600; margin: 0; }
    .gap-pill {
        display: inline-block;
        background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3);
        border-radius: 20px; padding: 4px 12px;
        font-size: 0.8em; color: #fca5a5 !important; margin: 3px; font-weight: 500;
    }
    .strength-pill {
        display: inline-block;
        background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.25);
        border-radius: 20px; padding: 4px 12px;
        font-size: 0.8em; color: #86efac !important; margin: 3px; font-weight: 500;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_css()

st.markdown('<div class="main-title">⚙️ SkillForge – Skill Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Analyze Skills • Detect Gaps • Build Career Intelligence")

# ================= ENV CONFIG =================
ADMIN_USERNAME = os.getenv("ADMIN_USER")
ADMIN_PASSWORD = os.getenv("ADMIN_PASS")
api_key        = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

client     = Groq(api_key=api_key)
MAIN_MODEL = "llama-3.1-8b-instant"
MCQ_MODEL  = "llama-3.3-70b-versatile"

MODEL_PRICING = {
    "llama-3.1-8b-instant":    0.0002,
    "llama-3.3-70b-versatile": 0.0006,
}

MAX_REQUESTS_PER_SESSION = 60
REQUEST_COOLDOWN         = 3
REQUEST_LOG_FILE         = "request_log.json"

# ================= SIDEBAR =================
st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio("", [
    "🤖 AI Career Copilot",
    "🔎 Skill Intelligence",
    "🎓 Mock Assessment",
    "📚 Guided Study Chat",
    "🤖 AI Learning Agent",
    "🎤 AI Interview Simulator",
    "💼 AI Job Finder (Premium)",
    "🔐 Admin Portal",
])

remaining = MAX_REQUESTS_PER_SESSION - st.session_state.get("request_count", 0)
st.sidebar.caption(f"🤖 AI Requests Remaining: {remaining}")

# Reset state on page switch
if page != "🤖 AI Career Copilot":
    for k in ["copilot_profile", "copilot_guidance", "copilot_started", "copilot_chat_msgs"]:
        st.session_state.pop(k, None)

if page != "🎓 Mock Assessment":
    for k in ["mock_questions", "start_time", "time_limit", "exam_submitted"]:
        st.session_state.pop(k, None)

if page != "📚 Guided Study Chat":
    for k in ["study_chat_started", "study_messages", "study_context"]:
        st.session_state.pop(k, None)

if page != "💼 AI Job Finder (Premium)":
    st.session_state.pop("job_analysis_result", None)

if page != "🎤 AI Interview Simulator":
    for k in ["interview_started", "interview_messages", "interview_context",
              "interview_round", "interview_score_log", "interview_complete"]:
        st.session_state.pop(k, None)

if page != "🤖 AI Learning Agent":
    for k in ["agent_started", "agent_plan", "agent_step", "agent_messages",
              "agent_quiz", "agent_quiz_submitted", "agent_scores",
              "agent_topic", "agent_level", "agent_name"]:
        st.session_state.pop(k, None)

# ================= SESSION TRACKING =================
if "current_user"    not in st.session_state: st.session_state.current_user    = "System"
if "current_feature" not in st.session_state: st.session_state.current_feature = "General"


# ══════════════════════════════════════════════════════════
# FILE-BASED REQUEST LIMIT
# ══════════════════════════════════════════════════════════

def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def _load_request_log() -> dict:
    try:
        with open(REQUEST_LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_request_log(log: dict):
    try:
        if len(log) > 500:
            cutoff = time.time() - 3600
            log    = {sid: ts for sid, ts in log.items()
                      if any(t > cutoff for t in ts)}
        with open(REQUEST_LOG_FILE, "w") as f:
            json.dump(log, f)
    except Exception:
        pass


def check_request_limit() -> bool:
    session_id = _get_session_id()
    now        = time.time()
    log        = _load_request_log()
    cutoff     = now - 3600
    timestamps = [t for t in log.get(session_id, []) if t > cutoff]

    if timestamps and (now - timestamps[-1] < REQUEST_COOLDOWN):
        st.warning("⏳ Please wait a few seconds before sending another request.")
        return False

    if len(timestamps) >= MAX_REQUESTS_PER_SESSION:
        st.error("⚠️ Hourly request limit reached. Please wait before continuing.")
        return False

    timestamps.append(now)
    log[session_id] = timestamps
    _save_request_log(log)
    st.session_state.request_count = len(timestamps)
    return True


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def log_api_usage(event_type, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_usage_log.txt", "a") as f:
        f.write(f"{timestamp} | {event_type} | {status}\n")


def safe_llm_call(model, messages, temperature=0.3, retries=3):
    user    = st.session_state.get("current_user",    "System")
    feature = st.session_state.get("current_feature", "General")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            content = response.choices[0].message.content.strip()

            prompt_tokens     = 0
            completion_tokens = 0
            total_tokens      = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens     = getattr(response.usage, "prompt_tokens",     0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens      = getattr(response.usage, "total_tokens",      0)

            price_per_1k   = MODEL_PRICING.get(model, 0.0005)
            estimated_cost = (total_tokens / 1000) * price_per_1k

            try:
                save_api_usage([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user, feature, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    round(estimated_cost, 6),
                ])
            except Exception as sheet_err:
                print("Sheet Logging Error:", sheet_err)

            log_api_usage(model, "SUCCESS")
            return content

        except Exception as e:
            wait = 2 ** attempt
            print(f"LLM Attempt {attempt+1} failed ({e}). Retrying in {wait}s…")
            time.sleep(wait)

    log_api_usage(model, "FAILED")
    return None


# ═══════════════════════════════════════════════════════════
# SAFE JSON LOADER
# ═══════════════════════════════════════════════════════════

def safe_json_load(text):
    if not text:
        return None
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()

        s = cleaned.find("[")
        e = cleaned.rfind("]") + 1
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e])
            except Exception:
                pass

        s = cleaned.find("{")
        e = cleaned.rfind("}") + 1
        if s != -1 and e > s:
            data = json.loads(cleaned[s:e])
            if isinstance(data, dict):
                for key in ("questions", "mcqs", "items", "data", "results",
                            "roadmap", "weeks", "interview", "rounds"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
            return data

    except Exception as err:
        print("JSON Parse Error:", err)
    return None


# ═══════════════════════════════════════════════════════════
# SKILL NORMALIZER
# ═══════════════════════════════════════════════════════════

MAX_SKILLS    = 20
MAX_SKILL_LEN = 50

def normalize_skills(skills_input: str) -> list:
    if not skills_input:
        return []
    sanitized = re.sub(r"[^\w\s+#.\-]", "", skills_input)
    skills    = [s.strip().lower() for s in sanitized.split(",") if s.strip()]
    skills    = [s for s in skills if len(s) <= MAX_SKILL_LEN]
    seen, out = set(), []
    for s in skills:
        if s not in seen:
            seen.add(s); out.append(s)
    return out[:MAX_SKILLS]


# ═══════════════════════════════════════════════════════════
# VECTOR MEMORY
# ═══════════════════════════════════════════════════════════

CHROMA_DIR = "chroma_study_db"

def _get_embedder():
    if "study_embedder" not in st.session_state:
        st.session_state.study_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return st.session_state.study_embedder


def _get_chroma_collection(user: str, topic: str):
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{user}_{topic}")[:60]
    client_c  = chromadb.PersistentClient(path=CHROMA_DIR)
    return client_c.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_add(user: str, topic: str, question: str, answer: str):
    try:
        col       = _get_chroma_collection(user, topic)
        embedder  = _get_embedder()
        text      = f"Q: {question}\nA: {answer}"
        embedding = embedder.encode([text], normalize_embeddings=True).tolist()[0]
        doc_id    = str(uuid.uuid4())
        col.add(documents=[text], embeddings=[embedding], ids=[doc_id])
    except Exception as e:
        print("ChromaDB add error:", e)


def _chroma_query(user: str, topic: str, query: str, top_k: int = 3) -> str:
    try:
        col = _get_chroma_collection(user, topic)
        if col.count() == 0:
            return ""
        embedder = _get_embedder()
        q_emb    = embedder.encode([query], normalize_embeddings=True).tolist()[0]
        results  = col.query(query_embeddings=[q_emb], n_results=min(top_k, col.count()))
        docs     = results.get("documents", [[]])[0]
        return "\n\n".join(docs)
    except Exception as e:
        print("ChromaDB query error:", e)
        return ""


def _init_faiss():
    if "study_faiss_index" not in st.session_state:
        embedder = _get_embedder()
        dim = embedder.get_sentence_embedding_dimension()
        st.session_state.study_faiss_index  = faiss.IndexFlatL2(dim)
        st.session_state.study_memory_texts = []


def _faiss_add(question: str, answer: str):
    _init_faiss()
    embedder  = _get_embedder()
    text      = f"Q: {question}\nA: {answer}"
    embedding = embedder.encode([text], normalize_embeddings=True).astype("float32")
    st.session_state.study_faiss_index.add(embedding)
    st.session_state.study_memory_texts.append(text)


def _faiss_query(query: str, top_k: int = 3) -> str:
    if "study_faiss_index" not in st.session_state:
        return ""
    index = st.session_state.study_faiss_index
    if index.ntotal == 0:
        return ""
    embedder = _get_embedder()
    q_emb    = embedder.encode([query], normalize_embeddings=True).astype("float32")
    k        = min(top_k, index.ntotal)
    _, idxs  = index.search(q_emb, k)
    texts    = st.session_state.study_memory_texts
    return "\n\n".join(texts[i] for i in idxs[0] if 0 <= i < len(texts))


def add_to_memory(question: str, answer: str):
    if not VECTOR_MEMORY_AVAILABLE:
        return
    user  = st.session_state.get("current_user", "guest")
    topic = st.session_state.get("study_topic",  "general")
    if CHROMA_AVAILABLE:
        _chroma_add(user, topic, question, answer)
    else:
        _faiss_add(question, answer)


def retrieve_memory(query: str, top_k: int = 3) -> str:
    if not VECTOR_MEMORY_AVAILABLE:
        return ""
    user  = st.session_state.get("current_user", "guest")
    topic = st.session_state.get("study_topic",  "general")
    if CHROMA_AVAILABLE:
        return _chroma_query(user, topic, query, top_k)
    return _faiss_query(query, top_k)


def reset_study_memory():
    for k in ("study_faiss_index", "study_memory_texts", "study_embedder"):
        st.session_state.pop(k, None)


# ═══════════════════════════════════════════════════════════
# CACHED DOMAIN / ROLE DETECTION
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def detect_domain_cached(skills_tuple):
    prompt = f"""
You are a career classification engine.
Given these skills: {", ".join(skills_tuple)}
Identify the professional career field (e.g. Data Analytics, Software Engineering, Cybersecurity).
Return ONLY the domain name.
"""
    return safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Classify career domain only."},
        {"role": "user",   "content": prompt},
    ]) or "General Domain"


@st.cache_data(ttl=3600)
def infer_role_cached(skills_tuple, domain):
    prompt = f"""
Skills: {", ".join(skills_tuple)}
Domain: {domain}
Suggest one realistic professional role. Return only the role name.
"""
    return safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return only role name."},
        {"role": "user",   "content": prompt},
    ]) or "Specialist"


# ═══════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════

def generate_growth(role, domain, education=""):
    prompt = f"""
Role: {role} | Domain: {domain} | Education: {education}
Suggest 6 skills that would most increase this person's competitiveness.
Calibrate to education level: fresher→foundational, graduate→intermediate, postgrad→advanced.
Return comma-separated skill names only. No explanations. No numbering.
"""
    r = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}])
    if not r: return []
    return [s.strip().title() for s in re.split(r",|\n", r) if s.strip()][:6]


def generate_certifications(role, domain):
    prompt = f"""
Role: {role} | Domain: {domain}
Suggest 6 globally recognized certifications. Return comma-separated names only. No numbering.
"""
    r = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.3)
    if not r: return []
    return [c.strip() for c in re.split(r",|\n", r) if c.strip()][:6]


def generate_platforms(role, domain, skills):
    prompt = f"""
Role: {role} | Domain: {domain} | Skills: {", ".join(skills)}
Provide certification platforms relevant to this domain.
Return ONLY pure JSON — no markdown, no explanation:
{{"free":[{{"name":"Platform","url":"https://example.com"}}],"paid":[{{"name":"Platform","url":"https://example.com"}}]}}
"""
    r = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return ONLY raw JSON. No text."},
        {"role": "user",   "content": prompt},
    ], temperature=0)
    if not r: return {"free": [], "paid": []}
    try:
        cleaned = r.strip().replace("```json","").replace("```","").strip()
        s = cleaned.find("{"); e = cleaned.rfind("}") + 1
        return json.loads(cleaned[s:e]) if s != -1 and e > s else {"free":[],"paid":[]}
    except Exception:
        return {"free": [], "paid": []}


def generate_market(role, domain, education=""):
    prompt = f"""
Role: {role} | Domain: {domain} | Education: {education}
Explain the job market in 4-6 lines: demand level, typical hiring scale, 3-5 year outlook,
what position someone with "{education}" can realistically target.
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or "Market data unavailable."


def generate_confidence(role, domain, education=""):
    prompt = f"""
Role: {role} | Domain: {domain} | Education: {education}
Return ONLY this format:
Confidence: X%
Risk: Low/Medium/High
Summary: 2-3 lines about market demand and what this education level can expect.
Do NOT add projects, roadmaps, or strategies.
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}]) or \
           "Confidence: 70%\nRisk: Medium\nSummary: Moderate outlook."


def generate_timeline(role, domain, growth_skills, hours_per_week):
    if not growth_skills:
        return 0
    prompt = f"""
Role: {role} | Domain: {domain}
Skills to learn: {", ".join(growth_skills)}
Weekly hours: {hours_per_week}
Estimate realistic weeks to learn these skills from basics.
Return ONLY a single integer. Nothing else.
"""
    r = safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.1)
    if r:
        m = re.search(r"\d+", r)
        if m: return int(m.group())
    return round((len(growth_skills) * 20) / hours_per_week)


# ═══════════════════════════════════════════════════════════
# SKILL GAP DETECTION
# ═══════════════════════════════════════════════════════════

def _skill_match(required_skill: str, user_skills: set) -> str:
    req_clean = required_skill.lower().strip()
    req_words = set(re.split(r"[\s/\-_]+", req_clean)) - {"and","or","the","of","in","for","a"}

    for us in user_skills:
        us_clean = us.lower().strip()
        if req_clean == us_clean:
            return "Have"
        if req_clean.replace(" ","") == us_clean.replace(" ",""):
            return "Have"

    overlap_count = 0
    for us in user_skills:
        us_words = set(re.split(r"[\s/\-_]+", us.lower()))
        common   = req_words & us_words
        if len(common) >= max(1, len(req_words) * 0.5):
            overlap_count += 1

    if overlap_count > 0:
        return "Partial"
    return "Missing"


def validate_skill_gaps(gaps: list, skills_tuple: tuple) -> list:
    user_skills = set(s.lower().strip() for s in skills_tuple)
    validated   = []
    for g in gaps:
        skill        = g.get("skill", "")
        ai_status    = g.get("status", "Missing")
        ground_truth = _skill_match(skill, user_skills)

        if ai_status == "Have" and ground_truth == "Missing":
            g["status"] = "Missing"
        elif ai_status == "Missing" and ground_truth in ("Have", "Partial"):
            g["status"] = ground_truth
        elif ai_status == "Partial" and ground_truth == "Have":
            g["status"] = "Have"
        validated.append(g)
    return validated


@st.cache_data(ttl=3600)
def detect_skill_gaps_cached(skills_tuple, target_role, domain):
    prompt = f"""
Target Role: {target_role}
Domain: {domain}
User's EXACT entered skills (this is the complete list — do NOT assume any other skills):
{", ".join(skills_tuple)}

TASK:
List the 20-25 most important skills required for {target_role} in {domain}.
For EACH required skill, set "status" using ONLY these strict rules:
  - "Have"    → the skill appears VERBATIM or near-verbatim in the user's list above
  - "Partial" → a closely related but less complete skill appears in the user's list
  - "Missing" → the skill is NOT present in the user's list at all

CRITICAL: Do NOT infer, assume, or guess. If a skill is not explicitly in the list, it is Missing.

Return ONLY valid JSON array — no markdown:
[{{"skill":"Skill Name","status":"Have|Missing|Partial","priority":"Critical|Important|Nice to Have","reason":"1 line why it matters for this role"}}]
"""
    r = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array. Be strict: only mark Have if the skill is explicitly in the user list."},
        {"role": "user",   "content": prompt},
    ], temperature=0.1)
    data = safe_json_load(r)
    if not isinstance(data, list):
        return []
    return validate_skill_gaps(data, skills_tuple)


# ═══════════════════════════════════════════════════════════
# LEARNING PATH (ROADMAP)
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def generate_learning_roadmap_cached(role, domain, growth_tuple, hours_per_week, total_weeks):
    if not growth_tuple:
        return []
    prompt = f"""
Role: {role} | Domain: {domain}
Skills to learn: {", ".join(growth_tuple)}
Weekly hours: {hours_per_week} | Total weeks: {total_weeks}

Create a week-by-week learning roadmap distributing the skills across {total_weeks} weeks.

Return ONLY valid JSON array — no markdown, no extra text:
[{{"week":1,"focus":"Skill Name","topics":["topic1","topic2"],"resource":"YouTube / Docs / Practice","milestone":"What learner can do by end of week"}}]
"""
    r = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array."},
        {"role": "user",   "content": prompt},
    ], temperature=0.3)
    data = safe_json_load(r)
    return data if isinstance(data, list) else []


# ═══════════════════════════════════════════════════════════
# RESUME SKILL EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_skills_from_resume(resume_text: str) -> list:
    if not resume_text or len(resume_text.strip()) < 50:
        return []
    prompt = f"""
Extract all technical and professional skills from this resume text.

Resume (first 2000 + last 1000 chars for full coverage):
{resume_text[:2000] + ("…" if len(resume_text) > 3000 else "") + resume_text[-1000:] if len(resume_text) > 2000 else resume_text}

Return ONLY a JSON array of skill name strings:
["skill1","skill2","skill3"]
No explanations. No markdown.
"""
    r = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return ONLY a JSON array of strings."},
        {"role": "user",   "content": prompt},
    ], temperature=0.1)
    data = safe_json_load(r)
    if isinstance(data, list):
        return [str(s).strip() for s in data if s][:30]
    return []


# ═══════════════════════════════════════════════════════════
# AI INTERVIEW SIMULATOR
# ═══════════════════════════════════════════════════════════

INTERVIEW_ROUNDS = [
    {"id": "intro",     "label": "🧊 Round 1 — Ice Breaker",        "focus": "background, motivation, career goals"},
    {"id": "technical", "label": "💻 Round 2 — Technical Deep Dive", "focus": "technical skills, problem-solving, code logic"},
    {"id": "behavioral","label": "🤝 Round 3 — Behavioural",         "focus": "teamwork, conflict, STAR-format stories"},
    {"id": "system",    "label": "🏗️ Round 4 — System Design",      "focus": "architecture, scalability, trade-offs"},
    {"id": "closing",   "label": "🎯 Round 5 — Closing & Feedback",  "focus": "candidate questions, overall impression"},
]


def generate_interview_opening(role, domain, difficulty, round_info, skills):
    prompt = (
        f"You are a senior {role} interviewer. "
        f"Open {round_info['label']} interview. "
        f"Focus: {round_info['focus']}. "
        f"Candidate skills: {', '.join(skills[:8])}. "
        f"Difficulty: {difficulty}. "
        f"Give a brief welcome (1 line) then ask your FIRST question for this round. "
        f"Keep total response under 80 words."
    )
    return safe_llm_call(MCQ_MODEL, [{"role": "user", "content": prompt}], temperature=0.5) or \
           f"Welcome to {round_info['label']}. Let's begin. Tell me about yourself."


def evaluate_interview_answer(question, answer, role, difficulty):
    if not answer or not answer.strip():
        return {"score": 0, "feedback": "No answer provided.", "follow_up": "Could you please attempt an answer?"}
    prompt = f"""
Senior {role} interviewer. Difficulty: {difficulty}.
Question asked: {question}
Candidate answered: {answer}

Evaluate strictly. Return ONLY JSON (no markdown):
{{"score":<0-10>,"feedback":"2-3 lines: strengths + what was missing or could be improved","follow_up":"one sharp follow-up question or probe"}}
Score guide: 9-10 excellent | 7-8 good minor gaps | 5-6 adequate | 3-4 weak | 0-2 off-track/blank
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON."},
        {"role": "user",   "content": prompt},
    ], temperature=0.3)
    if not r:
        return {"score": 0, "feedback": "Evaluation failed.", "follow_up": "Let's continue."}
    try:
        cleaned = r.strip().replace("```json","").replace("```","").strip()
        s = cleaned.find("{"); e = cleaned.rfind("}") + 1
        return json.loads(cleaned[s:e]) if s != -1 and e > s else \
               {"score": 0, "feedback": "Parse error.", "follow_up": "Let's continue."}
    except Exception:
        return {"score": 0, "feedback": "Could not parse evaluation.", "follow_up": "Let's continue."}


def generate_interview_report(role, score_log):
    avg = round(sum(s["score"] for s in score_log) / len(score_log), 1) if score_log else 0
    summary = "\n".join(
        f"Round {s['round']}: {s['question'][:80]}… → {s['score']}/10" for s in score_log
    )
    prompt = f"""
You are a senior {role} interviewer. The candidate has completed a full mock interview.
Score log:
{summary}
Average score: {avg}/10

Write a professional debrief (8-12 lines) covering:
1. Overall performance impression
2. Top 2 strengths demonstrated
3. Top 2 areas needing improvement
4. Hiring recommendation: Strong Yes / Yes / Maybe / No
5. One specific tip to prepare better
"""
    return safe_llm_call(MCQ_MODEL, [{"role": "user", "content": prompt}], temperature=0.4) or \
           f"Interview complete. Average score: {avg}/10."


def save_interview_result(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Interview_Results").sheet1
        except: sheet = gc.create("FutureProof_Interview_Results").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("Interview Sheet Error:", e)


# ═══════════════════════════════════════════════════════════
# REAL JOB AGGREGATION (SerpAPI)
# ═══════════════════════════════════════════════════════════

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def fetch_real_jobs(role, location="India", num=6):
    if not SERPAPI_KEY:
        return []
    try:
        import urllib.request, urllib.parse
        params = urllib.parse.urlencode({
            "engine": "google_jobs", "q": role, "location": location,
            "hl": "en", "api_key": SERPAPI_KEY,
        })
        url = f"https://serpapi.com/search.json?{params}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        jobs = []
        for j in data.get("jobs_results", [])[:num]:
            jobs.append({
                "title":      j.get("title", ""),
                "company":    j.get("company_name", ""),
                "location":   j.get("location", ""),
                "snippet":    j.get("description", "")[:200],
                "apply_link": (j.get("related_links") or [{}])[0].get("link", ""),
                "posted":     (j.get("detected_extensions") or {}).get("posted_at", ""),
                "salary":     (j.get("detected_extensions") or {}).get("salary", "Not listed"),
            })
        return jobs
    except Exception as e:
        print("SerpAPI error:", e)
        return []


def render_job_cards(jobs):
    if not jobs:
        return
    st.markdown("### 💼 Live Job Listings")
    for j in jobs:
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:16px 20px;
margin-bottom:12px;border:1px solid rgba(255,255,255,0.1);">
  <h4 style="margin:0;color:#60a5fa;">{j['title']}</h4>
  <p style="margin:4px 0;color:#94a3b8;">🏢 {j['company']} &nbsp;|&nbsp; 📍 {j['location']}
  &nbsp;|&nbsp; 💰 {j['salary']} &nbsp;|&nbsp; 🕐 {j['posted']}</p>
  <p style="color:#cbd5e1;font-size:0.9em;">{j['snippet']}…</p>
  {"<a href='" + j['apply_link'] + "' target='_blank' style='color:#3b82f6;'>🔗 Apply Now</a>" if j['apply_link'] else ""}
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# JOB SKILL MATCHING SCORE
# ═══════════════════════════════════════════════════════════

def compute_job_match_score(user_skills, jd_text):
    if not jd_text or not jd_text.strip():
        return {}

    semantic_score = 0
    if _ST_AVAILABLE:
        try:
            embedder    = _get_embedder()
            skills_text = ", ".join(user_skills)
            emb_skills  = embedder.encode([skills_text],    normalize_embeddings=True)
            emb_jd      = embedder.encode([jd_text[:1500]], normalize_embeddings=True)
            cos_sim     = float(np.dot(emb_skills[0], emb_jd[0]))
            semantic_score = round(max(0, min(cos_sim, 1)) * 100, 1)
        except Exception as e:
            print("Embedding error:", e)

    jd_lower   = jd_text.lower()
    user_lower = set(s.lower().strip() for s in user_skills)
    jd_words   = set(re.findall(r"\b[a-z][a-z0-9+#.\-]{1,30}\b", jd_lower))
    stopwords  = {"and","the","for","with","using","have","will","able","work",
                  "team","good","strong","knowledge","experience","understanding",
                  "proficiency","familiarity","years","role","position","job",
                  "responsibilities","requirements","preferred","plus","etc"}
    jd_skills  = jd_words - stopwords
    matched    = sorted(user_lower & jd_skills)
    missing    = sorted(jd_skills - user_lower - stopwords)[:15]
    keyword_score = round(len(matched) / max(len(jd_skills), 1) * 100, 1)
    keyword_score = min(keyword_score, 100)

    if _ST_AVAILABLE:
        overall = round(semantic_score * 0.60 + keyword_score * 0.40, 1)
    else:
        overall = keyword_score

    label = (
        "🟢 Excellent Match" if overall >= 75 else
        "🟡 Good Match"      if overall >= 55 else
        "🟠 Partial Match"   if overall >= 35 else
        "🔴 Low Match"
    )

    return {
        "overall_score":    overall,
        "semantic_score":   semantic_score,
        "keyword_score":    keyword_score,
        "matched_keywords": matched[:20],
        "missing_keywords": missing,
        "label":            label,
    }


def save_job_match(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Job_Matches").sheet1
        except: sheet = gc.create("FutureProof_Job_Matches").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("Job Match Sheet Error:", e)


# ═══════════════════════════════════════════════════════════
# AI LEARNING AGENT
# ═══════════════════════════════════════════════════════════

def generate_learning_plan(topic, level, goal):
    prompt = f"""
You are an expert curriculum designer.
Topic: {topic} | Level: {level} | Goal: {goal or "Thorough understanding"}

Break this topic into 5-7 sequential learning modules.
Return ONLY valid JSON array — no markdown:
[{{"step":1,"title":"Module Title","objective":"What the learner can do after this module"}}]
"""
    r = safe_llm_call(MAIN_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array."},
        {"role": "user",   "content": prompt},
    ], temperature=0.3)
    data = safe_json_load(r)
    return data if isinstance(data, list) else []


def generate_module_explanation(topic, module_title, module_objective, level, education, past_context=""):
    context_line = f"\nWhat student already knows: {past_context[:400]}" if past_context else ""
    prompt = (
        f"Expert tutor. Topic: {topic}. Level: {level}. Student background: {education}.\n"
        f"Teach this module: {module_title}\n"
        f"Objective: {module_objective}{context_line}\n"
        f"Rules: clear headings, concrete examples, end with a 1-line summary of what was covered. "
        f"Keep under 300 words."
    )
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.4) or \
           f"Module: {module_title} — explanation unavailable."


def generate_module_quiz(topic, module_title, level):
    prompt = f"""
Create 3 multiple choice questions to test understanding of:
Topic: {topic} | Module: {module_title} | Difficulty: {level}

Return ONLY valid JSON array:
[{{"question":"text","options":["A","B","C","D"],"answer":0,"explanation":"1-line why"}}]
- Exactly 4 options per question
- answer = index 0-3
- No markdown
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array."},
        {"role": "user",   "content": prompt},
    ], temperature=0.4)
    data = safe_json_load(r)
    return data if isinstance(data, list) else []


def generate_re_explanation(topic, module_title, level, wrong_questions):
    wrongs = "; ".join(q.get("question","")[:60] for q in wrong_questions)
    prompt = (
        f"The student struggled with: {wrongs}.\n"
        f"Re-explain '{module_title}' in '{topic}' at {level} level using a completely different "
        f"approach — use analogies, a worked example, or a visual description. "
        f"Under 250 words. End with one key takeaway sentence."
    )
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.5) or \
           "Let's try a different approach for this module."


def generate_mastery_report(name, topic, plan, scores):
    summary = "\n".join(
        f"Module {i+1} ({p.get('title','')[:40]}): {s}%" for i, (p, s) in enumerate(zip(plan, scores))
    )
    avg = round(sum(scores) / len(scores)) if scores else 0
    prompt = f"""
Student: {name} | Topic: {topic}
Module scores:
{summary}
Average: {avg}%

Write a concise mastery report (6-8 lines):
1. Overall mastery level
2. Strongest modules
3. Modules needing review
4. One actionable next step
5. Encouragement
"""
    return safe_llm_call(MCQ_MODEL, [{"role": "user", "content": prompt}], temperature=0.4) or \
           f"Topic complete! Average score: {avg}%."


def save_agent_progress(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Agent_Progress").sheet1
        except: sheet = gc.create("FutureProof_Agent_Progress").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("Agent Progress Sheet Error:", e)


def generate_mcqs(skills, difficulty, test_mode, mcq_count=10):
    if test_mode == "Theoretical Knowledge":
        mode_instr = "Theory-based conceptual MCQs. Focus on definitions, comparisons, best practices. No code."
    elif test_mode == "Logical Thinking":
        mode_instr = "Logical reasoning and scenario-based MCQs. Analytical and problem-solving."
    else:
        mode_instr = "Practical coding MCQs. Code snippets, debugging, output prediction, algorithms."

    prompt = f"""
Create {mcq_count} multiple choice questions.
Skills: {", ".join(skills)} | Difficulty: {difficulty}
Mode: {mode_instr}

Return ONLY valid JSON array:
[{{"question":"text","options":["A","B","C","D"],"answer":0}}]
- Exactly 4 options per question
- answer = index (0-3)
- No explanations, no markdown
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array."},
        {"role": "user",   "content": prompt},
    ], temperature=0.4)
    data = safe_json_load(r)
    if isinstance(data, list): return data[:mcq_count]
    return None


@st.cache_data(ttl=3600)
def cached_generate_mcqs(skills_tuple, difficulty, test_mode, mcq_count):
    return generate_mcqs(list(skills_tuple), difficulty, test_mode, mcq_count)


def generate_coding_written_questions(skills, difficulty, count):
    prompt = f"""
Create {count} written coding questions.
Skills: {", ".join(skills)} | Difficulty: {difficulty}

Return ONLY valid JSON array:
[{{"question":"Write a function that...","hints":"Think about hash maps."}}]
- Require actual code writing
- Include short hint
- No multiple choice, no answers
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON array."},
        {"role": "user",   "content": prompt},
    ], temperature=0.4)
    data = safe_json_load(r)
    if isinstance(data, list):
        for q in data: q["type"] = "written"
        return data
    return []


@st.cache_data(ttl=3600)
def cached_generate_written_questions(skills_tuple, difficulty, count):
    return generate_coding_written_questions(list(skills_tuple), difficulty, count)


def generate_explanation(question, correct_answer):
    prompt = f"""
Question: {question}
Correct Answer: {correct_answer}
Explain briefly (2-4 lines) why this answer is correct. Educational and clear.
"""
    return safe_llm_call(MAIN_MODEL, [{"role": "user", "content": prompt}], temperature=0.3) or "Explanation unavailable."


def evaluate_written_answer(question, user_answer, difficulty):
    if not user_answer or not user_answer.strip():
        return {"score": 0, "feedback": "No answer provided.", "model_answer": "N/A"}
    prompt = f"""
Evaluate this coding answer strictly.
Question: {question}
Answer: {user_answer}
Difficulty: {difficulty}

Return ONLY JSON — no markdown:
{{"score":<0-10>,"feedback":"2-3 lines: what was good / wrong","model_answer":"short correct code 3-6 lines"}}
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON."},
        {"role": "user",   "content": prompt},
    ], temperature=0.2)
    if not r: return {"score": 0, "feedback": "Evaluation failed.", "model_answer": "N/A"}
    try:
        cleaned = r.strip().replace("```json","").replace("```","").strip()
        s = cleaned.find("{"); e = cleaned.rfind("}") + 1
        return json.loads(cleaned[s:e]) if s != -1 and e > s else {"score": 0, "feedback": "Parse error.", "model_answer": "N/A"}
    except Exception:
        return {"score": 0, "feedback": "Could not parse evaluation.", "model_answer": "N/A"}


def get_time_limit(difficulty, mcq_count=10, test_mode="Theoretical Knowledge"):
    tpq = {"Beginner": 12, "Intermediate": 24, "Expert": 36}
    base = tpq.get(difficulty, 12)
    if test_mode == "Coding Based":
        half = mcq_count // 2
        return (half * base) + ((mcq_count - half) * base * 3)
    return base * mcq_count


# ═══════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════

def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(st.secrets["GOOGLE_SERVICE_ACCOUNT"], scopes=scopes)
    return gspread.authorize(creds)


def save_feedback(data_row):
    try:
        _gs_client().open("FutureProof_Feedback").sheet1.append_row(data_row)
    except Exception as e:
        st.error(f"Google Sheet Error: {e}")


def save_mock_result(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Mock_Results").sheet1
        except: sheet = gc.create("FutureProof_Mock_Results").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        st.error(f"Mock Sheet Error: {e}")


def save_api_usage(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_API_Usage").sheet1
        except: sheet = gc.create("FutureProof_API_Usage").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("API Usage Logging Error:", e)


def save_study_history(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Study_History").sheet1
        except: sheet = gc.create("FutureProof_Study_History").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        st.error(f"Study History Error: {e}")


def check_study_history(name, education, topic, level):
    try:
        sheet = _gs_client().open("FutureProof_Study_History").sheet1
        data  = sheet.get_all_records()
        if not data: return False
        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower()
        return not df[
            (df["name"].str.lower()      == name.lower())     &
            (df["education"].str.lower() == education.lower()) &
            (df["topic"].str.lower()     == topic.lower())     &
            (df["level"].str.lower()     == level.lower())
        ].empty
    except Exception:
        return False


def save_interview_result(data_row):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Interview_Results").sheet1
        except: sheet = gc.create("FutureProof_Interview_Results").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("Interview Sheet Error:", e)


# ═══════════════════════════════════════════════════════════
# ADMIN ANALYTICS
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_mock_results():
    try:
        sheet = _gs_client().open("FutureProof_Mock_Results").sheet1
        data  = sheet.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Analytics Load Error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# AI MENTOR ENGINE
# ═══════════════════════════════════════════════════════════

def analyze_user_trend(name):
    df = load_mock_results()
    if df.empty: return None
    df.columns = df.columns.str.strip().str.lower()
    if "candidate_name" not in df.columns or "percent" not in df.columns: return None
    user_df = df[df["candidate_name"].str.lower() == name.lower()]
    if user_df.empty: return None
    scores = pd.to_numeric(user_df["percent"], errors="coerce").dropna().tolist()
    if not scores: return None
    trend = "stable"
    if len(scores) >= 2:
        trend = "improving" if scores[-1] > scores[-2] else ("declining" if scores[-1] < scores[-2] else "stable")
    return {
        "latest": scores[-1], "best": max(scores),
        "average": sum(scores)/len(scores), "total_tests": len(scores), "trend": trend,
    }


def recommend_difficulty(perf):
    if not perf: return "Beginner"
    return "Expert" if perf["latest"] >= 85 else ("Intermediate" if perf["latest"] >= 60 else "Beginner")


def generate_mentor_response(name, perf):
    prompt = f"""
AI Career Mentor. User: {name}. Performance: {perf}.
Greet personally, analyze trend, encourage, suggest next step. Professional + motivating. Under 8 lines.
"""
    return safe_llm_call("llama-3.3-70b-versatile", [{"role":"user","content":prompt}], temperature=0.5)


# ═══════════════════════════════════════════════════════════
# ████████  AI CAREER COPILOT — FUNCTIONS  ████████████████
# ═══════════════════════════════════════════════════════════

def save_copilot_profile(data_row: list):
    try:
        gc = _gs_client()
        try:    sheet = gc.open("FutureProof_Copilot_Profiles").sheet1
        except: sheet = gc.create("FutureProof_Copilot_Profiles").sheet1
        sheet.append_row(data_row)
    except Exception as e:
        print("Copilot Profile Sheet Error:", e)


def _load_interview_history(name: str) -> dict:
    try:
        sheet = _gs_client().open("FutureProof_Interview_Results").sheet1
        rows  = sheet.get_all_records()
        if not rows: return {}
        df = pd.DataFrame(rows)
        df.columns = df.columns.str.lower().str.strip()
        if "name" not in df.columns: return {}
        user_df = df[df["name"].str.lower() == name.lower()]
        if user_df.empty: return {}
        scores = pd.to_numeric(
            user_df["avg_score"] if "avg_score" in user_df.columns else pd.Series(),
            errors="coerce"
        ).dropna()
        return {
            "avg_score":   float(scores.mean()) if len(scores) else 0.0,
            "rounds_done": int(user_df["rounds_completed"].sum())
                           if "rounds_completed" in user_df.columns else 0,
            "sessions":    len(user_df),
        }
    except Exception:
        return {}


def _load_agent_history(name: str) -> dict:
    try:
        sheet = _gs_client().open("FutureProof_Agent_Progress").sheet1
        rows  = sheet.get_all_records()
        if not rows: return {}
        df = pd.DataFrame(rows)
        df.columns = df.columns.str.lower().str.strip()
        if "name" not in df.columns: return {}
        user_df = df[df["name"].str.lower() == name.lower()]
        if user_df.empty: return {}
        avgs   = pd.to_numeric(
            user_df["avg_mastery"] if "avg_mastery" in user_df.columns else pd.Series(),
            errors="coerce"
        ).dropna()
        topics = user_df["topic"].dropna().tolist() if "topic" in user_df.columns else []
        return {
            "avg_score":    float(avgs.mean()) if len(avgs) else 0.0,
            "modules_done": int(user_df["modules_completed"].sum())
                            if "modules_completed" in user_df.columns else 0,
            "topics":       topics[-5:],
        }
    except Exception:
        return {}


def build_career_profile(name, goal_role, domain, education, skills,
                         skill_gaps, mock_history, interview_history, agent_history):
    skill_base    = min(len(skills) * 5, 30)
    gap_have      = [g for g in skill_gaps if g.get("status") == "Have"]
    gap_base      = round(len(gap_have) / max(len(skill_gaps), 1) * 30)
    mock_avg      = mock_history.get("average", 0)
    mock_base     = round(min(mock_avg, 100) / 100 * 20)
    iv_avg        = interview_history.get("avg_score", 0)
    iv_base       = round(min(iv_avg * 10, 100) / 100 * 20)
    readiness     = min(skill_base + gap_base + mock_base + iv_base, 100)

    critical_gaps  = [g["skill"] for g in skill_gaps
                      if g.get("status") in ("Missing","Partial") and g.get("priority") == "Critical"][:5]
    important_gaps = [g["skill"] for g in skill_gaps
                      if g.get("status") in ("Missing","Partial") and g.get("priority") == "Important"][:5]

    return {
        "name":             name,
        "goal_role":        goal_role,
        "domain":           domain,
        "education":        education,
        "skills":           skills,
        "skill_count":      len(skills),
        "readiness":        readiness,
        "critical_gaps":    critical_gaps,
        "important_gaps":   important_gaps,
        "mock_avg":         mock_avg,
        "mock_tests":       mock_history.get("total_tests", 0),
        "mock_trend":       mock_history.get("trend", "no data"),
        "mock_latest":      mock_history.get("latest", 0),
        "interview_avg":    iv_avg,
        "interview_rounds": interview_history.get("rounds_done", 0),
        "learning_modules": agent_history.get("modules_done", 0),
        "learning_avg":     agent_history.get("avg_score", 0),
        "learning_topics":  agent_history.get("topics", []),
    }


def generate_copilot_guidance(profile: dict) -> dict:
    profile_str = json.dumps(
        {k: v for k, v in profile.items() if k != "skills"}, indent=2
    )
    prompt = f"""
You are an elite AI Career Mentor with deep knowledge of hiring, skill development, and career growth.

Here is the candidate's complete career profile:
{profile_str}

Their goal: {profile['goal_role']} in {profile['domain']}
Education: {profile['education']}
Current readiness: {profile['readiness']}%

Based on ALL of this data, generate a highly personalised weekly career plan.

Return ONLY valid JSON — no markdown, no extra text:
{{
  "weekly_plan": [
    {{"day": "Monday",    "task": "Specific action", "type": "study|practice|apply|build", "duration": "30 min", "why": "1 line reason"}},
    {{"day": "Tuesday",   "task": "...", "type": "...", "duration": "...", "why": "..."}},
    {{"day": "Wednesday", "task": "...", "type": "...", "duration": "...", "why": "..."}},
    {{"day": "Friday",    "task": "...", "type": "...", "duration": "...", "why": "..."}},
    {{"day": "Weekend",   "task": "...", "type": "...", "duration": "...", "why": "..."}}
  ],
  "strengths":         ["strength 1", "strength 2", "strength 3"],
  "focus_areas":       ["area 1 with specific advice", "area 2 with specific advice"],
  "next_milestone":    "The single most important goal to achieve in the next 2-4 weeks",
  "motivation":        "1-2 lines of personalised, data-driven encouragement based on their actual numbers",
  "readiness_label":   "one of: Foundation Building | Skill Development | Interview Ready | Job Ready | Promotion Track",
  "job_readiness_tip": "One specific thing that would most increase job offer chances right now"
}}

Rules:
- weekly_plan must have exactly 5 items
- Be SPECIFIC — mention actual skill names from their profile, real numbers
- Be HONEST — if readiness is low, say so constructively
- Calibrate to their education and experience level
"""
    r = safe_llm_call(MCQ_MODEL, [
        {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
        {"role": "user",   "content": prompt},
    ], temperature=0.4)

    data = safe_json_load(r)
    if isinstance(data, dict) and "weekly_plan" in data:
        return data

    return {
        "weekly_plan": [
            {"day": "Monday",    "task": "Study top critical skill gap",       "type": "study",    "duration": "45 min", "why": "Closes your biggest gap"},
            {"day": "Wednesday", "task": "Take a Mock Assessment",             "type": "practice", "duration": "30 min", "why": "Track your progress"},
            {"day": "Thursday",  "task": "Run one Interview Simulator round",  "type": "practice", "duration": "20 min", "why": "Build confidence"},
            {"day": "Saturday",  "task": "Build a small portfolio project",    "type": "build",    "duration": "2 hrs",  "why": "Demonstrate skills"},
            {"day": "Sunday",    "task": "Review your Skill Gap report",       "type": "study",    "duration": "20 min", "why": "Stay focused"},
        ],
        "strengths":         ["Consistent engagement", "Multi-skill foundation"],
        "focus_areas":       ["Close critical skill gaps", "Improve mock test scores"],
        "next_milestone":    f"Achieve 80%+ on Mock Assessment for {profile.get('goal_role','')}",
        "motivation":        "Every session moves you closer. Keep the momentum.",
        "readiness_label":   "Skill Development",
        "job_readiness_tip": "Take 3 mock tests and review all explanations carefully.",
    }


def generate_copilot_chat_response(profile: dict, user_message: str, history: list) -> str:
    profile_summary = (
        f"Candidate: {profile['name']} | Goal: {profile['goal_role']} | "
        f"Domain: {profile['domain']} | Education: {profile['education']} | "
        f"Readiness: {profile['readiness']}% | "
        f"Critical gaps: {', '.join(profile['critical_gaps'][:3]) or 'None'} | "
        f"Mock avg: {profile['mock_avg']:.1f}% | Interview avg: {profile['interview_avg']:.1f}/10"
    )
    system = (
        f"You are an AI Career Copilot — a personal mentor for this candidate.\n"
        f"Profile: {profile_summary}\n\n"
        "Rules:\n"
        "- Answer ONLY career, learning, skill, or job-related questions\n"
        "- Always reference their actual data (scores, gaps, role) in your answers\n"
        "- Be direct, specific, and actionable — no vague platitudes\n"
        "- If they ask what to study, pick from their actual critical gaps\n"
        "- If they ask about jobs, factor in their current readiness score\n"
        "- Keep responses under 150 words unless they ask for detail\n"
        "- End every response with one concrete next action"
    )
    messages = [{"role": "system", "content": system}]
    messages += history[-10:]
    messages.append({"role": "user", "content": user_message})
    return safe_llm_call(MCQ_MODEL, messages, temperature=0.4) or \
           "I couldn't generate a response. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████  AI CAREER COPILOT — PAGE  ████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

if page == "🤖 AI Career Copilot":

    st.session_state.current_feature = "AI_Career_Copilot"

    st.markdown("""
    <div style="padding: 8px 0 4px 0;">
        <h1 style="margin:0; font-size:2em;">🤖 AI Career Copilot</h1>
        <p style="color:#64748b; margin:4px 0 0 0; font-size:0.95em;">
            Your personal AI mentor — connects Skill Intelligence, Mock Tests,
            Interview Simulator &amp; Learning Agent into one career operating system.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── SETUP FORM ──────────────────────────────────────────────────────────
    if not st.session_state.get("copilot_started"):

        st.markdown("### 👤 Set Up Your Career Copilot")
        st.caption(
            "Answer a few questions. The Copilot pulls your results from "
            "all SkillForge modules and generates your personalised plan."
        )
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            cp_name = st.text_input("Your Full Name",  key="cp_name_input",
                          placeholder="Used to pull your test history")
            cp_goal = st.text_input("Target Role",     key="cp_goal_input",
                          placeholder="e.g. Data Scientist, Backend Engineer")
        with col2:
            cp_edu  = st.text_input("Education Level", key="cp_edu_input",
                          placeholder="e.g. B.Tech, MBA, Self-taught")
            cp_exp  = st.selectbox("Experience Level",
                          ["Fresher","1-2 Years","3-5 Years","5+ Years"])

        cp_skills = st.text_input("Your Current Skills (comma-separated)",
                        placeholder="python, sql, machine learning, excel…")

        st.markdown("")
        with st.expander("🔍 What data does the Copilot collect?", expanded=False):
            st.markdown("""
| Module | Data Used |
|--------|-----------|
| 🔎 Skill Intelligence | Skill gaps, role domain, readiness score |
| 🎓 Mock Assessment | Average score, trend, number of tests |
| 🎤 Interview Simulator | Round scores, average performance |
| 🤖 AI Learning Agent | Module completion, mastery scores, topics |
            """)

        st.markdown("")
        if st.button("🚀 Activate My Career Copilot", use_container_width=True):
            if not cp_name or not cp_goal or not cp_skills:
                st.warning("⚠️ Please fill in Name, Target Role, and Skills.")
            else:
                if not check_request_limit(): st.stop()
                skills_clean = normalize_skills(cp_skills)
                if not skills_clean:
                    st.warning("⚠️ No valid skills detected."); st.stop()

                with st.spinner("🧠 Connecting all your SkillForge data…"):
                    domain        = detect_domain_cached(tuple(skills_clean)) or "Technology"
                    gaps          = detect_skill_gaps_cached(tuple(skills_clean), cp_goal, domain)
                    mock_hist     = analyze_user_trend(cp_name) or {
                        "average": 0, "latest": 0, "trend": "no data", "total_tests": 0
                    }
                    interview_hist = _load_interview_history(cp_name)
                    agent_hist     = _load_agent_history(cp_name)

                with st.spinner("⚡ Building your career profile…"):
                    profile = build_career_profile(
                        name=cp_name, goal_role=cp_goal, domain=domain,
                        education=cp_edu, skills=skills_clean,
                        skill_gaps=gaps, mock_history=mock_hist,
                        interview_history=interview_hist, agent_history=agent_hist,
                    )

                with st.spinner("🤖 Generating your personalised weekly plan…"):
                    guidance = generate_copilot_guidance(profile)

                st.session_state.copilot_started   = True
                st.session_state.copilot_profile   = profile
                st.session_state.copilot_guidance  = guidance
                st.session_state.copilot_chat_msgs = []
                st.session_state.current_user      = cp_name

                save_copilot_profile([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    cp_name, cp_goal, domain, cp_edu, cp_exp,
                    ", ".join(skills_clean),
                    profile["readiness"],
                    guidance.get("readiness_label", ""),
                    guidance.get("next_milestone", ""),
                    json.dumps(profile["critical_gaps"]),
                    json.dumps(guidance.get("weekly_plan", [])),
                ])
                st.rerun()

    # ── ACTIVE DASHBOARD ────────────────────────────────────────────────────
    else:
        profile  = st.session_state.copilot_profile
        guidance = st.session_state.copilot_guidance

        name_cp   = profile["name"]
        goal_cp   = profile["goal_role"]
        domain_cp = profile["domain"]
        readiness = profile["readiness"]

        tab_dash, tab_plan, tab_gaps, tab_chat, tab_reset = st.tabs([
            "🏠 Dashboard", "📅 Weekly Plan", "🔍 Gap Analysis", "💬 Ask Copilot", "🔄 Reset",
        ])

        # ═══════════════ DASHBOARD ══════════════════════════
        with tab_dash:
            hour     = datetime.now().hour
            greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
            ring_color = "#22c55e" if readiness >= 75 else "#f59e0b" if readiness >= 50 else "#ef4444"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(59,130,246,0.12));
                 border:1px solid rgba(99,102,241,0.3);border-radius:18px;padding:24px 28px;margin-bottom:20px;">
                <h2 style="margin:0 0 4px 0;">{greeting}, {name_cp} 👋</h2>
                <p style="color:#94a3b8;margin:0;">
                    Your AI Career Copilot is active · Goal: <strong style="color:#a5b4fc;">{goal_cp}</strong>
                    &nbsp;·&nbsp; Domain: <strong style="color:#60a5fa;">{domain_cp}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("🎯 Career Readiness", f"{readiness}%",
                               delta=profile["mock_trend"])
            with c2: st.metric("🎓 Mock Test Avg",    f"{profile['mock_avg']:.1f}%",
                               delta=f"{profile['mock_tests']} tests")
            with c3: st.metric("🎤 Interview Avg",
                               f"{profile['interview_avg']:.1f}/10" if profile['interview_avg'] else "No data",
                               delta=f"{profile['interview_rounds']} rounds")
            with c4: st.metric("📚 Modules Mastered",  f"{profile['learning_modules']}",
                               delta=f"Avg {profile['learning_avg']:.0f}%")

            st.markdown("")
            stage = guidance.get("readiness_label", "Skill Development")
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:5px;margin-bottom:4px;">'
                f'<div style="background:{ring_color};width:{readiness}%;height:14px;border-radius:8px;"></div></div>'
                f'<p style="color:{ring_color};font-weight:700;font-size:0.9em;margin:0 0 20px 0;">'
                f'🏷️ Stage: {stage} — {readiness}% career readiness</p>',
                unsafe_allow_html=True,
            )

            next_milestone = guidance.get("next_milestone", "")
            if next_milestone:
                st.markdown(f"""
                <div class="milestone-box">
                    <h4>🎯 Next Milestone</h4>
                    <p>{next_milestone}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")

            col_s, col_f = st.columns(2)
            with col_s:
                st.markdown("##### ✅ What's Going Well")
                for s in guidance.get("strengths", []):
                    st.markdown(f'<span class="strength-pill">✓ {s}</span>', unsafe_allow_html=True)
                st.markdown("")
            with col_f:
                st.markdown("##### 🎯 Where to Focus")
                for f_item in guidance.get("focus_areas", []):
                    st.markdown(f"• {f_item}")

            motivation = guidance.get("motivation", "")
            if motivation:
                st.markdown("")
                st.info(f"💬 **Copilot says:** {motivation}")

            tip = guidance.get("job_readiness_tip", "")
            if tip:
                st.markdown(f"**🚀 Fastest path to job offers:** {tip}")

            st.divider()
            st.markdown("#### ⚡ Quick Actions")
            st.caption("Jump directly to the tool your Copilot recommends most:")
            qa1, qa2, qa3, qa4 = st.columns(4)
            with qa1:
                if st.button("📝 Take Mock Test",     use_container_width=True):
                    st.info("👉 Navigate to **🎓 Mock Assessment** in the sidebar.")
            with qa2:
                if st.button("🎤 Practice Interview", use_container_width=True):
                    st.info("👉 Navigate to **🎤 AI Interview Simulator** in the sidebar.")
            with qa3:
                if st.button("📚 Study a Topic",      use_container_width=True):
                    if profile["critical_gaps"]:
                        st.info(f"👉 Go to **🤖 AI Learning Agent** → study: **{profile['critical_gaps'][0]}**")
                    else:
                        st.info("👉 Navigate to **📚 Guided Study Chat** in the sidebar.")
            with qa4:
                if st.button("💼 Find Jobs",           use_container_width=True):
                    st.info("👉 Navigate to **💼 AI Job Finder** in the sidebar.")

        # ═══════════════ WEEKLY PLAN ════════════════════════
        with tab_plan:
            st.markdown("### 📅 Your AI-Generated Weekly Plan")
            st.caption(f"Personalised for **{name_cp}** targeting **{goal_cp}** at **{readiness}%** readiness.")
            st.markdown("")

            TYPE_ICONS = {
                "study":    ("📖", "#6366f1"),
                "practice": ("⚡", "#f59e0b"),
                "build":    ("🔨", "#22c55e"),
                "apply":    ("🚀", "#ec4899"),
            }

            for task in guidance.get("weekly_plan", []):
                t_type = task.get("type","study").lower()
                icon, _ = TYPE_ICONS.get(t_type, ("📌", "#3b82f6"))
                st.markdown(f"""
                <div class="task-card {t_type}">
                    <div class="task-day">{task.get('day','')}</div>
                    <div class="task-text">{icon} {task.get('task','')}</div>
                    <div class="task-why">💡 {task.get('why','')} &nbsp;·&nbsp;
                         <span class="task-dur">⏱ {task.get('duration','')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.divider()
            if st.button("🔄 Regenerate Weekly Plan", use_container_width=False):
                if not check_request_limit(): st.stop()
                with st.spinner("🤖 Generating a fresh weekly plan…"):
                    new_guidance = generate_copilot_guidance(profile)
                st.session_state.copilot_guidance = new_guidance
                st.success("✅ Weekly plan refreshed!")
                st.rerun()

        # ═══════════════ GAP ANALYSIS ═══════════════════════
        with tab_gaps:
            st.markdown("### 🔍 Integrated Gap Analysis")
            st.caption(f"Skill gaps for **{goal_cp}** combined with your performance data.")
            st.markdown("")

            critical_gaps  = profile.get("critical_gaps",  [])
            important_gaps = profile.get("important_gaps", [])

            if critical_gaps:
                st.markdown("#### 🔴 Critical Gaps — Address Immediately")
                st.markdown(
                    " ".join(f'<span class="gap-pill">⚠ {g}</span>' for g in critical_gaps),
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown("**🤖 Copilot recommends:**")
                for gap in critical_gaps:
                    with st.expander(f"How to close: {gap}"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**📚 Study it**")
                            st.caption(f"AI Learning Agent → topic: {gap}")
                        with col_b:
                            st.markdown("**📝 Test yourself**")
                            st.caption(f"Mock Assessment → add {gap} to skills")
                        with col_c:
                            st.markdown("**🎤 Interview prep**")
                            st.caption(f"Interview Simulator → technical round")

            if important_gaps:
                st.markdown("#### 🟡 Important Gaps — Plan This Month")
                st.markdown(
                    " ".join(
                        f'<span style="display:inline-block;background:rgba(245,158,11,0.15);'
                        f'border:1px solid rgba(245,158,11,0.3);border-radius:20px;'
                        f'padding:4px 12px;font-size:0.8em;color:#fcd34d;margin:3px;font-weight:500;">○ {g}</span>'
                        for g in important_gaps
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown("")

            if not critical_gaps and not important_gaps:
                st.success("✅ No critical or important gaps detected. Focus on advanced skills and interview prep.")

            st.divider()
            st.markdown("#### 📊 Performance vs Gap Coverage")

            metrics_data = {
                "Mock Test Readiness":  int(min(profile["mock_avg"],         100)),
                "Interview Readiness":  int(min(profile["interview_avg"]*10, 100)),
                "Learning Progress":    int(min(profile["learning_avg"],     100)),
                "Career Readiness":     readiness,
            }
            col_m1, col_m2 = st.columns(2)
            for idx, (label, val) in enumerate(metrics_data.items()):
                col = col_m1 if idx % 2 == 0 else col_m2
                bar_c = "#22c55e" if val >= 75 else "#f59e0b" if val >= 50 else "#ef4444"
                with col:
                    st.markdown(f"**{label}** — {val}%")
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:3px;margin-bottom:14px;">'
                        f'<div style="background:{bar_c};width:{val}%;height:8px;border-radius:6px;"></div></div>',
                        unsafe_allow_html=True,
                    )

            topics_studied = profile.get("learning_topics", [])
            if topics_studied:
                st.divider()
                st.markdown("#### 📚 Topics You've Already Studied")
                for t in topics_studied:
                    st.markdown(f"✅ {t}")

        # ═══════════════ CHAT ════════════════════════════════
        with tab_chat:
            st.markdown("### 💬 Ask Your Career Copilot")
            st.caption(
                "Ask anything about your career, skills, what to study, "
                "interview prep, or how to improve scores. "
                "The Copilot answers using **your actual data**."
            )
            st.markdown("")

            if "copilot_chat_msgs" not in st.session_state:
                st.session_state.copilot_chat_msgs = []

            if not st.session_state.copilot_chat_msgs:
                st.markdown("**💡 Suggested questions:**")
                sugg_cols = st.columns(2)
                suggestions = [
                    "What should I study this week?",
                    f"Am I ready to apply for {goal_cp} roles?",
                    "What's my biggest weakness right now?",
                    "How do I improve my mock test score?",
                ]
                for i, sugg in enumerate(suggestions):
                    col = sugg_cols[i % 2]
                    with col:
                        if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                            if not check_request_limit(): st.stop()
                            with st.spinner("🤖 Copilot is thinking…"):
                                resp = generate_copilot_chat_response(
                                    profile, sugg, st.session_state.copilot_chat_msgs
                                )
                            st.session_state.copilot_chat_msgs.append({"role":"user","content":sugg})
                            st.session_state.copilot_chat_msgs.append({"role":"assistant","content":resp})
                            st.rerun()

            for msg in st.session_state.copilot_chat_msgs:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_q = st.chat_input("Ask your Career Copilot anything…")
            if user_q:
                if not check_request_limit(): st.stop()
                st.session_state.copilot_chat_msgs.append({"role":"user","content":user_q})
                with st.spinner("🤖 Copilot is thinking…"):
                    resp = generate_copilot_chat_response(
                        profile, user_q, st.session_state.copilot_chat_msgs
                    )
                st.session_state.copilot_chat_msgs.append({"role":"assistant","content":resp})
                st.rerun()

            if st.session_state.copilot_chat_msgs:
                if st.button("🗑️ Clear Chat", use_container_width=False):
                    st.session_state.copilot_chat_msgs = []
                    st.rerun()

        # ═══════════════ RESET ══════════════════════════════
        with tab_reset:
            st.markdown("### 🔄 Refresh Your Copilot")
            st.info(
                "Your Copilot profile is saved. Click **Refresh** to re-run the full "
                "analysis and pick up new mock tests, interviews, or learning progress."
            )

            st.markdown("**Current Snapshot:**")
            snap = {
                "Name":             profile["name"],
                "Goal Role":        profile["goal_role"],
                "Career Readiness": f"{profile['readiness']}%",
                "Stage":            guidance.get("readiness_label","—"),
                "Critical Gaps":    ", ".join(profile["critical_gaps"]) or "None",
                "Mock Avg":         f"{profile['mock_avg']:.1f}%",
                "Interview Avg":    f"{profile['interview_avg']:.1f}/10",
                "Learning Modules": str(profile["learning_modules"]),
            }
            for k, v in snap.items():
                st.markdown(f"**{k}:** {v}")

            st.markdown("")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button("🔁 Refresh Copilot (Keep Profile)", use_container_width=True):
                    if not check_request_limit(): st.stop()
                    with st.spinner("🔄 Re-fetching all module data…"):
                        mock_hist_r      = analyze_user_trend(profile["name"]) or {}
                        interview_hist_r = _load_interview_history(profile["name"])
                        agent_hist_r     = _load_agent_history(profile["name"])
                        new_profile = build_career_profile(
                            name=profile["name"], goal_role=profile["goal_role"],
                            domain=profile["domain"], education=profile["education"],
                            skills=profile["skills"],
                            skill_gaps=detect_skill_gaps_cached(
                                tuple(profile["skills"]), profile["goal_role"], profile["domain"]
                            ),
                            mock_history=mock_hist_r,
                            interview_history=interview_hist_r,
                            agent_history=agent_hist_r,
                        )
                    with st.spinner("🤖 Updating your weekly plan…"):
                        new_guidance = generate_copilot_guidance(new_profile)
                    st.session_state.copilot_profile  = new_profile
                    st.session_state.copilot_guidance = new_guidance
                    st.success("✅ Copilot refreshed with latest data!")
                    st.rerun()
            with col_r2:
                if st.button("🗑️ Reset Copilot Completely", use_container_width=True):
                    for k in ["copilot_started","copilot_profile","copilot_guidance","copilot_chat_msgs"]:
                        st.session_state.pop(k, None)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  SKILL INTELLIGENCE  ██████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Skill Intelligence":

    st.session_state.current_feature = "Skill_Intelligence"

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        st.session_state.current_user = name.strip() if name else "Guest"
    with col2:
        education = st.text_input("Education Level",
            placeholder="e.g. 10th Grade, B.Tech, MBA, Bootcamp Graduate…")

    skills_input = st.text_input("Current Skills (comma-separated)")
    hours        = st.slider("Weekly Learning Hours", 1, 40, 10)

    if st.button("🔎 Analyze Skill Intelligence", use_container_width=True):

        if not check_request_limit(): st.stop()

        if not name.strip():
            st.warning("⚠️ Please enter your name."); st.stop()
        if not education.strip():
            st.warning("⚠️ Please enter your education level."); st.stop()

        skills = normalize_skills(skills_input)
        if not skills and skills_input.strip():
            st.warning("⚠️ No valid skills detected."); st.stop()
        if not skills:
            st.warning("⚠️ Please enter at least one skill."); st.stop()
        if len(skills) == MAX_SKILLS:
            st.info(f"ℹ️ Analysis based on the first {MAX_SKILLS} skills entered.")

        skills_tuple = tuple(skills)

        with st.spinner("🔍 Detecting career domain…"):        domain         = detect_domain_cached(skills_tuple) or "General Domain"
        with st.spinner("🎯 Inferring best-fit role…"):        role           = infer_role_cached(skills_tuple, domain) or "Specialist"
        with st.spinner("📈 Building growth plan…"):           growth         = generate_growth(role, domain, education) or []
        with st.spinner("🎓 Finding certifications…"):         certifications = generate_certifications(role, domain) or []
        with st.spinner("🌍 Analyzing market outlook…"):       market         = generate_market(role, domain, education) or "Market data unavailable."
        with st.spinner("💡 Evaluating career confidence…"):   confidence     = generate_confidence(role, domain, education)
        with st.spinner("🌐 Loading learning platforms…"):     platforms      = generate_platforms(role, domain, skills) or {"free":[],"paid":[]}
        with st.spinner("⏳ Estimating learning timeline…"):   weeks          = generate_timeline(role, domain, growth, hours)
        with st.spinner("🔍 Detecting skill gaps…"):           gaps           = detect_skill_gaps_cached(skills_tuple, role, domain)
        with st.spinner("🗓️ Generating week-by-week roadmap…"): roadmap       = generate_learning_roadmap_cached(role, domain, tuple(growth), hours, weeks) if growth else []

        confidence_value = 70; risk_value = "Medium"; summary_value = "Moderate job outlook."
        if isinstance(confidence, str):
            cm = re.search(r"(\d+)%",                    confidence)
            rm = re.search(r"Risk:\s*(Low|Medium|High)", confidence)
            sm = re.search(r"Summary:\s*(.*)",           confidence)
            if cm: confidence_value = int(cm.group(1))
            if rm: risk_value       = rm.group(1)
            if sm: summary_value    = sm.group(1)

        skill_score  = min(len(skills) * 10, 40)
        growth_score = max(0, 30 - (len(growth) * 5))
        market_score = round((min(confidence_value, 100) / 100) * 30)
        total_score  = skill_score + growth_score + market_score

        skill_msg  = "Strong skill foundation." if skill_score >= 30 else "Good start — add 2-3 more core skills."
        growth_msg = "Minimal skill gaps — close to market-ready." if growth_score >= 20 else "Several growth skills recommended — focus on top 2-3 first."
        market_msg = "Market demand is strong for this role." if market_score >= 22 else "Moderate demand — consider adjacent roles too."

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Role Alignment", "📈 Growth Plan", "🎓 Certifications", "🌍 Market Outlook", "🔍 Skill Gap",
        ])

        with tab1:
            st.header(role)
            st.markdown(f"🧭 **Domain:** `{domain}`  &nbsp;&nbsp; 🎓 **Education:** `{education}`")
            st.divider()
            st.markdown("### 🏆 Career Readiness Scorecard")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Overall Readiness", f"{total_score}/100")
            with c2: st.metric("Skill Strength",    f"{skill_score}/40")
            with c3: st.metric("Growth Gap",        f"{growth_score}/30")
            with c4: st.metric("Market Demand",     f"{market_score}/30")

            bar_color = "#22c55e" if total_score >= 75 else "#f59e0b" if total_score >= 50 else "#ef4444"
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:4px;margin:8px 0 16px 0;">
                    <div style="background:{bar_color};width:{total_score}%;height:12px;border-radius:8px;"></div>
                </div>""", unsafe_allow_html=True)

            if   total_score >= 75: st.success("✅ **Strong Profile** — You are well-positioned for this role.")
            elif total_score >= 50: st.warning("⚠️ **Developing Profile** — Targeted improvements will boost your chances.")
            else:                   st.error("❌ **Early Stage** — Build core skills before applying.")

            st.divider()
            st.markdown("### 📌 Score Breakdown")
            ca, cb, cc = st.columns(3)
            with ca: st.markdown("**🔵 Skill Strength**"); st.caption(skill_msg)
            with cb: st.markdown("**🟡 Growth Gap**");     st.caption(growth_msg)
            with cc: st.markdown("**🟢 Market Demand**");  st.caption(market_msg)

            st.divider()
            st.markdown("### 📊 Market Signal")
            cx, cy = st.columns(2)
            with cx: st.metric("Hiring Confidence", f"{confidence_value}%")
            with cy:
                risk_color = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk_value,"🟡")
                st.metric("Market Risk", f"{risk_color} {risk_value}")
            st.markdown(f"_{summary_value}_")

            st.divider()
            st.markdown("### 🚀 Recommended Next Steps")
            if total_score >= 75:
                st.info("1. **Start applying** — your profile is competitive for entry to mid-level roles.")
                st.info("2. **Pick 1-2 certifications** from the Certifications tab.")
                st.info("3. **Build a portfolio project** using your top 3 skills.")
            elif total_score >= 50:
                st.info("1. **Focus on growth skills** — pick the top 2 from the Growth Plan tab.")
                st.info("2. **Get one certification** to validate skills to recruiters.")
                st.info("3. **Apply to junior/associate roles** while continuing to learn.")
            else:
                st.info("1. **Start with fundamentals** — use Guided Study Chat.")
                st.info("2. **Add 2-3 more skills** relevant to your target domain.")
                st.info("3. **Revisit this analysis** after 4-6 weeks of focused learning.")

        with tab2:
            st.markdown("### 📈 Recommended Growth Skills")
            st.caption(f"Based on **{role}** in **{domain}** with **{education}** background")
            if growth:
                for idx, skill in enumerate(growth, 1):
                    st.markdown(f"**{idx}.** {skill}")
                st.divider()
                st.markdown("### ⏳ Estimated Learning Timeline")
                st.markdown(f"At **{hours} hours/week**, you can cover these skills in approximately:")
                st.metric("Estimated Timeline", f"~{weeks} weeks")
                st.divider()
                st.markdown("### 🗓️ Week-by-Week Learning Roadmap")
                if roadmap:
                    for w in roadmap:
                        with st.expander(f"📅 Week {w.get('week','')} — {w.get('focus','')}"):
                            if w.get("topics"):
                                st.markdown("**Topics to cover:**")
                                for t in w["topics"]: st.markdown(f"  - {t}")
                            if w.get("resource"):   st.markdown(f"**Recommended resource:** {w['resource']}")
                            if w.get("milestone"):  st.markdown(f"**✅ Milestone:** {w['milestone']}")
                else:
                    st.info("Roadmap unavailable — try again or check your skills input.")
            else:
                st.info("No growth skill recommendations available.")

        with tab3:
            st.markdown("### 🎓 Recommended Certifications")
            if certifications:
                for cert in certifications: st.markdown(f"- {cert}")
            else:
                st.info("No certifications available.")
            st.divider()
            st.markdown("### 🌐 Learning Platforms")
            cf, cp = st.columns(2)
            with cf:
                st.markdown("#### 🆓 Free Platforms")
                for item in platforms.get("free", []): st.markdown(f"- [{item['name']}]({item['url']})")
                if not platforms.get("free"): st.caption("None listed.")
            with cp:
                st.markdown("#### 💼 Paid / Industry Recognized")
                for item in platforms.get("paid", []): st.markdown(f"- [{item['name']}]({item['url']})")
                if not platforms.get("paid"): st.caption("None listed.")

        with tab4:
            st.markdown("### 🌍 Market Outlook")
            st.caption(f"Demand analysis for **{role}** in **{domain}** — **{education}** level")
            st.markdown(market)

        with tab5:
            st.markdown("### 🔍 Skill Gap Analysis")
            st.caption(f"Your skills vs requirements for **{role}** in **{domain}**")
            if gaps:
                have    = [g for g in gaps if g.get("status") == "Have"]
                missing = [g for g in gaps if g.get("status") == "Missing"]
                partial = [g for g in gaps if g.get("status") == "Partial"]
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("✅ Skills You Have",  len(have))
                with c2: st.metric("❌ Missing Skills",   len(missing))
                with c3: st.metric("⚠️ Partial Skills",   len(partial))
                st.divider()

                critical  = [g for g in gaps if g.get("priority") == "Critical"     and g.get("status") != "Have"]
                important = [g for g in gaps if g.get("priority") == "Important"    and g.get("status") != "Have"]
                nice      = [g for g in gaps if g.get("priority") == "Nice to Have" and g.get("status") != "Have"]

                if critical:
                    st.markdown("#### 🔴 Critical Gaps (Must Learn)")
                    for g in critical: st.error(f"**{g['skill']}** — {g.get('reason','')}")
                if important:
                    st.markdown("#### 🟡 Important Gaps")
                    for g in important: st.warning(f"**{g['skill']}** — {g.get('reason','')}")
                if nice:
                    st.markdown("#### 🟢 Nice to Have")
                    for g in nice: st.info(f"**{g['skill']}** — {g.get('reason','')}")
                if have:
                    st.divider()
                    st.markdown("#### ✅ Skills You Already Have")
                    for g in have: st.success(f"**{g['skill']}** — {g.get('reason','')}")

                if gaps:
                    have_pct = round(len(have) / len(gaps) * 100)
                    st.divider()
                    st.markdown(f"**Role Coverage: {have_pct}% of required skills matched**")
                    bar_col = "#22c55e" if have_pct >= 70 else "#f59e0b" if have_pct >= 40 else "#ef4444"
                    st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:4px;">
                            <div style="background:{bar_col};width:{have_pct}%;height:10px;border-radius:8px;"></div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("Skill gap data unavailable. Try refreshing the analysis.")

        st.divider()
        rating        = st.slider("How useful was this analysis?", 1, 5, 4)
        feedback_text = st.text_area("What can we improve?")
        if st.button("Submit Feedback"):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("feedback_log.txt", "a") as f:
                f.write(f"{ts} | {name} | {rating} | {education} | {skills_input} | {feedback_text}\n")
            save_feedback([ts, name, rating, education, skills_input, feedback_text])
            st.success("✅ Feedback saved. Thank you!")


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  MOCK ASSESSMENT  █████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎓 Mock Assessment":

    st.header("🎓 Skill-Based Mock Assessment")

    candidate_name = st.text_input("Full Name")
    st.session_state.current_user    = candidate_name or "Guest"
    st.session_state.current_feature = "Mock_Assessment"

    if candidate_name:
        perf = analyze_user_trend(candidate_name)
        if perf:
            msg = generate_mentor_response(candidate_name, perf)
            st.success(msg)
            st.info(f"🎯 Recommended Difficulty: {recommend_difficulty(perf)}")
        else:
            st.info(f"👋 Welcome {candidate_name}! Let's build your career momentum 🚀")

    candidate_email     = st.text_input("Email")
    candidate_education = st.selectbox("Education Level",
        ["High School","Diploma","Graduation","Post Graduation","Other"])
    skills_input        = st.text_input("Skills (comma-separated)")
    difficulty          = st.selectbox("Select Difficulty", ["Beginner","Intermediate","Expert"])
    test_mode           = st.selectbox("Select Test Mode",
        ["Theoretical Knowledge","Logical Thinking","Coding Based"])

    if test_mode == "Coding Based":
        st.info("💡 **Coding Based** = 50% MCQs + 50% Written questions evaluated by AI.")

    mcq_count = st.select_slider("📝 Number of Questions", options=[5,10,15,20], value=10)

    tpq = {"Beginner":12,"Intermediate":24,"Expert":36}
    if test_mode == "Coding Based":
        half    = mcq_count // 2
        est_sec = (half * tpq[difficulty]) + ((mcq_count - half) * tpq[difficulty] * 3)
    else:
        est_sec = tpq[difficulty] * mcq_count
    st.caption(f"⏱️ Estimated time: ~{round(est_sec/60,1)} min")

    if "mock_questions" not in st.session_state:
        st.session_state.mock_questions = []

    if st.button("Generate Test"):
        if not check_request_limit(): st.stop()
        skills = normalize_skills(skills_input)
        if not skills and skills_input.strip():
            st.warning("⚠️ No valid skills detected."); st.stop()
        if not skills:
            st.warning("⚠️ Please enter at least one skill."); st.stop()

        stuple = tuple(skills)

        if test_mode == "Coding Based":
            half = mcq_count // 2; written_half = mcq_count - half
            with st.spinner(f"Generating {half} coding MCQs…"):
                mcq_qs = cached_generate_mcqs(stuple, difficulty, "Coding Based", half)
            with st.spinner(f"Generating {written_half} written questions…"):
                wr_qs  = cached_generate_written_questions(stuple, difficulty, written_half)

            if mcq_qs:
                for q in mcq_qs: q["type"] = "mcq"

            combined = []
            wi = mi = 0
            wl = wr_qs or []; ml = mcq_qs or []
            while wi < len(wl) or mi < len(ml):
                if wi < len(wl): combined.append(wl[wi]); wi += 1
                if mi < len(ml): combined.append(ml[mi]); mi += 1

            if combined:
                st.session_state.mock_questions      = combined
                st.session_state.written_evaluations = {}
            else:
                st.error("Failed to generate coding questions. Try again."); st.stop()
        else:
            with st.spinner("Generating questions…"):
                questions = cached_generate_mcqs(stuple, difficulty, test_mode, mcq_count)
            if questions and isinstance(questions, list):
                for q in questions: q["type"] = "mcq"
                st.session_state.mock_questions = questions
            else:
                st.error("Failed to generate test questions. Try again."); st.stop()

        st.session_state.start_time          = time.time()
        st.session_state.time_limit          = get_time_limit(difficulty, mcq_count, test_mode)
        st.session_state.exam_submitted      = False
        st.session_state.explanations        = {}
        st.session_state.written_evaluations = {}
        st.session_state.result_saved        = False
        st.session_state.final_score         = 0
        st.session_state.final_percent       = 0
        st.session_state.mcq_percent         = 0
        st.session_state.written_percent     = 0
        st.session_state.mcq_total           = 0
        st.session_state.written_total       = 0
        st.session_state.written_score_total = 0

    if st.session_state.get("mock_questions"):

        auto_submit     = False
        total_questions = len(st.session_state.mock_questions)

        if "start_time" in st.session_state:
            elapsed        = int(time.time() - st.session_state.start_time)
            remaining_time = st.session_state.time_limit - elapsed
            if remaining_time <= 0:
                auto_submit = True
                st.warning("⏰ Time is up! Auto-submitting…")
            else:
                m, s = divmod(remaining_time, 60)
                st.markdown(f"### ⏳ Time Remaining: {m:02d}:{s:02d}")

        submit_clicked = st.button("Submit Test")

        if (submit_clicked or auto_submit) and not st.session_state.get("exam_submitted"):
            st.session_state.exam_submitted = True

            mcq_score = mcq_total = written_score_total = written_total = 0

            for i, q in enumerate(st.session_state.mock_questions):
                qtype = q.get("type","mcq")

                if qtype == "mcq":
                    mcq_total += 1
                    selected    = st.session_state.get(f"mock_{i}")
                    correct_ans = q.get("answer")
                    correct_opt = None
                    if isinstance(correct_ans, int):
                        if 0 <= correct_ans < len(q["options"]): correct_opt = q["options"][correct_ans]
                    elif isinstance(correct_ans, str):
                        correct_ans = correct_ans.strip()
                        if correct_ans.isdigit():
                            idx = int(correct_ans)
                            if 0 <= idx < len(q["options"]): correct_opt = q["options"][idx]
                        elif correct_ans in q["options"]: correct_opt = correct_ans
                        elif correct_ans in "ABCD":
                            idx = ord(correct_ans) - ord("A")
                            if idx < len(q["options"]): correct_opt = q["options"][idx]
                    if selected and correct_opt and selected.strip().lower() == correct_opt.strip().lower():
                        mcq_score += 1
                    st.session_state.final_score = mcq_score
                    st.session_state.mcq_total   = mcq_total

                elif qtype == "written":
                    written_total += 1
                    user_ans = st.session_state.get(f"written_{i}", "")
                    try:
                        with st.spinner(f"🤖 AI evaluating written Q{i+1}…"):
                            ev = evaluate_written_answer(q["question"], user_ans, difficulty)
                    except Exception as e:
                        ev = {"score":0,"feedback":f"Evaluation error: {e}","model_answer":"N/A"}
                    st.session_state.written_evaluations[i] = ev
                    written_score_total += ev.get("score", 0)
                    st.session_state.written_total       = written_total
                    st.session_state.written_score_total = written_score_total

            mcq_pct     = (mcq_score / mcq_total * 100)                       if mcq_total     > 0 else 0
            written_pct = (written_score_total / (written_total * 10) * 100)  if written_total > 0 else 0
            if mcq_total > 0 and written_total > 0:
                overall_pct = (mcq_pct + written_pct) / 2
            elif mcq_total > 0:
                overall_pct = mcq_pct
            else:
                overall_pct = written_pct

            st.session_state.final_percent       = overall_pct
            st.session_state.mcq_percent         = mcq_pct
            st.session_state.written_percent     = written_pct
            st.session_state.pop("start_time", None)
            st.session_state.pop("time_limit",  None)

        for i, q in enumerate(st.session_state.mock_questions):
            qtype = q.get("type","mcq")

            if qtype == "written":
                st.markdown(f"### ✍️ Q{i+1}. {q['question']}")
                if q.get("hints"): st.caption(f"💡 Hint: {q['hints']}")
                if not st.session_state.get("exam_submitted"):
                    st.text_area("Write your code / answer here:", key=f"written_{i}", height=200,
                                 placeholder="# Write your solution here…\ndef solution():\n    pass")
                else:
                    ans = st.session_state.get(f"written_{i}","")
                    st.code(ans or "No answer provided.", language="python")
                    ev  = st.session_state.get("written_evaluations",{}).get(i)
                    if ev:
                        sv = ev.get("score",0)
                        (st.success if sv>=8 else st.warning if sv>=5 else st.error)(
                            f"{'✅' if sv>=8 else '⚠️' if sv>=5 else '❌'} AI Score: {sv}/10"
                        )
                        st.markdown("📘 **Feedback:**"); st.info(ev.get("feedback",""))
                        st.markdown("📗 **Model Answer:**"); st.code(ev.get("model_answer","N/A"), language="python")
            else:
                st.markdown(f"### 🔘 Q{i+1}. {q['question']}")
                st.radio("", q["options"], index=None, key=f"mock_{i}",
                         disabled=st.session_state.get("exam_submitted", False))

                if st.session_state.get("exam_submitted"):
                    correct_ans = q.get("answer"); correct_opt = None
                    if isinstance(correct_ans, int):
                        if correct_ans < len(q["options"]): correct_opt = q["options"][correct_ans]
                    elif isinstance(correct_ans, str):
                        correct_ans = correct_ans.strip()
                        if correct_ans.isdigit():
                            idx = int(correct_ans)
                            if 0 <= idx < len(q["options"]): correct_opt = q["options"][idx]
                        elif correct_ans in q["options"]: correct_opt = correct_ans
                        elif correct_ans in "ABCD":
                            idx = ord(correct_ans) - ord("A")
                            if idx < len(q["options"]): correct_opt = q["options"][idx]

                    sel = st.session_state.get(f"mock_{i}")
                    if sel == correct_opt: st.success(f"✅ Correct: {correct_opt}")
                    else:
                        st.error(f"❌ Your Answer: {sel}")
                        st.info(f"✔ Correct: {correct_opt}")

                    if i not in st.session_state.explanations:
                        with st.spinner("📘 Generating explanation…"):
                            st.session_state.explanations[i] = generate_explanation(q["question"], correct_opt) or "Unavailable."
                    st.markdown("📘 **Explanation:**")
                    st.info(st.session_state.explanations.get(i,"Unavailable."))

            st.divider()

        if st.session_state.get("exam_submitted"):
            st.markdown("## 📊 Test Result")
            mt = st.session_state.get("mcq_total",0)
            wt = st.session_state.get("written_total",0)

            if mt > 0 and wt > 0:
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("🔘 MCQ Score",    f"{st.session_state.get('final_score',0)}/{mt}", f"{st.session_state.get('mcq_percent',0):.1f}%")
                with c2: st.metric("✍️ Written Score", f"{st.session_state.get('written_score_total',0)}/{wt*10} pts", f"{st.session_state.get('written_percent',0):.1f}%")
                with c3: st.metric("🏆 Overall Score", f"{st.session_state.get('final_percent',0):.1f}%")
            else:
                st.markdown(f"### Score: {st.session_state.get('final_score',0)}/{total_questions}")
                st.markdown(f"### Percentage: {st.session_state.get('final_percent',0):.2f}%")

            fp = st.session_state.get("final_percent",0)
            if   fp >= 80: st.success("✅ Qualified (80%+)")
            elif fp >= 60: st.warning("⚠️ Average (60-79%) — Keep practising!")
            else:          st.error("❌ Not Qualified (<60%) — Review fundamentals.")

            if not st.session_state.get("result_saved"):
                save_mock_result([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    candidate_name, candidate_email, candidate_education,
                    skills_input, difficulty, test_mode,
                    st.session_state.get("final_score",0),
                    round(st.session_state.get("final_percent",0),2),
                    total_questions, mt, wt,
                ])
                st.session_state.result_saved = True


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  GUIDED STUDY CHAT  ███████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 Guided Study Chat":

    st.header("📚 AI Guided Study Chat")
    st.session_state.current_feature = "Guided_Study_Chat"

    candidate_name = st.text_input("Full Name")
    st.session_state.current_user = candidate_name or "Guest"

    if candidate_name:
        perf = analyze_user_trend(candidate_name)
        if perf:
            st.success(generate_mentor_response(candidate_name, perf))
        else:
            st.info(f"👋 Hi {candidate_name}! Let's build your expertise 🚀")

    education   = st.text_input("Education Level", placeholder="e.g. 10th Grade, B.Tech, Self-taught, MBA…")
    topic       = st.text_input("Topic You Want To Study")
    book_source = st.text_input("📖 Reference Book / Source (Optional)",
                    placeholder="e.g. NCERT, RD Sharma, HC Verma, CBSE…")
    if book_source:
        st.caption(f"📌 Tutor will follow the structure and style of: **{book_source}**")

    level        = st.selectbox("Skill Level", ["Beginner","Intermediate","Expert"])
    learning_goal = st.text_input("Learning Goal")

    if "study_chat_started" not in st.session_state:
        st.session_state.study_chat_started = False
    if "study_messages" not in st.session_state:
        st.session_state.study_messages = []

    if st.button("Start Learning"):
        if topic and candidate_name and education:
            already_studied = check_study_history(candidate_name, education, topic, level)

            if already_studied:
                st.warning(f"You have already studied **{topic} ({level})**.")
                action = st.selectbox("What would you like to do?",
                    ["Revise","Study in Detailed Mode","Test Yourself"])
                if action == "Test Yourself":
                    qs = generate_mcqs([topic], level, "Theoretical Knowledge")
                    if qs: st.session_state.quick_test = qs[:5]
                elif action == "Revise":
                    st.session_state.study_chat_started = True
                elif action == "Study in Detailed Mode":
                    level = "Expert"
                    st.session_state.study_chat_started = True
            else:
                save_study_history([
                    candidate_name, education, topic, level,
                    book_source or "General",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ])
                st.session_state.study_chat_started = True

            book_line = (
                f"Book/Source: {book_source}. Follow its structure, sequence, terminology, "
                f"and teaching style exactly. Stay within what it covers for this topic."
                if book_source else "Use standard curriculum knowledge."
            )
            edu_line = (
                "Simple language, real-world analogies, examples before theory."
                if any(x in education.lower() for x in ["10","12","school","high"])
                else "Mix theory and practice, standard terminology with brief explanations."
                if any(x in education.lower() for x in ["diploma","under","btech","b.tech","bsc","bootcamp"])
                else "Precise technical language, skip basics, focus on depth and edge cases."
            )

            st.session_state.study_context = (
                f"Expert tutor. Teach ONLY: {topic}. "
                f"Level: {level}. Student background: {education}. "
                f"Goal: {learning_goal or 'General understanding'}.\n"
                f"{book_line}\n"
                f"Language style: {edu_line}\n"
                "Rules: structured headings, examples where helpful, "
                "end every reply with one follow-up question or next concept. "
                f"If asked off-topic, redirect back to {topic}."
            )
            st.session_state.study_messages = []
            st.session_state.study_topic    = topic
            reset_study_memory()
        else:
            if not candidate_name: st.warning("⚠️ Please enter your name.")
            elif not education:    st.warning("⚠️ Please enter your education level.")
            elif not topic:        st.warning("⚠️ Please enter the topic.")

    if st.session_state.study_chat_started:
        st.markdown("---")
        st.subheader(f"📘 {topic}" + (f"  |  📖 {book_source}" if book_source else ""))

        if VECTOR_MEMORY_AVAILABLE:
            backend = "ChromaDB (persistent)" if CHROMA_AVAILABLE else "FAISS (session only)"
            st.caption(f"🧠 Vector memory active ({backend}) — past context referenced automatically.")

        user_input = st.chat_input("Ask your question about this topic…")

        if user_input:
            if not check_request_limit(): st.stop()
            st.session_state.study_messages.append({"role":"user","content":user_input})

            system_content = st.session_state.study_context
            if VECTOR_MEMORY_AVAILABLE:
                memory_ctx = retrieve_memory(user_input)
                if memory_ctx:
                    system_content += f"\n\n[Relevant context from this session:\n{memory_ctx}]"

            messages = [{"role":"system","content":system_content}] + st.session_state.study_messages
            response = safe_llm_call(MAIN_MODEL, messages, temperature=0.4)
            st.session_state.study_messages.append({"role":"assistant","content":response})

            if VECTOR_MEMORY_AVAILABLE:
                add_to_memory(user_input, response or "")

        for msg in st.session_state.study_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if "quick_test" in st.session_state:
        st.markdown("### 📝 Quick Self-Test")
        score = 0
        for i, q in enumerate(st.session_state.quick_test):
            sel = st.radio(q["question"], q["options"], key=f"quick_{i}")
            if sel == q["options"][q["answer"]]: score += 1
        if st.button("Submit Quick Test"):
            pct = score / len(st.session_state.quick_test) * 100
            (st.success if pct >= 80 else st.info)(
                "🔥 Excellent! Ready for the next level!" if pct >= 80
                else "Revise this level once more before upgrading."
            )
            del st.session_state.quick_test


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  AI JOB FINDER  ██████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💼 AI Job Finder (Premium)":

    st.header("💼 AI Career Job Finder")
    st.success("🎯 AI-Powered Smart Job Matching Activated")
    st.session_state.current_feature = "Job_Finder"

    name = st.text_input("Full Name")
    st.session_state.current_user = name or "Guest"

    age           = st.number_input("Age", min_value=16, max_value=65, step=1)
    education     = st.text_input("Education Level", placeholder="e.g. B.Tech, MBA, Bootcamp Graduate…")
    skills_input  = st.text_input("Skills (comma-separated)")
    experience    = st.selectbox("Years of Experience", ["Fresher","1-2 Years","3-5 Years","5+ Years"])
    current_field = st.text_input("Current Field / Industry")
    target_role   = st.text_input("Target Job Profile You Are Looking For")

    resume_file = st.file_uploader("Upload Resume (PDF, JPG, PNG supported)", type=["pdf","png","jpg","jpeg"])

    st.divider()
    st.markdown("### 📋 Paste a Job Description (Optional — for Match Score)")
    jd_text = st.text_area(
        "Paste the full job description here", height=160,
        placeholder="Paste the job description you want to apply for…",
        key="jd_text_input",
    )
    if jd_text:
        st.caption("✅ Job description loaded — match score will be computed after analysis.")

    st.divider()
    if st.button("🔍 Analyze & Find Jobs", use_container_width=True):
        if name and skills_input and target_role:
            if not check_request_limit(): st.stop()
            skills = normalize_skills(skills_input)
            if not skills and skills_input.strip():
                st.warning("⚠️ No valid skills detected."); st.stop()

            with st.spinner("Analyzing profile and matching jobs…"):
                resume_text = ""

                if resume_file is not None:
                    if resume_file.type == "application/pdf":
                        try:
                            pdf_reader  = PyPDF2.PdfReader(resume_file)
                            pages       = [p.extract_text() for p in pdf_reader.pages]
                            resume_text = "\n".join(p for p in pages if p)
                            if not resume_text.strip():
                                resume_text = "PDF uploaded but no text extracted."
                            else:
                                st.success(f"✅ Extracted {len(resume_text.split())} words from PDF.")
                        except Exception as e:
                            resume_text = "PDF extraction failed."; st.warning(f"⚠️ {e}")
                    elif resume_file.type in ["image/png","image/jpeg"]:
                        image = Image.open(resume_file)
                        st.image(image, caption="Uploaded Resume", use_column_width=True)
                        if OCR_AVAILABLE:
                            try:
                                resume_text = pytesseract.image_to_string(image).strip()
                                if not resume_text:
                                    resume_text = "No readable text found."
                                    st.warning("⚠️ No text detected. Try a PDF for best results.")
                                else:
                                    st.success(f"✅ Extracted {len(resume_text.split())} words via OCR.")
                            except Exception as e:
                                resume_text = "OCR failed."; st.warning(f"⚠️ {e}")
                        else:
                            resume_text = "Image uploaded (OCR unavailable)."
                            st.info("💡 Install pytesseract + tesseract-ocr for image text extraction.")

                extracted_resume_skills = []
                if resume_text and len(resume_text.strip()) > 50:
                    with st.spinner("🤖 Extracting skills from your resume…"):
                        extracted_resume_skills = extract_skills_from_resume(resume_text)
                    if extracted_resume_skills:
                        st.success(f"🎯 **{len(extracted_resume_skills)} skills detected in resume:**")
                        st.write(", ".join(s.title() for s in extracted_resume_skills))
                        user_set = set(skills); ext_set = set(s.lower() for s in extracted_resume_skills)
                        new_only = ext_set - user_set
                        if new_only:
                            st.info(f"➕ **{len(new_only)} additional skills found in resume:** "
                                    f"{', '.join(s.title() for s in sorted(new_only))}")
                        skills = list(user_set | ext_set)

                prompt = f"""
You are an AI Career Placement Advisor.
Candidate Profile:
Name: {name} | Age: {age} | Education: {education}
Skills: {", ".join(skills)} | Experience: {experience}
Current Field: {current_field} | Target Role: {target_role}
Resume Content: {resume_text[:2000] if resume_text else "Not provided"}

Analyze compatibility between the candidate and the target role. Provide:
1. Job Fit Score (0-100%)
2. Strengths
3. Skill Gaps
4. Resume Improvements
5. 3 Alternative Roles
6. Suggested Industries
7. Recommended Job Search Keywords
"""
                response = safe_llm_call("llama-3.3-70b-versatile",
                    [{"role":"user","content":prompt}], temperature=0.4)

            st.markdown("## 🎯 AI Career Recommendations")
            st.markdown(response)

            if jd_text and jd_text.strip():
                st.divider()
                st.markdown("## 🎯 Job Description Match Score")
                with st.spinner("🔢 Computing skill match score…"):
                    match = compute_job_match_score(skills, jd_text)

                if match:
                    overall = match["overall_score"]
                    bar_col = ("#22c55e" if overall >= 75 else "#f59e0b" if overall >= 55 else
                               "#f97316" if overall >= 35 else "#ef4444")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("🎯 Overall Match",  f"{overall}%")
                    with c2: st.metric("🧠 Semantic Score", f"{match['semantic_score']}%")
                    with c3: st.metric("🔑 Keyword Score",  f"{match['keyword_score']}%")

                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:4px;margin:8px 0 12px 0;">
  <div style="background:{bar_col};width:{overall}%;height:14px;border-radius:8px;"></div>
</div>""", unsafe_allow_html=True)

                    st.markdown(f"**{match['label']}** — {overall}% alignment with this job.")

                    col_m, col_miss = st.columns(2)
                    with col_m:
                        st.markdown("#### ✅ Matched Keywords")
                        if match["matched_keywords"]: st.success(", ".join(match["matched_keywords"][:15]))
                        else: st.caption("No keyword matches found.")
                    with col_miss:
                        st.markdown("#### ❌ Keywords You're Missing")
                        if match["missing_keywords"]: st.error(", ".join(match["missing_keywords"][:15]))
                        else: st.caption("No major gaps detected.")

                    save_job_match([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        name, target_role, overall,
                        match["semantic_score"], match["keyword_score"],
                        len(match["matched_keywords"]), len(match["missing_keywords"]),
                    ])

            st.divider()
            with st.spinner("🔍 Fetching live job listings…"):
                live_jobs = fetch_real_jobs(target_role, location="India", num=6)

            if live_jobs:
                render_job_cards(live_jobs)
                st.caption("📡 Live results via SerpAPI Google Jobs")
            else:
                enc = target_role.replace(" ", "%20")
                st.markdown("### 🔗 Search Jobs Directly")
                if not SERPAPI_KEY:
                    st.info("💡 Set `SERPAPI_KEY` env variable for live job cards.")
                st.markdown(f"🔹 [LinkedIn Jobs](https://www.linkedin.com/jobs/search/?keywords={enc})")
                st.markdown(f"🔹 [Indeed Jobs](https://www.indeed.com/jobs?q={enc})")
                st.markdown(f"🔹 [Naukri Jobs](https://www.naukri.com/{target_role.replace(' ','-')}-jobs)")
                st.markdown(f"🔹 [Glassdoor Jobs](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={enc})")
        else:
            st.warning("Please fill required fields (Name, Skills, Target Role).")


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  AI INTERVIEW SIMULATOR  ████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎤 AI Interview Simulator":

    st.header("🎤 AI Interview Simulator")
    st.caption("Full mock interview — 5 rounds, real-time AI feedback, scored debrief.")
    st.session_state.current_feature = "Interview_Simulator"

    if not st.session_state.get("interview_started"):

        col1, col2 = st.columns(2)
        with col1:
            iv_name = st.text_input("Your Full Name", key="iv_name_input")
            iv_role = st.text_input("Target Role", placeholder="e.g. Data Scientist, Backend Engineer")
        with col2:
            iv_edu  = st.text_input("Education Level", placeholder="e.g. B.Tech, MBA, Self-taught")
            iv_exp  = st.selectbox("Experience Level", ["Fresher","1-2 Years","3-5 Years","5+ Years"])

        iv_skills = st.text_input("Your Skills (comma-separated)")
        iv_diff   = st.selectbox("Interview Difficulty", ["Beginner","Intermediate","Expert"])

        rounds_available = [r["label"] for r in INTERVIEW_ROUNDS]
        iv_rounds = st.multiselect("Select Interview Rounds", rounds_available,
                        default=rounds_available[:3], help="Pick 1-5 rounds.")

        if st.button("🎤 Start Interview", use_container_width=True):
            if not iv_name or not iv_role or not iv_skills:
                st.warning("⚠️ Please fill in Name, Target Role, and Skills.")
            elif not iv_rounds:
                st.warning("⚠️ Please select at least one interview round.")
            else:
                if not check_request_limit(): st.stop()
                skills_clean  = normalize_skills(iv_skills)
                domain_iv     = detect_domain_cached(tuple(skills_clean)) or "Technology"
                selected_rounds = [r for r in INTERVIEW_ROUNDS if r["label"] in iv_rounds]

                st.session_state.interview_started   = True
                st.session_state.iv_name             = iv_name
                st.session_state.iv_role             = iv_role
                st.session_state.iv_domain           = domain_iv
                st.session_state.iv_difficulty       = iv_diff
                st.session_state.iv_skills           = skills_clean
                st.session_state.iv_rounds           = selected_rounds
                st.session_state.interview_round     = 0
                st.session_state.interview_messages  = []
                st.session_state.interview_score_log = []
                st.session_state.interview_complete  = False
                st.session_state.interview_q_count   = 0
                st.session_state.current_user        = iv_name
                st.rerun()

    else:
        iv_name   = st.session_state.iv_name
        iv_role   = st.session_state.iv_role
        iv_domain = st.session_state.iv_domain
        iv_diff   = st.session_state.iv_difficulty
        iv_skills = st.session_state.iv_skills
        rounds    = st.session_state.iv_rounds
        round_idx = st.session_state.interview_round
        messages  = st.session_state.interview_messages
        score_log = st.session_state.interview_score_log
        complete  = st.session_state.interview_complete

        total_rounds = len(rounds)
        if not complete:
            st.markdown(f"**Round {round_idx + 1} of {total_rounds} — {rounds[round_idx]['label']}**")
            st.progress(round_idx / total_rounds)
        else:
            st.progress(1.0)

        if score_log:
            avg_live = round(sum(s["score"] for s in score_log) / len(score_log), 1)
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Live Score")
            st.sidebar.metric("Current Avg", f"{avg_live}/10")
            st.sidebar.caption(f"Based on {len(score_log)} answer(s) scored")

        if not complete and not messages:
            with st.spinner(f"🎤 Preparing {rounds[round_idx]['label']}…"):
                opening = generate_interview_opening(
                    iv_role, iv_domain, iv_diff, rounds[round_idx], iv_skills
                )
            messages.append({"role":"assistant","content":opening,"round":round_idx})
            st.session_state.interview_messages = messages
            st.session_state.interview_q_count  = 1

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("score") is not None:
                    sc = msg["score"]
                    color = "#22c55e" if sc >= 7 else "#f59e0b" if sc >= 5 else "#ef4444"
                    st.markdown(f'<span style="color:{color};font-weight:700;">Score: {sc}/10</span>',
                                unsafe_allow_html=True)

        if not complete:
            col_input, col_next = st.columns([5, 1])
            with col_input:
                user_answer = st.chat_input("Type your answer here… (or type 'skip' to move on)")
            with col_next:
                next_round_btn = st.button("Next Round ➡️", disabled=(round_idx >= total_rounds - 1))

            if user_answer:
                if not check_request_limit(): st.stop()
                messages.append({"role":"user","content":user_answer,"round":round_idx})
                skip = user_answer.strip().lower() == "skip"

                if not skip and messages:
                    last_q = next((m["content"] for m in reversed(messages[:-1]) if m["role"] == "assistant"), "General question")
                    with st.spinner("🤖 Evaluating your answer…"):
                        ev = evaluate_interview_answer(last_q, user_answer, iv_role, iv_diff)

                    score     = ev.get("score", 0)
                    feedback  = ev.get("feedback", "")
                    follow_up = ev.get("follow_up", "Let's continue.")

                    score_log.append({
                        "round": rounds[round_idx]["label"],
                        "question": last_q[:120], "answer": user_answer[:120], "score": score,
                    })
                    st.session_state.interview_score_log = score_log

                    q_count = st.session_state.get("interview_q_count", 1)
                    MAX_Q_PER_ROUND = 3

                    if q_count >= MAX_Q_PER_ROUND:
                        reply = (f"**Feedback:** {feedback}\n\n**Score: {score}/10**\n\n"
                                 f"✅ Good work on this round! Click **Next Round** to continue.")
                    else:
                        reply = (f"**Feedback:** {feedback}\n\n**Score: {score}/10**\n\n"
                                 f"**Follow-up:** {follow_up}")
                        st.session_state.interview_q_count = q_count + 1

                    messages.append({"role":"assistant","content":reply,"round":round_idx,"score":score})
                elif skip:
                    messages.append({"role":"assistant",
                                     "content":"No problem — let's move on. " + rounds[round_idx]["focus"].capitalize() + " continues…",
                                     "round":round_idx})

                st.session_state.interview_messages = messages
                st.rerun()

            if next_round_btn:
                next_idx = round_idx + 1
                if next_idx >= total_rounds:
                    st.session_state.interview_complete = True
                else:
                    st.session_state.interview_round   = next_idx
                    st.session_state.interview_q_count = 0
                    with st.spinner(f"🎤 Starting {rounds[next_idx]['label']}…"):
                        opening = generate_interview_opening(iv_role, iv_domain, iv_diff, rounds[next_idx], iv_skills)
                    messages.append({"role":"assistant",
                                     "content":f"---\n### {rounds[next_idx]['label']}\n\n{opening}",
                                     "round":next_idx})
                    st.session_state.interview_messages = messages
                    st.session_state.interview_q_count  = 1
                st.rerun()

        else:
            st.divider()
            st.markdown("## 🏁 Interview Complete — Final Debrief")

            if score_log:
                avg_final = round(sum(s["score"] for s in score_log) / len(score_log), 1)
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("🏆 Final Score",     f"{avg_final}/10")
                with c2: st.metric("✅ Questions Scored", len(score_log))
                with c3: st.metric("🎯 Rounds Completed", len(rounds))

                bar_col = "#22c55e" if avg_final >= 7 else "#f59e0b" if avg_final >= 5 else "#ef4444"
                st.markdown(f"""
<div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:4px;margin:8px 0 16px 0;">
  <div style="background:{bar_col};width:{int(avg_final*10)}%;height:12px;border-radius:8px;"></div>
</div>""", unsafe_allow_html=True)

                if   avg_final >= 8: st.success("✅ **Excellent performance** — Strong Hire recommendation.")
                elif avg_final >= 6: st.warning("⚠️ **Good performance** — Hire with reservations.")
                else:                st.error("❌ **Needs improvement** — More practice recommended.")

                st.divider()
                st.markdown("### 📋 Score Breakdown by Question")
                for i, s in enumerate(score_log, 1):
                    sc = s["score"]
                    color = "🟢" if sc >= 7 else "🟡" if sc >= 5 else "🔴"
                    st.markdown(f"{color} **Q{i} ({s['round']}):** {s['question'][:90]}… → **{sc}/10**")

                st.divider()
                st.markdown("### 🤖 AI Debrief")
                with st.spinner("Generating your personalised debrief…"):
                    debrief = generate_interview_report(iv_role, score_log)
                st.info(debrief)

                save_interview_result([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    iv_name, iv_role, iv_domain, iv_diff,
                    ", ".join(iv_skills[:8]), avg_final, len(score_log), len(rounds),
                ])
            else:
                st.info("No answers were scored. Run the interview and answer questions to get your report.")

            if st.button("🔄 Start New Interview"):
                for k in ["interview_started","interview_messages","interview_context",
                          "interview_round","interview_score_log","interview_complete",
                          "interview_q_count","iv_name","iv_role","iv_domain",
                          "iv_difficulty","iv_skills","iv_rounds"]:
                    st.session_state.pop(k, None)
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████  AI LEARNING AGENT  ████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AI Learning Agent":

    st.header("🤖 AI Learning Agent")
    st.caption("Adaptive study loop — AI teaches, quizzes, re-explains, and tracks your mastery.")
    st.session_state.current_feature = "AI_Learning_Agent"

    if not st.session_state.get("agent_started"):

        col1, col2 = st.columns(2)
        with col1:
            ag_name  = st.text_input("Your Name",       key="ag_name_input")
            ag_topic = st.text_input("Topic to Master",
                placeholder="e.g. Python, Machine Learning, SQL, React…")
        with col2:
            ag_edu   = st.text_input("Education Level",
                placeholder="e.g. B.Tech, 12th Grade, Self-taught…")
            ag_level = st.selectbox("Difficulty Level", ["Beginner","Intermediate","Expert"])

        ag_goal = st.text_input("Learning Goal (optional)",
            placeholder="e.g. Crack interviews, Build projects, Pass exam…")

        st.info(
            "🔄 **How the AI Learning Agent works:**\n\n"
            "1. Generates a personalised 5–7 module learning plan\n"
            "2. Teaches each module with structured explanation + examples\n"
            "3. Quizzes you after every module (3 questions)\n"
            "4. Score ≥ 60% → proceed to next module\n"
            "5. Score < 60% → AI re-explains using a completely different approach\n"
            "6. Generates a final AI Mastery Report"
        )

        if st.button("🚀 Start Learning", use_container_width=True):
            if not ag_name or not ag_topic:
                st.warning("⚠️ Please enter your name and the topic to study.")
            else:
                if not check_request_limit(): st.stop()
                with st.spinner("🧠 Building your personalised learning plan…"):
                    plan = generate_learning_plan(ag_topic, ag_level, ag_goal)
                if not plan:
                    st.error("Could not generate learning plan. Please try again."); st.stop()

                st.session_state.agent_started  = True
                st.session_state.agent_name     = ag_name
                st.session_state.agent_topic    = ag_topic
                st.session_state.agent_edu      = ag_edu or "General"
                st.session_state.agent_level    = ag_level
                st.session_state.agent_goal     = ag_goal
                st.session_state.agent_plan     = plan
                st.session_state.agent_step     = 0
                st.session_state.agent_scores   = []
                st.session_state.agent_phase    = "explain"
                st.session_state.agent_quiz     = None
                st.session_state.agent_quiz_submitted = False
                st.session_state.current_user   = ag_name
                st.rerun()

    else:
        name_ag  = st.session_state.agent_name
        topic_ag = st.session_state.agent_topic
        level_ag = st.session_state.agent_level
        edu_ag   = st.session_state.agent_edu
        plan     = st.session_state.agent_plan
        step     = st.session_state.agent_step
        scores   = st.session_state.agent_scores
        phase    = st.session_state.agent_phase
        total    = len(plan)

        st.progress(min(step / total, 1.0))
        if step < total:
            st.markdown(f"**Module {step + 1} of {total} — {plan[step].get('title', '')}**")
        else:
            st.markdown("**✅ All modules complete!**")

        if scores:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Live Mastery")
            avg_live = round(sum(scores) / len(scores))
            st.sidebar.metric("Average Score", f"{avg_live}%")
            for i, s in enumerate(scores):
                icon = "🟢" if s >= 80 else "🟡" if s >= 60 else "🔴"
                st.sidebar.caption(f"{icon} Module {i + 1}: {s}%")

        # EXPLAIN phase
        if phase == "explain" and step < total:
            module  = plan[step]
            msg_key = f"explain_{step}"

            if msg_key not in st.session_state:
                past_ctx = " → ".join(plan[i].get("title","") for i in range(step)) if step > 0 else ""
                with st.spinner(f"📖 Teaching: {module.get('title','')}…"):
                    explanation = generate_module_explanation(
                        topic_ag, module.get("title",""),
                        module.get("objective",""), level_ag, edu_ag, past_ctx,
                    )
                st.session_state[msg_key] = explanation

            st.markdown(f"### 📖 Module {step + 1}: {module.get('title','')}")
            st.markdown(f"*Objective: {module.get('objective','')}*")
            st.divider()
            st.markdown(st.session_state[msg_key])
            st.divider()

            if st.button("✅ I understood this — Take the Quiz", use_container_width=True):
                if not check_request_limit(): st.stop()
                with st.spinner("📝 Generating quiz…"):
                    quiz = generate_module_quiz(topic_ag, module.get("title",""), level_ag)
                st.session_state.agent_quiz           = quiz
                st.session_state.agent_quiz_submitted = False
                st.session_state.agent_phase          = "quiz"
                st.rerun()

        # QUIZ phase
        elif phase == "quiz" and step < total:
            module = plan[step]
            quiz   = st.session_state.agent_quiz or []

            st.markdown(f"### 📝 Quiz — Module {step + 1}: {module.get('title','')}")

            if not quiz:
                st.warning("Quiz generation failed. Crediting module and moving on.")
                st.session_state.agent_scores.append(60)
                nxt = step + 1
                st.session_state.agent_step  = nxt
                st.session_state.agent_phase = "explain" if nxt < total else "done"
                st.rerun()

            for qi, q in enumerate(quiz):
                st.markdown(f"**Q{qi + 1}. {q.get('question','')}**")
                st.radio("", q.get("options",[]), index=None, key=f"aq_{step}_{qi}",
                         disabled=st.session_state.agent_quiz_submitted)

            if not st.session_state.agent_quiz_submitted:
                if st.button("Submit Quiz", use_container_width=True):
                    correct  = 0
                    wrong_qs = []
                    for qi, q in enumerate(quiz):
                        ans_idx     = q.get("answer", 0)
                        correct_opt = (q["options"][ans_idx] if ans_idx < len(q.get("options",[])) else "")
                        selected    = st.session_state.get(f"aq_{step}_{qi}")
                        if selected and selected == correct_opt:
                            correct += 1
                        else:
                            wrong_qs.append(q)
                    pct = round(correct / len(quiz) * 100) if quiz else 0
                    st.session_state.agent_quiz_submitted       = True
                    st.session_state[f"quiz_result_{step}"]     = {
                        "pct": pct, "correct": correct, "total": len(quiz), "wrong": wrong_qs,
                    }
                    st.rerun()

            if st.session_state.agent_quiz_submitted:
                result = st.session_state.get(f"quiz_result_{step}", {})
                pct    = result.get("pct", 0)
                wrong  = result.get("wrong", [])

                for qi, q in enumerate(quiz):
                    ans_idx     = q.get("answer", 0)
                    correct_opt = (q["options"][ans_idx] if ans_idx < len(q.get("options",[])) else "")
                    selected    = st.session_state.get(f"aq_{step}_{qi}")
                    if selected == correct_opt:
                        st.success(f"Q{qi + 1} ✅  {correct_opt}")
                    else:
                        st.error(f"Q{qi + 1} ❌  Your answer: {selected or '(none)'}  |  Correct: {correct_opt}")
                    if q.get("explanation"):
                        st.caption(f"💡 {q['explanation']}")

                st.divider()
                score_col = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:12px;padding:16px;text-align:center;">'
                    f'<h2 style="color:{score_col};margin:0;">{pct}%</h2>'
                    f'<p style="color:#94a3b8;margin:4px 0;">Module {step + 1} Score</p></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                if pct >= 60:
                    st.success(f"✅ {'Excellent mastery!' if pct >= 80 else 'Good — moving to next module.'}")
                    scores.append(pct)
                    st.session_state.agent_scores = scores
                    if st.button("Next Module ➡️", use_container_width=True):
                        nxt = step + 1
                        st.session_state.agent_step  = nxt
                        st.session_state.agent_phase = "explain" if nxt < total else "done"
                        st.session_state.agent_quiz  = None
                        st.session_state.agent_quiz_submitted = False
                        st.rerun()
                else:
                    st.warning(f"⚠️ {pct}% — below 60%. AI will re-explain differently before retry.")
                    re_key = f"reexplain_{step}"
                    if re_key not in st.session_state:
                        with st.spinner("🔄 Generating alternative explanation…"):
                            re_exp = generate_re_explanation(topic_ag, module.get("title",""), level_ag, wrong)
                        st.session_state[re_key] = re_exp
                    st.divider()
                    st.markdown("### 🔄 Alternative Explanation")
                    st.info(st.session_state[re_key])

                    if st.button("Try Quiz Again 🔁", use_container_width=True):
                        if not check_request_limit(): st.stop()
                        with st.spinner("📝 Generating a new quiz attempt…"):
                            new_quiz = generate_module_quiz(topic_ag, module.get("title",""), level_ag)
                        scores.append(max(pct, 40))
                        st.session_state.agent_scores         = scores
                        st.session_state.agent_quiz           = new_quiz
                        st.session_state.agent_quiz_submitted = False
                        st.session_state.pop(re_key, None)
                        st.session_state[f"quiz_result_{step}"] = {}
                        nxt = step + 1
                        st.session_state.agent_step  = nxt
                        st.session_state.agent_phase = "explain" if nxt < total else "done"
                        st.rerun()

        # DONE phase
        elif phase == "done" or step >= total:
            st.divider()
            st.markdown("## 🏆 Learning Complete — Mastery Report")

            final_scores = st.session_state.agent_scores
            if final_scores:
                avg     = round(sum(final_scores) / len(final_scores))
                bar_col = "#22c55e" if avg >= 80 else "#f59e0b" if avg >= 60 else "#ef4444"

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("🏆 Avg Mastery",  f"{avg}%")
                with c2: st.metric("✅ Modules Done", len(final_scores))
                with c3: st.metric("📚 Topic",        topic_ag)

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:10px;padding:4px;margin:8px 0 16px 0;">'
                    f'<div style="background:{bar_col};width:{avg}%;height:14px;border-radius:8px;"></div></div>',
                    unsafe_allow_html=True,
                )

                if   avg >= 80: st.success("✅ **Strong mastery** — you're ready to apply this knowledge.")
                elif avg >= 60: st.warning("⚠️ **Good progress** — review the weaker modules before moving on.")
                else:           st.error("❌ **Needs more practice** — revisit this topic with a lower difficulty.")

                st.markdown("### 📊 Module-by-Module Breakdown")
                for i, (mod, sc) in enumerate(zip(plan[:len(final_scores)], final_scores)):
                    icon = "🟢" if sc >= 80 else "🟡" if sc >= 60 else "🔴"
                    st.markdown(f"{icon} **Module {i + 1} — {mod.get('title','')}**: {sc}%")

                st.divider()
                st.markdown("### 🤖 AI Mastery Report")
                with st.spinner("Generating your personalised mastery report…"):
                    report = generate_mastery_report(name_ag, topic_ag, plan, final_scores)
                st.info(report)

                save_agent_progress([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name_ag, topic_ag, level_ag, edu_ag,
                    total, len(final_scores), avg,
                    json.dumps(final_scores),
                ])
            else:
                st.info("No module scores recorded yet.")

            if st.button("🔄 Start New Topic", use_container_width=True):
                keys_to_clear = ["agent_started","agent_plan","agent_step","agent_messages",
                                  "agent_quiz","agent_quiz_submitted","agent_scores",
                                  "agent_topic","agent_level","agent_name","agent_phase",
                                  "agent_edu","agent_goal"]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)
                for k in list(st.session_state.keys()):
                    if k.startswith(("explain_","reexplain_","quiz_result_","aq_")):
                        del st.session_state[k]
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████  ADMIN PORTAL  █████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔐 Admin Portal":

    st.header("🔐 Admin Portal")
    username = st.text_input("Admin Username")
    password = st.text_input("Admin Password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.success("✅ Admin Logged In")

            df = load_mock_results()
            if df.empty:
                st.warning("No mock test data available yet."); st.stop()

            df.columns = df.columns.str.strip().str.lower()
            df = df.rename(columns={
                "percentage":"percent","marks":"score",
                "level of exam":"difficulty","education level":"education",
                "name":"candidate_name","email":"candidate_email",
            })
            for col in ["percent","difficulty","score"]:
                if col not in df.columns:
                    st.error(f"Missing column: {col}"); st.write(df.columns.tolist()); st.stop()

            df["percent"] = pd.to_numeric(df["percent"], errors="coerce")
            df["score"]   = pd.to_numeric(df["score"],   errors="coerce")

            st.markdown("## 📊 Platform Overview")
            total_tests = len(df)
            avg_score   = df["percent"].mean()
            pass_rate   = (df["percent"] >= 80).mean() * 100

            def metric_card(title, value):
                st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.05);padding:20px;border-radius:12px;
                    text-align:center;border:1px solid rgba(255,255,255,0.1);">
                    <h4 style="color:#94a3b8;margin-bottom:10px;">{title}</h4>
                    <h2 style="color:white;font-weight:700;">{value}</h2></div>""",
                    unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            with c1: metric_card("Total Tests",   total_tests)
            with c2: metric_card("Average Score", f"{avg_score:.2f}%")
            with c3: metric_card("Pass Rate",     f"{pass_rate:.2f}%")

            st.divider()
            st.markdown("## 📈 Difficulty Breakdown")
            st.bar_chart(df["difficulty"].value_counts())
            st.divider()
            st.markdown("## 📊 Score Distribution")
            st.bar_chart(df["percent"])
            st.divider()
            st.markdown("## 🏆 Top Performers")
            st.dataframe(df.sort_values("percent",ascending=False).head(5)[
                ["candidate_name","candidate_email","difficulty","percent"]])
            st.divider()
            st.markdown("## 📂 Full Dataset")
            st.dataframe(df)

            @st.cache_data(ttl=300)
            def load_api_usage():
                try:
                    sheet = _gs_client().open("FutureProof_API_Usage").sheet1
                    data  = sheet.get_all_records()
                    return pd.DataFrame(data) if data else pd.DataFrame()
                except Exception:
                    return pd.DataFrame()

            st.divider()
            api_df = load_api_usage()

            if not api_df.empty:
                api_df.columns = api_df.columns.str.strip().str.lower()
                for col in ["estimated_cost","total_tokens"]:
                    if col in api_df.columns:
                        api_df[col] = pd.to_numeric(api_df[col], errors="coerce")

                if "timestamp" in api_df.columns:
                    api_df["timestamp"] = pd.to_datetime(api_df["timestamp"], errors="coerce")
                    today_start = pd.Timestamp.now().normalize()
                    today_df    = api_df[(api_df["timestamp"] >= today_start) &
                                         (api_df["timestamp"] < today_start + pd.Timedelta(days=1))]
                    st.markdown("## 🧠 Platform Health (Today)")
                    c1,c2,c3,c4 = st.columns(4)
                    with c1: st.metric("📊 Requests Today",     len(today_df))
                    with c2: st.metric("⚡ Avg Tokens/Request", int(today_df["total_tokens"].mean()) if "total_tokens" in today_df and len(today_df) else 0)
                    with c3: st.metric("💰 Today's AI Cost",    f"${today_df['estimated_cost'].sum():.4f}" if "estimated_cost" in today_df else "$0")
                    with c4: st.metric("👥 Active Users Today", today_df["user"].nunique() if "user" in today_df.columns else 0)
                    st.divider()

                st.markdown("## 💰 API Cost Analytics")
                st.metric("Total Platform API Cost", f"${api_df.get('estimated_cost', pd.Series([0])).sum():.4f}")

                if "user" in api_df.columns:
                    st.markdown("### 💵 Cost Per User")
                    st.dataframe(api_df.groupby("user")["estimated_cost"].sum().reset_index())
                    st.divider()
                    st.markdown("## 🔥 Most Active Users")
                    st.bar_chart(api_df["user"].value_counts().head(10))
                    st.divider()

                if "feature" in api_df.columns:
                    st.markdown("## 📊 AI Usage by Feature")
                    st.bar_chart(api_df["feature"].value_counts())
                    st.divider()

                if "model" in api_df.columns:
                    st.markdown("## 💰 Cost by AI Model")
                    st.bar_chart(api_df.groupby("model")["estimated_cost"].sum())
                    st.divider()

                if "total_tokens" in api_df.columns:
                    st.markdown("## 📈 Token Usage Trend")
                    st.line_chart(api_df["total_tokens"])
            else:
                st.info("No API usage data available yet.")
        else:
            st.error("❌ Invalid Admin Credentials")
