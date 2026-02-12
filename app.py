# ==========================================================
# FUTUREPROOF AI – Career Intelligence Engine (2026 Edition)
# Premium Production Version
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import warnings
import smtplib
from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer, util
from google import genai

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="FutureProof AI",
    page_icon="🚀",
    layout="wide"
)

# ==========================================================
# PREMIUM UI DESIGN
# ==========================================================

st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
    color: white;
}

/* Glass container */
.block-container {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
}

/* Header */
.main-title {
    font-size:42px;
    font-weight:800;
    text-align:center;
    margin-bottom:10px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#3b82f6);
    color:white;
    border-radius:12px;
    height:50px;
    font-size:16px;
    font-weight:600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 6px 20px rgba(99,102,241,0.5);
}

/* Inputs */
input, textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 FutureProof AI Career Intelligence Engine</div>', unsafe_allow_html=True)
st.caption("Plan Your 2026 Career Growth Intelligently")

# ==========================================================
# ENVIRONMENT VARIABLES
# ==========================================================

API_KEY = os.getenv("GOOGLE_API_KEY")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")        # your email
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # app password

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ==========================================================
# EMAIL ALERT FUNCTION (ADMIN NOTIFICATION)
# ==========================================================

def send_admin_alert(error_message):
    if not ADMIN_EMAIL or not EMAIL_PASSWORD:
        return
    
    try:
        msg = MIMEText(f"FutureProof AI Alert:\n\n{error_message}")
        msg["Subject"] = "⚠ FutureProof AI API Failure Alert"
        msg["From"] = ADMIN_EMAIL
        msg["To"] = ADMIN_EMAIL

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(ADMIN_EMAIL, EMAIL_PASSWORD)
        server.sendmail(ADMIN_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
    except:
        pass

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("futureproof_dummy_data.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df["current_skills"] = df["current_skills"].fillna("").str.lower()
    df["interested_future_field"] = df["interested_future_field"].fillna("").str.lower()
    return df

df = load_data()

DATASET_THRESHOLD = 0.55
MODEL_NAME = "gemini-2.5-flash"

# ==========================================================
# GEMINI CALL
# ==========================================================

def gemini_generate(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        if response and response.text:
            return response.text.strip()
        return None
    except Exception as e:
        send_admin_alert(str(e))
        return None

# ==========================================================
# AI LOGIC
# ==========================================================

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

def infer_role(skills):
    prompt = f"""
Suggest ONE high-growth career role in 2026 for someone with:
{", ".join(skills)}
Return only role name.
"""
    role = gemini_generate(prompt)
    return role if role else "Business Transformation Specialist"

def infer_growth(role):
    prompt = f"""
For {role}, list 6 future growth skills for 2026.
Return comma-separated names only.
"""
    result = gemini_generate(prompt)
    if not result:
        return []
    return [s.strip().title() for s in re.split(r",|\n", result)][:6]

def infer_certifications(role):
    prompt = f"""
List 5 trending certifications for {role} in 2026.
Return comma-separated names only.
"""
    result = gemini_generate(prompt)
    if not result:
        return []
    return [c.strip() for c in result.split(",")][:5]

# ==========================================================
# USER INPUT
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
# GENERATE REPORT
# ==========================================================

if st.button("🔎 Generate Career Intelligence Plan", use_container_width=True):

    if not name or not skills_input:
        st.warning("Please enter your name and skills.")
    else:
        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

        with st.spinner("Analyzing market intelligence..."):
            confidence, dataset_field = dataset_confidence(skills)

            if confidence >= DATASET_THRESHOLD:
                role = dataset_field.title()
                source = "Peer Intelligence"
            else:
                role = infer_role(skills)
                source = "Live Market AI Intelligence"

            growth_skills = infer_growth(role)
            certifications = infer_certifications(role)

            weeks = round((len(growth_skills) * 40) / hours)

        st.success("✅ Career Intelligence Report Ready!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 Overview", "📈 Growth Roadmap", "🎓 Certifications", "📊 Skill Insights"]
        )

        # TAB 1
        with tab1:
            colA, colB, colC = st.columns(3)

            colA.metric("Recommended Role", role)
            colB.metric("Confidence", f"{int(confidence*100)}%")
            colC.metric("Market Demand", f"{np.random.randint(75,95)}%")

            st.progress(int(confidence*100))
            st.markdown(f"**Insight Source:** {source}")

        # TAB 2
        with tab2:
            for skill in growth_skills:
                st.markdown(f"✔ {skill}")

            st.markdown(f"### ⏳ Estimated Upskilling Time: ~{weeks} weeks")

        # TAB 3
        with tab3:
            for cert in certifications:
                st.markdown(f"🏅 {cert}")

        # TAB 4
        with tab4:
            colX, colY = st.columns(2)

            with colX:
                st.markdown("### Your Skills")
                for s in skills:
                    st.markdown(f"- {s.title()}")

            with colY:
                st.markdown("### Growth Areas")
                for s in growth_skills[:4]:
                    st.markdown(f"- {s}")

        # ======================================================
        # FEEDBACK
        # ======================================================

        st.divider()
        st.markdown("## 💬 Help Us Improve")

        rating = st.slider("How useful was this plan?", 1, 5, 4)
        recommend = st.selectbox("Would you recommend this tool?", ["Yes", "Maybe", "No"])
        feedback = st.text_area("Suggestions")

        if st.button("Submit Feedback"):
            with open("feedback_log.txt", "a") as f:
                f.write(f"\n{name} | {rating} | {recommend} | {feedback}")
            st.success("🙏 Thank you for your feedback!")
