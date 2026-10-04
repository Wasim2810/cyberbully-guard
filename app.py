"""
app.py
------
Main Streamlit application for the Cyberbullying Detection System.

Pages:
    - Home
    - About Project
    - Cyberbullying Detection (live prediction)
    - Statistics
    - History
    - Admin Dashboard
    - Model Accuracy

Run with:
    streamlit run app.py
"""

import os
import io
import json
import uuid
import time
import datetime
import pandas as pd
import streamlit as st

from utils.preprocessing import clean_text, get_pipeline_status
from utils.helper import (
    detect_harmful_words,
    highlight_text,
    get_risk_level,
    get_reason,
    get_suggestion,
    validate_input,
    sanitize_input,
    flag_suspicious_patterns,
    global_rate_limiter,
)
from utils.database import (
    insert_prediction,
    fetch_all_predictions,
    fetch_stats,
    delete_all_history,
    delete_prediction_by_id,
    log_admin_action,
    logger,
)
from utils.visualization import (
    build_pie_chart,
    build_bar_chart,
    build_history_timeline,
    build_top_words_chart,
    build_confidence_gauge,
    build_model_comparison_chart,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --------------------------------------------------------------------------
# THEME / CUSTOM CSS
# Design tokens:
#   Background : #0a0e17 (near-black navy) / surface #121826
#   Accent     : #00e5ff (cyan - "scan" accent), #ff3860 (danger), #00e676 (safe)
#   Type       : 'JetBrains Mono' for headers/labels (security-terminal feel),
#                'Inter' for body copy
#   Signature  : animated horizontal "scan line" under the hero title
# --------------------------------------------------------------------------
def inject_css(dark: bool):
    if dark:
        bg = "#0a0e17"
        surface = "#121826"
        surface2 = "#1a2236"
        text = "#e6e9f0"
        muted = "#8b93a7"
        border = "#232c42"
    else:
        bg = "#f5f7fa"
        surface = "#ffffff"
        surface2 = "#eef1f6"
        text = "#0f1420"
        muted = "#5a6273"
        border = "#dbe0ea"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {text};
        }}

        .stApp {{
            background-color: {bg};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {surface};
            border-right: 1px solid {border};
        }}

        h1, h2, h3, .hero-title {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .hero-wrap {{
            padding: 28px 32px;
            border-radius: 14px;
            background: linear-gradient(135deg, {surface} 0%, {surface2} 100%);
            border: 1px solid {border};
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }}
        .hero-title {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: {text};
            margin-bottom: 4px;
        }}
        .hero-sub {{
            color: {muted};
            font-size: 1.0rem;
            font-family: 'Inter', sans-serif;
        }}
        .scan-line {{
            height: 3px;
            width: 100%;
            margin-top: 18px;
            border-radius: 3px;
            background: linear-gradient(90deg, #00e5ff, #6a5cff, #ff3860, #00e5ff);
            background-size: 300% 100%;
            animation: scan 4s linear infinite;
        }}
        @keyframes scan {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 300% 50%; }}
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            margin-right: 6px;
        }}
        .badge-safe {{ background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.4);}}
        .badge-offensive {{ background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.4);}}
        .badge-toxic {{ background: rgba(230,126,34,0.15); color: #e67e22; border: 1px solid rgba(230,126,34,0.4);}}
        .badge-cyberbullying {{ background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4);}}
        .badge-hate_speech {{ background: rgba(155,89,182,0.15); color: #9b59b6; border: 1px solid rgba(155,89,182,0.4);}}
        .badge-threat {{ background: rgba(192,57,43,0.18); color: #ff3860; border: 1px solid rgba(255,56,96,0.5);}}

        .metric-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 18px 20px;
        }}

        .result-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 22px 24px;
            margin-top: 14px;
        }}

        .footer-note {{
            color: {muted};
            font-size: 0.82rem;
            text-align: center;
            margin-top: 40px;
            font-family: 'JetBrains Mono', monospace;
        }}

        mark {{
            font-weight: 600;
        }}

        div.stButton > button {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.dark_mode)

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ CyberGuard AI")
st.sidebar.caption("Cyberbullying Detection System")
st.sidebar.divider()

PAGES = [
    "🏠 Home",
    "📖 About Project",
    "🔍 Cyberbullying Detection",
    "📊 Statistics",
    "🕘 History",
    "🔐 Admin Dashboard",
    "🎯 Model Accuracy",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
dark_toggle = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
if dark_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_toggle
    st.rerun()

st.sidebar.caption(f"NLP Engine: `{get_pipeline_status()}`")
st.sidebar.caption(f"Session: `{st.session_state.session_id[:8]}`")

MODEL_EXISTS = os.path.exists(os.path.join(MODEL_DIR, "model.pkl"))
if not MODEL_EXISTS:
    st.sidebar.error("⚠️ Model not trained yet. Run `python train_model.py` first.")


# ==========================================================================
# PAGE: HOME
# ==========================================================================
def page_home():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">🛡️ CyberGuard AI</div>
            <div class="hero-sub">Real-time Cyberbullying &amp; Toxic Language Detection powered by NLP + Machine Learning</div>
            <div class="scan-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    stats = fetch_stats()
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>📥 Total Scanned</h3>'
            f'<h1>{stats["total"]}</h1></div>', unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>✅ Safe Messages</h3>'
            f'<h1 style="color:#00e676">{stats["safe_count"]}</h1></div>', unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><h3>🚨 Harmful Flagged</h3>'
            f'<h1 style="color:#ff3860">{stats["harmful_count"]}</h1></div>', unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("What this system detects")
    labels = [
        ("Safe", "badge-safe", "Respectful, non-harmful communication"),
        ("Offensive", "badge-offensive", "Rude or disrespectful language"),
        ("Toxic", "badge-toxic", "Language likely to make someone leave a discussion"),
        ("Cyberbullying", "badge-cyberbullying", "Targeted, repeated abusive behavior online"),
        ("Hate Speech", "badge-hate_speech", "Attacks based on identity/group membership"),
        ("Threat", "badge-threat", "Explicit intent to cause harm"),
    ]
    for name, css_class, desc in labels:
        st.markdown(
            f'<span class="badge {css_class}">{name}</span> {desc}',
            unsafe_allow_html=True,
        )
        st.write("")

    st.write("")
    st.info(
        "👉 Head to **🔍 Cyberbullying Detection** in the sidebar to analyze a message in real time."
    )


# ==========================================================================
# PAGE: ABOUT PROJECT
# ==========================================================================
def page_about():
    st.markdown('<div class="hero-title">📖 About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
        ### Project Title
        **Cyberbullying Detection System using NLP and Machine Learning**

        ### Objective
        Cyberbullying is a growing threat across social media platforms, messaging apps,
        and online communities. This project builds an end-to-end system that automatically
        classifies text messages into six categories — **Safe, Offensive, Toxic,
        Cyberbullying, Hate Speech, and Threat** — using classical NLP preprocessing and
        supervised machine learning.

        ### Pipeline Overview
        1. **Data Collection** — Labeled text dataset (Jigsaw / Kaggle-style cyberbullying corpus).
        2. **Text Preprocessing** — Lowercasing, URL/HTML/number/emoji/punctuation removal,
           stopword removal, tokenization, and lemmatization.
        3. **Feature Extraction** — TF-IDF vectorization (unigrams + bigrams).
        4. **Model Training** — Logistic Regression, Naive Bayes, Linear SVM, and Random Forest
           are trained and benchmarked; the best performer is automatically selected.
        5. **Explainability** — Harmful keyword highlighting, confidence scoring, and
           risk-level assessment accompany every prediction.
        6. **Persistence & Auditing** — Every prediction is stored in a SQLite database for
           statistics, history review, and admin auditing.
        7. **Security Hardening** — Input validation, SQL-injection-safe parameterized
           queries, XSS sanitization, rate limiting, and activity logging.

        ### Tech Stack
        | Layer | Technology |
        |---|---|
        | Frontend | Streamlit |
        | ML | Scikit-learn |
        | NLP | NLTK / spaCy (with offline fallback) |
        | Database | SQLite |
        | Visualization | Plotly, Matplotlib |
        | Persistence | Joblib |

        ### Disclaimer
        This is an academic / final-year project. The bundled dataset is a synthetically
        generated demonstration dataset. For production or research use, retrain on a
        real-world corpus such as the Jigsaw Toxic Comment Classification dataset.
        """
    )


# ==========================================================================
# PAGE: DETECTION
# ==========================================================================
def page_detection():
    st.markdown('<div class="hero-title">🔍 Cyberbullying Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    if not MODEL_EXISTS:
        st.error("Model artifacts not found. Please run `python train_model.py` first, then restart the app.")
        return

    input_mode = st.radio("Input mode", ["⌨️ Type a message", "🎙️ Speech-to-text (upload audio)"], horizontal=True)

    text_input = ""
    if input_mode == "⌨️ Type a message":
        text_input = st.text_area(
            "Enter a message to analyze",
            placeholder="e.g. You are useless and everyone hates you.",
            height=120,
            key="live_text_input",
        )
        # Live typing detection: quick heuristic feedback as the user types
        if text_input.strip():
            quick_hits = detect_harmful_words(text_input)
            if quick_hits:
                st.warning(
                    f"⚡ Live scan: potentially harmful terms detected while typing "
                    f"({len(quick_hits)} match(es)). Click 'Analyze' for a full report."
                )
    else:
        st.info(
            "🎙️ Speech-to-text is a bonus feature. Upload a `.wav` audio file — if the "
            "`SpeechRecognition` package is installed, it will be transcribed automatically."
        )
        audio_file = st.file_uploader("Upload audio (.wav)", type=["wav"])
        if audio_file is not None:
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                text_input = recognizer.recognize_google(audio_data)
                st.success(f"Transcribed text: {text_input}")
            except ImportError:
                st.error(
                    "The 'SpeechRecognition' package is not installed. "
                    "Run: pip install SpeechRecognition pyaudio"
                )
            except Exception as exc:
                st.error(f"Could not transcribe audio: {exc}")

    analyze_clicked = st.button("🔎 Analyze Message", type="primary", use_container_width=True)

    if analyze_clicked:
        # ---- Security: rate limiting ----
        if not global_rate_limiter.allow(st.session_state.session_id):
            st.error("🚫 Rate limit exceeded. Please wait a moment before analyzing more messages.")
            logger.warning(f"Rate limit hit for session {st.session_state.session_id}")
            return

        # ---- Security: input validation ----
        is_valid, error_msg = validate_input(text_input)
        if not is_valid:
            st.error(f"⚠️ {error_msg}")
            return

        # ---- Security: SQLi / XSS pattern flagging ----
        warnings = flag_suspicious_patterns(text_input)
        for w in warnings:
            st.warning(f"🛡️ {w}")
        safe_display_text = sanitize_input(text_input)

        # ---- Run inference ----
        from predict import predict_text  # local import keeps app.py importable without a trained model
        with st.spinner("Analyzing message..."):
            time.sleep(0.3)  # brief UX delay for the "scanning" feel
            result = predict_text(text_input)

        # ---- Persist to database ----
        insert_prediction(
            input_text=safe_display_text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
        )
        st.session_state.last_result = result

    # ---- Render last result (persists across reruns within the session) ----
    result = st.session_state.last_result
    if result:
        st.markdown("### 📋 Detection Report")
        badge_class = f"badge-{result['prediction']}"
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown(
                f"""
                <div class="result-card">
                    <span class="badge {badge_class}">{result['prediction'].replace('_',' ').upper()}</span>
                    <span class="badge" style="background:rgba(255,255,255,0.06);color:#8b93a7;border:1px solid #333;">
                        RISK: {result['risk_level'].upper()}
                    </span>
                    <p style="margin-top:14px;"><b>Reason for Detection:</b><br>{result['reason']}</p>
                    <p><b>Recommended Action:</b><br>{result['suggestion']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["harmful_words"]:
                st.markdown("**Highlighted Harmful Words:**")
                highlighted = highlight_text(text_input or "", result["harmful_words"])
                st.markdown(f'<div class="result-card">{highlighted}</div>', unsafe_allow_html=True)

        with c2:
            fig = build_confidence_gauge(result["confidence"], result["prediction"])
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📈 Full class probability breakdown"):
            prob_df = pd.DataFrame(
                sorted(result["all_class_probabilities"].items(), key=lambda x: -x[1]),
                columns=["Class", "Probability"],
            )
            prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: STATISTICS
# ==========================================================================
def page_statistics():
    st.markdown('<div class="hero-title">📊 Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    stats = fetch_stats()
    if stats["total"] == 0:
        st.info("No predictions yet. Analyze a message on the Detection page to populate statistics.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", stats["total"])
    col2.metric("Safe", stats["safe_count"])
    col3.metric("Harmful", stats["harmful_count"])
    col4.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_pie_chart(stats["by_class"]), use_container_width=True)
    with c2:
        st.plotly_chart(build_bar_chart(stats["by_class"]), use_container_width=True)

    history = fetch_all_predictions()
    df = pd.DataFrame(history)
    if not df.empty:
        st.plotly_chart(build_history_timeline(df), use_container_width=True)

        # Most frequent offensive words across all flagged (non-safe) messages
        word_freq = {}
        for text in df[df["prediction"] != "safe"]["input_text"]:
            for word, _cat in detect_harmful_words(text):
                word_freq[word] = word_freq.get(word, 0) + 1
        if word_freq:
            st.plotly_chart(build_top_words_chart(word_freq), use_container_width=True)


# ==========================================================================
# PAGE: HISTORY
# ==========================================================================
def page_history():
    st.markdown('<div class="hero-title">🕘 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    history = fetch_all_predictions()
    if not history:
        st.info("No history yet. Analyze a message on the Detection page first.")
        return

    df = pd.DataFrame(history)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        class_filter = st.multiselect(
            "Filter by class", options=sorted(df["prediction"].unique()), default=None
        )
    with col_b:
        search_term = st.text_input("Search text", "")

    filtered = df.copy()
    if class_filter:
        filtered = filtered[filtered["prediction"].isin(class_filter)]
    if search_term:
        filtered = filtered[filtered["input_text"].str.contains(search_term, case=False, na=False)]

    st.dataframe(
        filtered[["id", "input_text", "prediction", "confidence", "risk_level", "timestamp"]],
        use_container_width=True,
        hide_index=True,
    )

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Download filtered history as CSV",
        data=csv_buffer.getvalue(),
        file_name="cyberbullying_history.csv",
        mime="text/csv",
    )


# ==========================================================================
# PAGE: ADMIN DASHBOARD
# ==========================================================================
def page_admin():
    st.markdown('<div class="hero-title">🔐 Admin Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")  # override via env var in production
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.warning("🔒 Admin authentication required.")
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                log_admin_action("login", "Admin logged in")
                st.rerun()
            else:
                st.error("Incorrect password.")
                logger.warning("Failed admin login attempt.")
        st.caption("Default demo password: `admin123` (set the ADMIN_PASSWORD environment variable to change it).")
        return

    stats = fetch_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", stats["total"])
    col2.metric("Safe Messages", stats["safe_count"])
    col3.metric("Toxic/Harmful Messages", stats["harmful_count"])

    st.write("")
    history = fetch_all_predictions()
    df = pd.DataFrame(history)

    if not df.empty:
        word_freq = {}
        for text in df[df["prediction"] != "safe"]["input_text"]:
            for word, _cat in detect_harmful_words(text):
                word_freq[word] = word_freq.get(word, 0) + 1
        if word_freq:
            st.plotly_chart(build_top_words_chart(word_freq, top_n=15), use_container_width=True)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Full CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"cyberbullying_report_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    st.write("")
    st.subheader("⚠️ Danger Zone")
    danger_col1, danger_col2 = st.columns(2)
    with danger_col1:
        confirm = st.checkbox("I understand this will permanently delete all history.")
        if st.button("🗑️ Delete All History", disabled=not confirm):
            delete_all_history()
            log_admin_action("delete_history", "All prediction history deleted")
            st.success("All history deleted.")
            st.rerun()
    with danger_col2:
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            log_admin_action("logout", "Admin logged out")
            st.rerun()


# ==========================================================================
# PAGE: MODEL ACCURACY
# ==========================================================================
def page_model_accuracy():
    st.markdown('<div class="hero-title">🎯 Model Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.write("")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.error("No metrics found. Please run `python train_model.py` first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    st.success(f"🏆 Best Model Selected: **{metrics['best_model']}**")

    results = metrics["results"]
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall": f"{m['recall']*100:.2f}%",
            "F1 Score": f"{m['f1_score']*100:.2f}%",
            "Train Time (s)": f"{m['train_time_sec']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.plotly_chart(build_model_comparison_chart(results), use_container_width=True)

    st.write("")
    c1, c2 = st.columns(2)
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    with c1:
        st.markdown("**Confusion Matrix**")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
    with c2:
        st.markdown("**ROC Curve (One-vs-Rest)**")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)

    st.caption(f"Trained on {metrics['trained_on_rows']} rows | Classes: {', '.join(metrics['classes'])}")


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
if page == "🏠 Home":
    page_home()
elif page == "📖 About Project":
    page_about()
elif page == "🔍 Cyberbullying Detection":
    page_detection()
elif page == "📊 Statistics":
    page_statistics()
elif page == "🕘 History":
    page_history()
elif page == "🔐 Admin Dashboard":
    page_admin()
elif page == "🎯 Model Accuracy":
    page_model_accuracy()

st.markdown(
    '<div class="footer-note">CyberGuard AI &copy; 2026 | Final Year Cybersecurity Project | '
    'Built with Streamlit + Scikit-learn</div>',
    unsafe_allow_html=True,
)
