# ============================================================
# CDS6344 GROUP 9 — STREAMLIT APP
# Aspect-Based Sentiment Analysis of Social Networking App Reviews
# Using Machine Learning, Transformer Models, and Opinion Spam Detection
# ============================================================

import os
import re
import json
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional model imports. The app still opens even if transformers/torch
# are not installed, but the ABSA live model will require them.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    softmax = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ABSA · Group 9 · CDS6344",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

DATA_DIR = "data"
ASSETS_DIR = "assets"
MODEL_DIR = "model"

LOCAL_MODEL_DIR = os.path.join(MODEL_DIR, "sentiment_roberta_finetuned")
HF_FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

REVIEW_SPAM_FILE = os.path.join(DATA_DIR, "review_level_social_reviews_with_spam_risk.csv")
MODEL_COMPARISON_FILE = os.path.join(DATA_DIR, "final_aspect_level_model_comparison_updated.csv")
BEST_MODEL_FILE = os.path.join(DATA_DIR, "final_best_model_summary_updated.csv")
SPAM_FINDINGS_FILE = os.path.join(DATA_DIR, "final_spam_detection_findings.csv")

FINAL_ACCURACY = "82.26%"
FINAL_MACRO_F1 = "81.97%"
FINAL_WEIGHTED_F1 = "82.21%"
FINAL_MODEL_NAME = "Sentiment RoBERTa + Threshold Tuning"


# ============================================================
# GLOBAL CSS — DARK TECH THEME
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent: #00d4aa;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --text-faint: #64748b;
    --border: rgba(0,212,170,0.16);
    --danger: #ef4444;
    --warn: #f59e0b;
    --success: #10b981;
    --font-display: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-main) !important;
    font-family: var(--font-body) !important;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, section.main {
    background: radial-gradient(circle at 15% 5%, rgba(0,212,170,0.10), transparent 24%),
                radial-gradient(circle at 85% 15%, rgba(124,58,237,0.10), transparent 28%),
                #0a0e1a !important;
    color: var(--text-main) !important;
}

.main .block-container {
    background: transparent !important;
    color: var(--text-main) !important;
    padding: 1.4rem 2rem 3rem 2rem !important;
    max-width: 1450px !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

p, li, span, div, label { color: var(--text-main); }
h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
[data-testid="stMarkdownContainer"] { color: var(--text-main) !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--accent) !important;
    background: rgba(0,212,170,0.08) !important;
}

.page-hero {
    background: linear-gradient(135deg, rgba(17,24,39,0.98) 0%, rgba(26,34,53,0.94) 55%, rgba(15,23,42,0.98) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.4rem 2.5rem 2.1rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 50px rgba(0,0,0,0.22);
}
.page-hero::before {
    content: "";
    position: absolute;
    top: -90px;
    right: -90px;
    width: 270px;
    height: 270px;
    background: radial-gradient(circle, rgba(0,212,170,0.13) 0%, transparent 70%);
    pointer-events: none;
}
.page-hero-tag {
    font-family: var(--font-display) !important;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    color: var(--accent) !important;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.page-hero h1 {
    font-family: var(--font-display) !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    margin: 0 0 0.65rem 0 !important;
}
.page-hero p {
    color: var(--text-muted) !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
    max-width: 820px;
    margin: 0 !important;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card, .info-card, .member-card {
    background: rgba(17,24,39,0.96);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover, .info-card:hover, .member-card:hover {
    border-color: rgba(0,212,170,0.42);
    transform: translateY(-2px);
    box-shadow: 0 12px 38px rgba(0,0,0,0.22);
}
.stat-card::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.stat-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-faint) !important;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: var(--font-display) !important;
    font-size: 1.58rem;
    font-weight: 700;
    color: var(--accent) !important;
    line-height: 1;
}
.stat-sub {
    font-size: 0.78rem;
    color: var(--text-faint) !important;
    margin-top: 0.30rem;
}

.section-label {
    font-family: var(--font-display) !important;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    color: var(--accent) !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: var(--font-display) !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    margin: 0 0 1.1rem 0 !important;
}

.pipeline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    margin: 1rem 0;
    gap: 0.25rem;
}
.pipeline-step {
    background: rgba(26,34,53,0.92);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.50rem 0.85rem;
    font-size: 0.78rem;
    color: var(--text-main) !important;
    white-space: nowrap;
}
.pipeline-arrow {
    color: var(--accent) !important;
    font-size: 1rem;
    padding: 0 0.25rem;
}

.member-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.member-avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display) !important;
    font-weight: 700;
    font-size: 0.9rem;
    color: white !important;
    flex-shrink: 0;
}
.member-name { font-weight: 600; font-size: 0.95rem; color: var(--text-main) !important; }
.member-id { font-family: var(--font-display) !important; font-size: 0.72rem; color: var(--text-faint) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00a896) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: var(--font-display) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 0.64rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,212,170,0.35) !important;
}

.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text-main) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--text-main) !important;
}
.stSelectbox > div > div, div[data-baseweb="select"] {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}
div[data-baseweb="select"] * { color: var(--text-main) !important; }

.result-positive, .result-negative, .result-warning {
    border-radius: 11px;
    padding: 1rem 1.25rem;
    min-height: 112px;
}
.result-positive { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.4); }
.result-negative { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4); }
.result-warning { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); }
.result-label {
    font-family: var(--font-display) !important;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.result-value {
    font-family: var(--font-display) !important;
    font-size: 1.42rem;
    font-weight: 700;
}
.result-positive .result-label, .result-positive .result-value { color: var(--success) !important; }
.result-negative .result-label, .result-negative .result-value { color: var(--danger) !important; }
.result-warning .result-label, .result-warning .result-value { color: var(--warn) !important; }

.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.55rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), #00a896);
}

.tag-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.tag {
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: var(--accent) !important;
}
.tag.danger { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.25); color: var(--danger) !important; }
.tag.warn { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.25); color: var(--warn) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(17,24,39,0.96) !important;
    border-radius: 11px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), #00a896) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background: rgba(17,24,39,0.96) !important;
    border: 1px solid var(--border) !important;
    border-radius: 11px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--font-display) !important; }

.stAlert, [data-testid="stAlert"] {
    background-color: #111827 !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 11px !important;
}

.stCodeBlock, code, pre {
    background-color: #111827 !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
}

[data-testid="stDataFrame"] {
    background-color: #111827 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.3); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


review_spam_df = load_csv(REVIEW_SPAM_FILE)
model_comparison_df = load_csv(MODEL_COMPARISON_FILE)
best_model_df = load_csv(BEST_MODEL_FILE)
spam_findings_df = load_csv(SPAM_FINDINGS_FILE)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_absa_model():
    """Load local fine-tuned model first; fallback to Hugging Face base model."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None, "unavailable", None

    local_config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    streamlit_config_path = os.path.join(LOCAL_MODEL_DIR, "streamlit_model_config.json")

    if os.path.exists(local_config_path):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
        model.eval()

        app_config = {}
        if os.path.exists(streamlit_config_path):
            with open(streamlit_config_path, "r", encoding="utf-8") as f:
                app_config = json.load(f)

        threshold = float(app_config.get("best_threshold", 0.50))
        return tokenizer, model, threshold, "local_finetuned", app_config

    tokenizer = AutoTokenizer.from_pretrained(HF_FALLBACK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(HF_FALLBACK_MODEL)
    model.eval()
    return tokenizer, model, None, "hf_fallback", None


try:
    absa_tokenizer, absa_model, absa_threshold, absa_model_source, absa_app_config = load_absa_model()
    ABSA_MODEL_READY = absa_model is not None
    ABSA_LOAD_ERROR = None
except Exception as e:
    absa_tokenizer, absa_model, absa_threshold, absa_model_source, absa_app_config = None, None, None, "error", None
    ABSA_MODEL_READY = False
    ABSA_LOAD_ERROR = str(e)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_image_if_exists(filename, caption=None):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="background:rgba(0,212,170,0.05);border:1px dashed rgba(0,212,170,0.2);
                        border-radius:10px;padding:2rem;text-align:center;color:#94a3b8;font-size:0.82rem;">
                📁 &nbsp;<code>{filename}</code><br>
                <span style="font-size:0.75rem;margin-top:0.25rem;display:block;color:#64748b;">
                    Place this file in the <code>assets/</code> folder to display
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def calculate_uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def repeated_word_ratio(text):
    words = re.findall(r"\b\w+\b", str(text).lower())
    if not words:
        return 0
    return 1 - (len(set(words)) / len(words))


def simple_lexicon_sentiment(text):
    pos = {
        "good", "great", "excellent", "amazing", "awesome", "best", "love", "liked", "like",
        "perfect", "nice", "easy", "useful", "helpful", "fast", "smooth", "secure", "safe",
        "clear", "reliable", "recommend", "happy", "satisfied", "works", "working", "cool", "fine",
    }
    neg = {
        "bad", "poor", "terrible", "awful", "worst", "hate", "hated", "problem", "problems",
        "issue", "issues", "bug", "bugs", "crash", "crashes", "crashed", "slow", "lag", "broken",
        "fail", "failed", "failure", "error", "errors", "annoying", "difficult", "hard",
        "privacy", "unsafe", "spam", "scam", "expensive", "waste", "not", "never", "cannot", "cant",
    }
    words = re.findall(r"\b\w+\b", str(text).lower())
    p = sum(1 for w in words if w in pos)
    n = sum(1 for w in words if w in neg)
    return "Positive" if p > n else "Negative" if n > p else "Neutral"


def assign_rating_sentiment(rating):
    if rating >= 4:
        return "Positive"
    if rating <= 2:
        return "Negative"
    return "Neutral"


def detect_rating_text_conflict(rating_sentiment, text_sentiment):
    return int(
        (rating_sentiment == "Positive" and text_sentiment == "Negative")
        or (rating_sentiment == "Negative" and text_sentiment == "Positive")
    )


def roberta_absa_predict(sentence, category, term):
    """Predict aspect sentiment using local fine-tuned model when available."""
    if not ABSA_MODEL_READY:
        raise RuntimeError("ABSA model is not loaded.")

    model_input = (
        f"Review sentence: {sentence} "
        f"Aspect category: {category} "
        f"Aspect term: {term}"
    )

    encoded = absa_tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        output = absa_model(**encoded)

    logits = output.logits[0].detach().cpu().numpy()
    probabilities = softmax(logits)

    # Local fine-tuned model is binary: negative=0, positive=1.
    if absa_model_source == "local_finetuned" and len(probabilities) == 2:
        negative_score = float(probabilities[0])
        positive_score = float(probabilities[1])
        threshold = absa_threshold if absa_threshold is not None else 0.50
        prediction = "positive" if positive_score >= threshold else "negative"
        confidence = positive_score if prediction == "positive" else negative_score
        neutral_score = 0.0
    else:
        # Hugging Face fallback model is 3-class: negative, neutral, positive.
        id2label = absa_model.config.id2label
        label_scores = {id2label[i].lower(): float(probabilities[i]) for i in range(len(probabilities))}
        negative_score = label_scores.get("negative", 0.0)
        neutral_score = label_scores.get("neutral", 0.0)
        positive_score = label_scores.get("positive", 0.0)
        prediction = "positive" if positive_score >= negative_score else "negative"
        confidence = positive_score if prediction == "positive" else negative_score

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "positive_score": float(positive_score),
        "neutral_score": float(neutral_score),
        "negative_score": float(negative_score),
        "model_input": model_input,
        "source": absa_model_source,
        "threshold": absa_threshold,
    }


def spam_risk_detector(review_text, rating, app_avg=3.22):
    review_text = str(review_text)
    word_count = len(review_text.split())

    very_short = int(word_count <= 5)
    very_long = int(word_count >= 214)
    uc_ratio = calculate_uppercase_ratio(review_text)
    high_uc = int(uc_ratio >= 0.30)
    rep_exc = int(bool(re.search(r"!{2,}", review_text)))
    rep_que = int(bool(re.search(r"\?{2,}", review_text)))
    rep_punc = int(rep_exc or rep_que)

    url_pat = r"(http|https|www\.|\.com|\.net|\.org)"
    email_pat = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
    strict_terms = [
        "visit", "website", "click", "subscribe", "email", "link", "deal", "win",
        "prize", "promo", "promotion", "discount", "limited offer", "claim now", "sign up",
    ]
    weak_terms = ["free", "download", "offer", "contact"]

    url_flag = int(bool(re.search(url_pat, review_text, re.IGNORECASE)))
    email_flag = int(bool(re.search(email_pat, review_text, re.IGNORECASE)))
    strict_flag = int(bool(re.search(r"\b(" + "|".join(strict_terms) + r")\b", review_text, re.IGNORECASE)))
    weak_flag = int(bool(re.search(r"\b(" + "|".join(weak_terms) + r")\b", review_text, re.IGNORECASE)))

    external_contact_or_promo = int(url_flag or email_flag or strict_flag)
    repetition_ratio = repeated_word_ratio(review_text)
    high_rep = int(repetition_ratio >= 0.50)
    rating_deviation = rating - app_avg
    rating_outlier = int(abs(rating_deviation) >= 2)
    rating_sentiment = assign_rating_sentiment(rating)
    text_sentiment = simple_lexicon_sentiment(review_text)
    conflict = detect_rating_text_conflict(rating_sentiment, text_sentiment)

    score = (
        conflict * 2
        + external_contact_or_promo * 2
        + rating_outlier
        + high_rep
        + very_short
        + very_long
        + high_uc
        + rep_punc
    )

    level = "High Risk" if score >= 4 else "Medium Risk" if score >= 2 else "Low Risk"

    reasons = []
    if conflict:
        reasons.append("Rating–text sentiment conflict")
    if external_contact_or_promo:
        reasons.append("External promotional or contact signal")
    if rating_outlier:
        reasons.append("Rating strongly deviates from app average")
    if high_rep:
        reasons.append("High word repetition ratio")
    if very_short:
        reasons.append("Very short review ≤ 5 words")
    if very_long:
        reasons.append("Very long review ≥ 214 words")
    if high_uc:
        reasons.append("High uppercase letter ratio")
    if rep_punc:
        reasons.append("Repeated punctuation")
    if weak_flag and reasons:
        reasons.append("Weak commercial app-review terms detected")
    if not reasons:
        reasons.append("No major spam-risk signals detected")

    features = {
        "word_count": word_count,
        "uppercase_ratio": round(uc_ratio, 3),
        "repeated_word_ratio": round(repetition_ratio, 3),
        "rating_sentiment": rating_sentiment,
        "text_sentiment": text_sentiment,
        "rating_deviation": round(rating_deviation, 3),
        "strict_promo_flag": strict_flag,
        "weak_commercial_flag": weak_flag,
    }

    return level, score, reasons, features


def hero(tag, title, desc):
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero-tag">▸ &nbsp;{tag}</div>
            <h1>{title}</h1>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1.5rem 0;">
            <div style="font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;color:#00d4aa;letter-spacing:0.05em;">ABSA · Group 9</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:2px;letter-spacing:0.04em;">CDS6344 Social Media Computing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🔍  ABSA Predictor",
            "🚨  Spam Detector",
            "📊  Dashboard",
            "🤖  Model Summary",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:0.75rem;">Project Snapshot</div>
        """,
        unsafe_allow_html=True,
    )

    snapshot = [
        ("Dataset", "AWARE SNS"),
        ("Review Rows", "1,615"),
        ("Aspect Rows", "3,097"),
        ("Apps", "6"),
        ("Best Model", "RoBERTa"),
        ("Accuracy", FINAL_ACCURACY),
    ]
    for label, val in snapshot:
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.35rem 0;border-bottom:1px solid rgba(0,212,170,0.07);">
                <span style="font-size:0.78rem;color:#64748b;">{label}</span>
                <span style="font-size:0.78rem;font-weight:600;color:#e2e8f0;">{val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.7rem;color:#334155;text-align:center;">MMU · Faculty of Computing<br>Group 9 · 2025/2026</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if "Overview" in page:
    hero(
        "CDS6344 · Social Media Computing",
        "Aspect-Based Sentiment Analysis<br>of Social Networking App Reviews",
        "An end-to-end NLP pipeline combining traditional machine learning, BiLSTM, transformer models, opinion mining, ABSA insights, and opinion spam-risk detection.",
    )

    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-label">Review Dataset</div><div class="stat-value">1,615</div><div class="stat-sub">Unique app reviews</div></div>
            <div class="stat-card"><div class="stat-label">Aspect Dataset</div><div class="stat-value">3,097</div><div class="stat-sub">Aspect-level samples</div></div>
            <div class="stat-card"><div class="stat-label">Social Apps</div><div class="stat-value">6</div><div class="stat-sub">WhatsApp · Discord · Facebook · more</div></div>
            <div class="stat-card"><div class="stat-label">Best Accuracy</div><div class="stat-value">{FINAL_ACCURACY}</div><div class="stat-sub">Fine-tuned Sentiment RoBERTa</div></div>
            <div class="stat-card"><div class="stat-label">Macro F1</div><div class="stat-value">{FINAL_MACRO_F1}</div><div class="stat-sub">Final aspect-level model</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-label">▸ Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Project Workflow</div>', unsafe_allow_html=True)
        steps = ["Dataset", "Preprocessing", "EDA", "Opinion Mining", "ABSA", "ML", "BiLSTM", "Transformers", "Spam Detection", "App"]
        html = '<div class="pipeline">'
        for i, step in enumerate(steps):
            html += f'<div class="pipeline-step">{step}</div>'
            if i < len(steps) - 1:
                html += '<div class="pipeline-arrow">→</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AWARE Social Networking App Reviews</div>', unsafe_allow_html=True)
        dataset_info = [
            ("Domain", "Social networking app reviews"),
            ("Review-level task", "Rating-based Positive / Neutral / Negative"),
            ("Aspect-level task", "Positive / Negative sentiment classification"),
            ("Aspect categories", "Usability, Security, Cost, Reliability, Safety, and more"),
            ("Spam detection", "Rule-based spam-risk scoring with duplicate and conflict features"),
        ]
        for key, val in dataset_info:
            st.markdown(
                f"""
                <div style="padding:0.55rem 0;border-bottom:1px solid rgba(0,212,170,0.08);">
                    <span style="font-size:0.73rem;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;">{key}</span><br>
                    <span style="font-size:0.92rem;color:#e2e8f0;font-weight:500;">{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="section-label">▸ Team</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Group 9 Members</div>', unsafe_allow_html=True)
        members = [("VN", "Venggadanaathan", "1231303562"), ("TT", "Tharraniah Tamilwanan", "1211111799")]
        for initials, name, sid in members:
            st.markdown(
                f"""
                <div class="member-card">
                    <div class="member-avatar">{initials}</div>
                    <div><div class="member-name">{name}</div><div class="member-id">ID: {sid} · Group 9</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Best Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Final Performance</div>', unsafe_allow_html=True)
        if best_model_df is not None:
            st.dataframe(best_model_df, use_container_width=True, hide_index=True)
        else:
            for metric, value, color in [("Accuracy", FINAL_ACCURACY, "#00d4aa"), ("Macro F1", FINAL_MACRO_F1, "#7c3aed"), ("Weighted F1", FINAL_WEIGHTED_F1, "#f59e0b")]:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0.75rem;background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:0.4rem;">
                        <span style="font-size:0.82rem;color:#94a3b8;">{metric}</span>
                        <span style="font-family:'Space Mono',monospace;font-weight:700;font-size:1rem;color:{color};">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# PAGE 2: ABSA PREDICTOR
# ============================================================

elif "ABSA" in page:
    hero(
        "Interactive Model Demo",
        "ABSA Sentiment Predictor",
        "Enter a review sentence, aspect category, and aspect term. The app loads the local fine-tuned Sentiment RoBERTa model when available, and falls back to the Hugging Face sentiment-pretrained RoBERTa model if the local checkpoint is missing.",
    )

    if ABSA_MODEL_READY:
        if absa_model_source == "local_finetuned":
            st.markdown(
                f"""
                <div style="background:rgba(0,212,170,0.07);border:1px solid rgba(0,212,170,0.22);border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:1.5rem;font-size:0.84rem;color:#94a3b8;">
                    ✅ &nbsp;Using <strong style="color:#00d4aa;">local fine-tuned Sentiment RoBERTa model</strong> from <code>model/sentiment_roberta_finetuned/</code>.<br>
                    Final evaluated performance: Accuracy {FINAL_ACCURACY}, Macro F1 {FINAL_MACRO_F1}, Weighted F1 {FINAL_WEIGHTED_F1}.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.25);border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:1.5rem;font-size:0.84rem;color:#e2e8f0;">
                    ⚠️ &nbsp;Local fine-tuned model not found. Using Hugging Face sentiment-pretrained RoBERTa fallback.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.error("ABSA model could not be loaded. Check requirements and model folder.")
        if ABSA_LOAD_ERROR:
            st.code(ABSA_LOAD_ERROR)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-label">▸ Input</div>', unsafe_allow_html=True)
        sentence = st.text_area(
            "Review sentence",
            value="The app is useful but the notification system is very annoying.",
            height=105,
        )
        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox(
                "Aspect category",
                ["usability", "effectiveness", "efficiency", "reliability", "security", "cost", "safety", "enjoyability", "general"],
            )
        with c2:
            term = st.text_input("Aspect term", value="notification")
        predict_btn = st.button("▶  Predict Aspect Sentiment", use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">▸ Input Format</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transformer Prompt Structure</div>', unsafe_allow_html=True)
        preview = f"Review sentence: {sentence[:65]}{'…' if len(sentence) > 65 else ''}\nAspect category: {category}\nAspect term: {term}"
        st.code(preview)
        st.caption("The fine-tuned model was trained using this prompt-style input format.")

    if predict_btn and sentence.strip():
        if not ABSA_MODEL_READY:
            st.error("Model is not available. Please install dependencies and check model folder.")
        else:
            with st.spinner("Running Sentiment RoBERTa prediction..."):
                result = roberta_absa_predict(sentence, category, term)

            prediction = result["prediction"]
            confidence = result["confidence"]
            css_class = "result-positive" if prediction == "positive" else "result-negative"
            emoji = "✓" if prediction == "positive" else "✗"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">▸ Results</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <div class="result-label">Predicted Sentiment</div>
                        <div class="result-value">{emoji} &nbsp;{prediction.upper()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="result-positive" style="background:rgba(0,212,170,0.08);border-color:rgba(0,212,170,0.3);">
                        <div class="result-label" style="color:#00d4aa;">Confidence</div>
                        <div class="result-value" style="color:#00d4aa;">{confidence:.1%}</div>
                        <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{confidence*100:.0f}%;"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                source_label = "Local fine-tuned" if result["source"] == "local_finetuned" else "HF fallback"
                threshold_text = f"{result['threshold']:.2f}" if result["threshold"] is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="info-card" style="margin:0;min-height:112px;">
                        <div class="result-label" style="color:#64748b;">Model Source</div>
                        <div style="font-size:0.86rem;color:#e2e8f0;margin-top:0.35rem;font-weight:600;">{source_label}</div>
                        <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">Threshold: <strong>{threshold_text}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-label">▸ Probability Scores</div>', unsafe_allow_html=True)
                score_rows = [
                    ("Positive", result["positive_score"], "#10b981"),
                    ("Neutral", result["neutral_score"], "#f59e0b"),
                    ("Negative", result["negative_score"], "#ef4444"),
                ]
                for label, score, color in score_rows:
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;padding:0.45rem 0;border-bottom:1px solid rgba(0,212,170,0.07);">
                            <span style="font-size:0.84rem;color:#94a3b8;">{label}</span>
                            <span style="font-family:'Space Mono',monospace;color:{color};font-weight:700;">{score:.3f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown('<div class="section-label">▸ Model Input</div>', unsafe_allow_html=True)
                st.code(result["model_input"])


# ============================================================
# PAGE 3: SPAM DETECTOR
# ============================================================

elif "Spam" in page:
    hero(
        "Opinion Spam Detection",
        "Spam-Risk Detector",
        "Paste a review and set its rating to receive a rule-based spam-risk assessment. High risk means manual inspection is required, not confirmed spam.",
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-label">▸ Input</div>', unsafe_allow_html=True)
        review_text = st.text_area(
            "Full review text",
            value="Unbelievable!!! This app keeps asking me to click links and it does not work properly.",
            height=120,
        )
        c1, c2 = st.columns(2)
        with c1:
            rating = st.slider("Review rating", 1, 5, 1)
        with c2:
            app_avg = st.number_input("App average rating", 1.0, 5.0, 3.22, 0.01)
        check_btn = st.button("▶  Check Spam Risk", use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">▸ Detection Signals</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">8 Risk Signals</div>', unsafe_allow_html=True)
        signals = [
            ("🔴", "Rating–text conflict", "×2"),
            ("🔴", "External promo/contact", "×2"),
            ("🟡", "Rating deviation", "×1"),
            ("🟡", "Word repetition", "×1"),
            ("🟡", "Very short review", "×1"),
            ("🟡", "Very long review", "×1"),
            ("🟡", "High uppercase", "×1"),
            ("🟡", "Repeated punctuation", "×1"),
        ]
        for icon, name, weight in signals:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0.5rem;border-bottom:1px solid rgba(0,212,170,0.07);">
                    <span style="font-size:0.82rem;color:#94a3b8;">{icon} &nbsp;{name}</span>
                    <span style="font-size:0.72rem;color:#64748b;font-family:'Space Mono',monospace;">{weight}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if check_btn and review_text.strip():
        level, score, reasons, features = spam_risk_detector(review_text, rating, app_avg)
        css = {"High Risk": "result-negative", "Medium Risk": "result-warning", "Low Risk": "result-positive"}
        icons = {"High Risk": "⚠", "Medium Risk": "!", "Low Risk": "✓"}

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Assessment Results</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                f"""
                <div class="{css[level]}">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value">{icons[level]} &nbsp;{level.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            pct = min(score / 10, 1.0)
            color = "#ef4444" if level == "High Risk" else "#f59e0b" if level == "Medium Risk" else "#10b981"
            st.markdown(
                f"""
                <div class="info-card" style="margin:0;min-height:112px;">
                    <div class="result-label" style="color:#64748b;">Risk Score</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:{color};">{score} <span style="font-size:0.8rem;color:#64748b;">/ 10</span></div>
                    <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{pct*100:.0f}%;background:linear-gradient(90deg,{color},{color}88);"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="info-card" style="margin:0;min-height:112px;">
                    <div class="result-label" style="color:#64748b;">Sentiment Check</div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.35rem;">Rating-based: <strong>{features['rating_sentiment']}</strong></div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">Text-based: <strong>{features['text_sentiment']}</strong></div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">Deviation: <strong>{features['rating_deviation']:+.2f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">▸ Triggered Signals</div>', unsafe_allow_html=True)
            tag_class = "danger" if level == "High Risk" else "warn" if level == "Medium Risk" else ""
            tags = '<div class="tag-grid">' + "".join([f'<span class="tag {tag_class}">{r}</span>' for r in reasons]) + "</div>"
            st.markdown(tags, unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-label">▸ Feature Breakdown</div>', unsafe_allow_html=True)
            rows = [
                ("Word count", features["word_count"]),
                ("Uppercase ratio", f"{features['uppercase_ratio']:.1%}"),
                ("Repetition ratio", f"{features['repeated_word_ratio']:.1%}"),
                ("Strict promo flag", "Yes" if features["strict_promo_flag"] else "No"),
                ("Weak commercial flag", "Yes" if features["weak_commercial_flag"] else "No"),
            ]
            for key, value in rows:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid rgba(0,212,170,0.06);">
                        <span style="font-size:0.8rem;color:#64748b;">{key}</span>
                        <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#e2e8f0;">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div style="margin-top:1rem;font-size:0.72rem;color:#94a3b8;background:rgba(0,0,0,0.20);border-radius:8px;padding:0.6rem 0.9rem;">
                ⚠️ High Risk does not confirm spam. It flags the review for manual inspection only.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 4: DASHBOARD
# ============================================================

elif "Dashboard" in page:
    hero(
        "Visual Analytics",
        "Project Dashboard",
        "Curated visual outputs from all four notebooks: EDA, opinion mining, model comparison, and spam-risk detection.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sentiment EDA", "🔎 ABSA Insights", "🤖 Model Results", "🚨 Spam Detection"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rating-Based Sentiment Distribution**")
            display_image_if_exists("rating_based_sentiment_distribution.png")
        with c2:
            st.markdown("**Review Length Distribution**")
            display_image_if_exists("review_length_distribution.png")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Aspect Terms — Sentiment Comparison**")
            display_image_if_exists("top_aspect_terms_sentiment_comparison.png")
        with c2:
            st.markdown("**High-Risk Negative Aspect Terms**")
            display_image_if_exists("high_risk_negative_aspect_terms.png")
        st.markdown("**VADER vs Dataset Sentiment**")
        display_image_if_exists("vader_vs_dataset_sentiment_heatmap.png")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Final Aspect-Level Model Comparison**")
            display_image_if_exists("final_updated_aspect_model_comparison_macro_f1.png")
        with c2:
            st.markdown("**Final Best Model Confusion Matrix**")
            display_image_if_exists("final_best_model_confusion_matrix.png")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Spam-Risk Level Distribution**")
            display_image_if_exists("spam_risk_level_distribution.png")
        with c2:
            st.markdown("**High Spam-Risk Percentage by App**")
            display_image_if_exists("high_spam_risk_percentage_by_app.png")
        st.markdown("**Feature Contributions Among High Spam-Risk Reviews**")
        display_image_if_exists("spam_feature_contribution_high_risk.png")


# ============================================================
# PAGE 5: MODEL SUMMARY
# ============================================================

elif "Model" in page:
    hero(
        "Model Evaluation",
        "Model Summary & Comparison",
        "Performance breakdown across traditional ML, BiLSTM, DistilBERT, RoBERTa, and the final fine-tuned Sentiment RoBERTa model.",
    )

    st.markdown('<div class="section-label">▸ Final Best Model</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-label">Model</div><div class="stat-value" style="font-size:1.02rem;line-height:1.35;">Sentiment<br>RoBERTa</div><div class="stat-sub">Threshold tuning</div></div>
            <div class="stat-card"><div class="stat-label">Accuracy</div><div class="stat-value">{FINAL_ACCURACY}</div><div class="stat-sub">Test set</div></div>
            <div class="stat-card"><div class="stat-label">Macro F1</div><div class="stat-value">{FINAL_MACRO_F1}</div><div class="stat-sub">Main selection metric</div></div>
            <div class="stat-card"><div class="stat-label">Weighted F1</div><div class="stat-value">{FINAL_WEIGHTED_F1}</div><div class="stat-sub">Class-weighted performance</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">▸ All Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aspect-Level Model Comparison</div>', unsafe_allow_html=True)
    if model_comparison_df is not None:
        st.dataframe(model_comparison_df, use_container_width=True, hide_index=True)
        if "Macro F1" in model_comparison_df.columns and "Model" in model_comparison_df.columns:
            chart_df = model_comparison_df[["Model", "Macro F1"]].copy().sort_values("Macro F1", ascending=True)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#111827")
            colors = ["#00d4aa" if m == chart_df.iloc[-1]["Model"] else "#1e3a4a" for m in chart_df["Model"]]
            bars = ax.barh(chart_df["Model"], chart_df["Macro F1"], color=colors, height=0.62)
            ax.set_xlabel("Macro F1-score", color="#94a3b8")
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.xaxis.grid(True, color="#1e293b", linewidth=0.8)
            ax.set_axisbelow(True)
            for bar in bars:
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", color="#e2e8f0", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Model comparison CSV not found in data/ folder.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">▸ Spam Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Spam-Risk Findings</div>', unsafe_allow_html=True)
    if spam_findings_df is not None:
        st.dataframe(spam_findings_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Spam findings CSV not found in data/ folder.")

    if review_spam_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Dataset Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Spam-Risk Labelled Reviews</div>', unsafe_allow_html=True)
        st.dataframe(review_spam_df.head(20), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 6: ABOUT
# ============================================================

elif "About" in page:
    hero(
        "Documentation",
        "About This Application",
        "Technical documentation covering the project scope, model architecture, spam-risk methodology, and deployment notes.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">▸ Project</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Scope & Objectives</div>', unsafe_allow_html=True)
        items = [
            ("Subject", "CDS6344 Social Media Computing"),
            ("Dataset", "AWARE Social Networking App Reviews"),
            ("Review rows", "1,615"),
            ("Aspect rows", "3,097"),
            ("Task", "Aspect-Based Sentiment Analysis"),
            ("Add-on", "Opinion Spam-Risk Detection"),
            ("Final model", FINAL_MODEL_NAME),
        ]
        for key, val in items:
            st.markdown(
                f"""
                <div style="display:flex;gap:0.75rem;padding:0.55rem 0;border-bottom:1px solid rgba(0,212,170,0.07);">
                    <span style="font-size:0.72rem;color:#64748b;min-width:92px;text-transform:uppercase;letter-spacing:0.07em;">{key}</span>
                    <span style="font-size:0.86rem;color:#e2e8f0;">{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Local Model Status</div>', unsafe_allow_html=True)
        status = "Loaded local fine-tuned model" if absa_model_source == "local_finetuned" else "Using fallback / model missing"
        st.markdown(f"**{status}**")
        st.code(LOCAL_MODEL_DIR)

    with c2:
        st.markdown('<div class="section-label">▸ Spam Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
        methods = [
            ("01", "Exact duplicate detection", "Identify identical reviews"),
            ("02", "Near-duplicate detection", "TF-IDF cosine similarity"),
            ("03", "Rating-text conflict", "Compare rating sentiment and lexicon sentiment"),
            ("04", "Rating deviation", "Flag outliers versus app average"),
            ("05", "Repeated punctuation", "Detect !! and ?? patterns"),
            ("06", "Strict promotional signals", "URLs, email, external-action terms"),
            ("07", "Weak commercial terms", "Tracked separately to reduce false positives"),
            ("08", "Uppercase ratio", "Flag excessive capitalization"),
        ]
        for num, key, val in methods:
            st.markdown(
                f"""
                <div style="display:flex;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid rgba(0,212,170,0.07);">
                    <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#00d4aa;min-width:24px;">{num}</span>
                    <div><div style="font-size:0.84rem;color:#e2e8f0;font-weight:500;">{key}</div><div style="font-size:0.75rem;color:#64748b;">{val}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Tech Stack</div>', unsafe_allow_html=True)
        tags = ["Python", "Streamlit", "Pandas", "PyTorch", "Transformers", "RoBERTa", "Matplotlib", "Google Colab"]
        html = '<div class="tag-grid">' + "".join([f'<span class="tag">{t}</span>' for t in tags]) + "</div>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown(
            """
            <div style="margin-top:1rem;background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.15);border-radius:10px;padding:1rem 1.25rem;font-size:0.82rem;color:#94a3b8;line-height:1.7;">
                <strong style="color:#00d4aa;">⚠ Disclaimer</strong><br>
                Spam-risk labels are not confirmed spam judgements. High-risk reviews require manual inspection before action is taken.
            </div>
            """,
            unsafe_allow_html=True,
        )
