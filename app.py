"""
app.py
------
Streamlit Web Application for Spam Message Detection.
Course: DCS 404 - Artificial Intelligence & Machine Learning (Westcliff University)
Final Project: Machine Learning-Based Spam Message Detection

Features the complete updated pipeline:
1. Home
2. Dataset Overview
3. Data Exploration
4. Spam and Ham Word Clouds
5. Model Performance
6. Model Comparison
7. Live Spam Prediction (Logistic Regression, Naive Bayes, DistilBERT, DistilRoBERTa)
"""

import os
import time
import json
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Spam Message Detection Benchmark | DCS 404",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Global typography and surface */
    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    *, *::before, *::after {
        border-radius: 0px !important;
    }

    .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #000000 !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #000000 !important;
        margin: 1.5rem 0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif !important;
        font-weight: 700 !important;
        color: #000000 !important;
        letter-spacing: -0.03em !important;
        text-align: left !important;
        margin-top: 0 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        padding: 0.7rem 1.4rem !important;
        box-shadow: none !important;
        transition: none !important;
    }

    .stButton > button:hover {
        background-color: #D32F2F !important;
        border-color: #D32F2F !important;
        color: #FFFFFF !important;
    }

    /* Form inputs */
    .stTextArea textarea {
        border: 1px solid #000000 !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        padding: 12px !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid #D32F2F !important;
        box-shadow: none !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        border: 1px solid #000000 !important;
        background-color: #FFFFFF !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #000000 !important;
    }

    /* Custom Layout utilities */
    .header-banner {
        border-bottom: 2px solid #000000;
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
    }

    .sub-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #444444;
        margin-bottom: 0.35rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.04em;
        color: #000000;
        margin: 0 0 0.5rem 0;
    }

    .deck-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #222222;
        max-width: 900px;
        margin: 0;
    }

    .card {
        border: 1px solid #000000;
        background-color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .card-neutral {
        border: 1px solid #000000;
        background-color: #F8F9FA;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 1px solid #000000;
        padding-bottom: 0.35rem;
        margin-bottom: 1rem;
        color: #000000;
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #E5E7EB;
        font-size: 0.82rem;
    }

    .spec-label {
        font-weight: 600;
        color: #444444;
    }

    .spec-value {
        font-weight: 700;
        color: #000000;
        font-variant-numeric: tabular-nums;
        text-align: right;
    }

    .result-spam {
        border: 2px solid #D32F2F;
        background-color: #FFF5F5;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .result-ham {
        border: 2px solid #000000;
        background-color: #F8F9FA;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .tag-spam {
        display: inline-block;
        background-color: #D32F2F;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 5px 12px;
        margin-bottom: 0.75rem;
    }

    .tag-ham {
        display: inline-block;
        background-color: #000000;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 5px 12px;
        margin-bottom: 0.75rem;
    }

    .big-stat {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.04em;
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.5rem;
    }

    .image-box {
        border: 1px solid #000000;
        padding: 8px;
        background-color: #FFFFFF;
        margin-bottom: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ASSET_DIR = os.path.join(BASE_DIR, 'assets')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
TRANSFORMER_DIR = os.path.join(MODEL_DIR, 'transformer_models')


def get_image_path(filename):
    """Find image in assets/ or fallback to figures/."""
    p_assets = os.path.join(ASSET_DIR, filename)
    if os.path.exists(p_assets):
        return p_assets
    p_fig = os.path.join(FIG_DIR, filename)
    if os.path.exists(p_fig):
        return p_fig
    return None


# Loaders
@st.cache_resource
def load_traditional_artifacts():
    """Load TF-IDF vectorizer and scikit-learn models."""
    artifacts = {}
    
    # Vectorizer
    vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    if not os.path.exists(vec_path):
        vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    if os.path.exists(vec_path):
        artifacts['tfidf'] = joblib.load(vec_path)
    
    # Logistic Regression
    lr_path = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
    if not os.path.exists(lr_path):
        lr_path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
    if os.path.exists(lr_path):
        artifacts['lr'] = joblib.load(lr_path)

    # Naive Bayes
    nb_path = os.path.join(MODEL_DIR, 'naive_bayes.pkl')
    if not os.path.exists(nb_path):
        nb_path = os.path.join(MODEL_DIR, 'naive_bayes.joblib')
    if os.path.exists(nb_path):
        artifacts['nb'] = joblib.load(nb_path)

    return artifacts


@st.cache_resource
def load_transformer_pipeline(model_subpath):
    """Load Hugging Face tokenizer and model on demand."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        target_dir = os.path.join(TRANSFORMER_DIR, model_subpath)
        if os.path.exists(target_dir):
            tokenizer = AutoTokenizer.from_pretrained(target_dir)
            model = AutoModelForSequenceClassification.from_pretrained(target_dir)
            model.eval()
            return tokenizer, model
    except Exception as e:
        st.warning(f"Transformer loading exception: {e}")
    return None, None


@st.cache_data
def load_metadata():
    meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_sample_dataset():
    csv_path = os.path.join(DATA_DIR, 'spam_messages.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raw_path = os.path.join(DATA_DIR, 'SMSSpamCollection')
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, sep='\t', header=None, names=['label', 'text']).dropna().drop_duplicates()
        return df
    return None


trad_artifacts = load_traditional_artifacts()
metadata = load_metadata()
df_sample = load_sample_dataset()

# Sidebar Navigation (The 7 Suggested Sections)
SECTIONS = [
    "1. Home",
    "2. Dataset Overview",
    "3. Data Exploration",
    "4. Spam and Ham Word Clouds",
    "5. Model Performance",
    "6. Model Comparison",
    "7. Live Spam Prediction"
]

with st.sidebar:
    st.markdown('<div class="sub-tag">PROJECT NAVIGATION</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 1.25rem; margin-bottom: 1rem;">Spam Detection</h2>', unsafe_allow_html=True)
    
    active_section = st.radio(
        "Select Section:",
        options=SECTIONS,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Pipeline Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="spec-row"><span class="spec-label">Dataset</span><span class="spec-value">UCI SMS Spam</span></div>
    <div class="spec-row"><span class="spec-label">Total Records</span><span class="spec-value">5,169 Clean</span></div>
    <div class="spec-row"><span class="spec-label">Class Ratio</span><span class="spec-value">87.4% Ham / 12.6% Spam</span></div>
    <div class="spec-row"><span class="spec-label">Split</span><span class="spec-value">80% Train / 20% Test</span></div>
    <div class="spec-row"><span class="spec-label">Balancing</span><span class="spec-value">SMOTE on Train Set</span></div>
    <div class="spec-row"><span class="spec-label">Features</span><span class="spec-value">5,000 TF-IDF N-Grams</span></div>
    <div class="spec-row"><span class="spec-label">Transformers</span><span class="spec-value">DistilBERT & RoBERTa</span></div>
    """, unsafe_allow_html=True)

    if metadata and "best_model" in metadata:
        st.markdown(f"""
        <div class="card-neutral" style="margin-top: 1.25rem; padding: 0.75rem 1rem;">
            <div class="sub-tag">Top Benchmark Model</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #000000;">{metadata['best_model']}</div>
            <div style="font-size: 0.8rem; color: #444444; margin-top: 4px;">F1-Score: 95.4% | Accuracy: 98.8%</div>
        </div>
        """, unsafe_allow_html=True)

# Main Application Layout
st.markdown("""
<div class="header-banner">
    <div class="sub-tag">WESTCLIFF UNIVERSITY · DCS 404 ARTIFICIAL INTELLIGENCE & MACHINE LEARNING</div>
    <h1 class="main-title">Spam Message Detection System</h1>
    <p class="deck-text">
        An end-to-end natural language processing benchmark comparing SMOTE-balanced traditional machine learning 
        (Logistic Regression, Naive Bayes) against fine-tuned deep bidirectional transformers (DistilBERT, DistilRoBERTa)
        on the UCI SMS Spam Collection.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: HOME
# ==============================================================================
if active_section == "1. Home":
    st.markdown('<div class="sub-tag">01 / OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<h2>Project Architecture & Methodology</h2>', unsafe_allow_html=True)
    
    col_intro1, col_intro2 = st.columns([3, 2], gap="large")
    
    with col_intro1:
        st.markdown("""
        ### Research Background & Motivation
        Unsolicited SMS communications (**Spam**) represent significant security, privacy, and economic challenges, including:
        - **Credential Phishing & Smishing:** Deceptive URLs impersonating financial institutions and delivery providers.
        - **Financial Advance-Fee Scams:** Fake lottery prizes, cash awards, and bogus loan approvals.
        - **Network Abuse:** Bulk unsolicited advertising degrading telecommunications reliability.

        Traditional spam filters relied on brittle keyword blacklists. Modern attackers actively evade these rules through typographical permutations. 
        This project synthesizes **statistical NLP feature extraction (TF-IDF)**, **synthetic oversampling (SMOTE)**, and **state-of-the-art transformer contextual representations** to build a robust, generalizable classifier.
        """)

        st.markdown("""
        ### The End-to-End Pipeline Workflow
        ```text
        UCI SMS Spam Collection (5,169 Messages)
                       ↓
            Text Preprocessing & Cleaning
                       ↓
          Stratified 80/20 Train-Test Split (Anti-Leakage)
                       ↓
          TF-IDF Vectorization (Fit ONLY on Training Set)
                       ↓
          SMOTE Oversampling (Applied ONLY to Training Set)
         ┌─────────────────────────┴─────────────────────────┐
         ↓                                                   ↓
   Traditional Machine Learning                       Deep Transformers
   - Logistic Regression (SMOTE)                      - DistilBERT (Contextual)
   - Multinomial Naive Bayes                          - DistilRoBERTa (Contextual)
         └─────────────────────────┬─────────────────────────┘
                       ↓
         Held-Out Test Set Evaluation (Original 12.6% Prior)
                       ↓
         Streamlit Interactive Dashboard & Live Inference
        ```
        """)

    with col_intro2:
        st.markdown('<div class="card-neutral">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Methodological Pillars</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. Strict Data Leakage Prevention:**  
        The vectorizer is fit strictly on training text. The test partition remains untouched to evaluate real-world generalizability.

        **2. SMOTE Imbalance Remediation:**  
        The minority Spam class represents only 12.6% of messages. SMOTE synthesizes vectors along feature interpolations in training space to boost minority recall without overfitting.

        **3. Bayesian Prior Analysis:**  
        We demonstrate why applying SMOTE to Multinomial Naive Bayes damages precision due to distortion of the Bayes class prior $P(y)$.

        **4. Dual Paradigm Comparison:**  
        We contrast the microsecond execution of Logistic Regression against the contextual semantic superiority of Transformers.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Proceed to Live Spam Prediction →", type="primary"):
            st.session_state["nav_select"] = "7. Live Spam Prediction"
            st.rerun()


# ==============================================================================
# SECTION 2: DATASET OVERVIEW
# ==============================================================================
elif active_section == "2. Dataset Overview":
    st.markdown('<div class="sub-tag">02 / CORPUS SPECIFICATION</div>', unsafe_allow_html=True)
    st.markdown('<h2>UCI SMS Spam Collection Overview</h2>', unsafe_allow_html=True)

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown("""
        <div class="card">
            <div class="sub-tag">Total Messages</div>
            <div class="big-stat">5,169</div>
            <div style="font-size: 0.8rem; color: #555;">Deduplicated records</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown("""
        <div class="card">
            <div class="sub-tag">Legitimate (Ham)</div>
            <div class="big-stat" style="color: #000000;">4,516</div>
            <div style="font-size: 0.8rem; color: #555;">87.37% majority class</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown("""
        <div class="card">
            <div class="sub-tag">Spam Messages</div>
            <div class="big-stat" style="color: #D32F2F;">653</div>
            <div style="font-size: 0.8rem; color: #D32F2F;">12.63% minority class</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat4:
        st.markdown("""
        <div class="card">
            <div class="sub-tag">Held-Out Test Partition</div>
            <div class="big-stat">1,034</div>
            <div style="font-size: 0.8rem; color: #555;">20% Stratified split</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### Corpus Description
    The **UCI SMS Spam Collection** is a benchmark dataset of mobile text messages compiled by Tiago A. Almeida and José María Gómez Hidalgo. 
    It incorporates messages from:
    1. **Grumbled SMS:** A UK forum of mobile phone users complaining about unsolicited messages.
    2. **NUS SMS Corpus:** Academic conversational text collected from National University of Singapore students.
    3. **Caroline Tag's PhD Thesis:** A collection of authentic British English interpersonal mobile messages.
    4. **SMS Spam Corpus v.0.1 Big:** 1,002 SMS spam messages collected from public UK mobile forums.
    """)

    if df_sample is not None:
        st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Interactive Data Preview</div>', unsafe_allow_html=True)
        col_flt1, col_flt2 = st.columns([1, 3])
        with col_flt1:
            label_filter = st.selectbox("Filter Category:", ["All", "spam", "ham"])
        with col_flt2:
            search_query = st.text_input("Search message content:", placeholder="Type a keyword to filter messages...")

        filtered_df = df_sample.copy()
        if label_filter != "All":
            filtered_df = filtered_df[filtered_df['label'] == label_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df['text'].str.contains(search_query, case=False, na=False)]

        st.dataframe(filtered_df[['label', 'text']].head(25), use_container_width=True, hide_index=True)
        st.caption(f"Displaying top matching records out of {len(filtered_df):,} filtered rows.")


# ==============================================================================
# SECTION 3: DATA EXPLORATION
# ==============================================================================
elif active_section == "3. Data Exploration":
    st.markdown('<div class="sub-tag">03 / EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<h2>Distributions & SMOTE Imbalance Remediation</h2>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        st.markdown('<div class="section-title">Raw Class Distribution</div>', unsafe_allow_html=True)
        p_class = get_image_path("class_distribution.png")
        if p_class:
            st.image(p_class, caption="Cleaned UCI SMS dataset class proportions (87.4% Ham vs 12.6% Spam)", use_container_width=True)
        st.markdown("""
        **Severe Class Skew:**  
        The severe ~7:1 class imbalance poses an acute machine learning challenge: a naive "dummy classifier" predicting *Ham* 100% of the time achieves **87.4% Accuracy**, yet captures **0% of spam attacks**. 
        Thus, Precision, Recall, and F1-Score are the vital evaluation metrics.
        """)

    with col_e2:
        st.markdown('<div class="section-title">Message Character Length Distribution</div>', unsafe_allow_html=True)
        p_len = get_image_path("message_length_distribution.png")
        if p_len:
            st.image(p_len, caption="Character length density: Ham (conversational, short) vs Spam (dense, saturated)", use_container_width=True)
        st.markdown("""
        **Length Bimodality:**  
        - **Legitimate (Ham) messages:** Skewed toward short exchanges (mean length ~54 characters).
        - **Spam messages:** Clustered near the standard 160-character mobile SMS billing limit (mean length ~138 characters), designed to cram marketing hooks, phone numbers, and urgent calls-to-action into a single SMS credit.
        """)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Class Balancing with SMOTE (Synthetic Minority Over-sampling Technique)</div>', unsafe_allow_html=True)

    p_smote = get_image_path("smote_class_distribution.png")
    if p_smote:
        st.image(p_smote, caption="Training partition before and after SMOTE balancing (50% Ham / 50% Spam)", use_container_width=True)

    col_s1, col_s2 = st.columns(2, gap="large")
    with col_s1:
        st.markdown(r"""
        ### How SMOTE Works
        Instead of duplicating existing minority examples (random oversampling, which causes severe decision-boundary overfitting), **SMOTE** (Chawla et al., 2002) synthesizes new minority points:
        1. For each spam instance $\mathbf{x}_i$, find its $k$-nearest neighbors in the 5,000-dimensional TF-IDF space.
        2. Select a neighbor $\mathbf{x}_{zi}$ randomly.
        3. Synthesize a new sample: 
           $$\mathbf{x}_{new} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$
        This creates continuous feature paths across the minority domain, enabling linear classifiers like Logistic Regression to construct balanced decision boundaries.
        """)
    with col_s2:
        st.markdown("""
        ### Critical Methodological Rules Applied
        > **Rule 1: Fit TF-IDF Only on Training Data**  
        > We split the raw dataset 80/20 first. The TF-IDF vectorizer is fit strictly on the 4,135 training samples. This ensures vocabulary and document frequency from the test set never leak into feature extraction.

        > **Rule 2: Apply SMOTE Only to Training Features**  
        > Synthetic interpolation is performed strictly on the training feature matrix. The 1,034 test samples remain completely untouched, preserving authentic natural prevalence (12.6% spam) for uncompromised validation.
        """)


# ==============================================================================
# SECTION 4: SPAM AND HAM WORD CLOUDS
# ==============================================================================
elif active_section == "4. Spam and Ham Word Clouds":
    st.markdown('<div class="sub-tag">04 / TEXTUAL VISUALIZATIONS</div>', unsafe_allow_html=True)
    st.markdown('<h2>Spam vs. Ham High-Frequency Vocabulary</h2>', unsafe_allow_html=True)

    col_wc1, col_wc2 = st.columns(2, gap="large")

    with col_wc1:
        st.markdown('<div class="section-title">Spam Word Cloud (High-Risk Triggers)</div>', unsafe_allow_html=True)
        p_wc_spam = get_image_path("spam_wordcloud.png")
        if p_wc_spam:
            st.image(p_wc_spam, caption="Spam vocabulary dominated by financial lures, urgency, and telephone codes", use_container_width=True)
        st.markdown("""
        **Characteristic Spam Lexicon:**
        - **Urgency Triggers:** `URGENT`, `NOW`, `TODAY`, `EXPIRES`, `IMMEDIATELY`
        - **Monetary & Reward Hooks:** `FREE`, `CASH`, `PRIZE`, `WIN`, `AWARD`, `£1000`
        - **Call-to-Action Codes:** `CALL`, `TXT`, `REPLY`, `CLAIM`, `STOP`, `88202`, `09061`
        - **Audience Flattery:** `CUSTOMER`, `SELECTED`, `VALUED`, `CONGRATULATIONS`
        """)

    with col_wc2:
        st.markdown('<div class="section-title">Ham Word Cloud (Conversational Terms)</div>', unsafe_allow_html=True)
        p_wc_ham = get_image_path("ham_wordcloud.png")
        if p_wc_ham:
            st.image(p_wc_ham, caption="Ham vocabulary dominated by interpersonal, casual, and temporal terms", use_container_width=True)
        st.markdown("""
        **Characteristic Ham Lexicon:**
        - **Conversational Salutations:** `ok`, `good`, `got`, `come`, `home`, `love`, `see`
        - **Temporal & Scheduling:** `now`, `later`, `night`, `today`, `tomorrow`, `time`
        - **Coordination & Querying:** `going`, `call`, `tell`, `sure`, `know`, `need`, `think`
        - **Personal Pronouns & Slang:** `u`, `ur`, `lor`, `da`, `pls`, `wat`, `dun`
        """)


# ==============================================================================
# SECTION 5: MODEL PERFORMANCE
# ==============================================================================
elif active_section == "5. Model Performance":
    st.markdown('<div class="sub-tag">05 / INDIVIDUAL MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.markdown('<h2>Detailed Performance & Diagnostic Matrices</h2>', unsafe_allow_html=True)

    if metadata and "metrics" in metadata:
        m_list = metadata["metrics"]
        
        tab_lr, tab_nb, tab_db, tab_dr = st.tabs([
            "Logistic Regression (SMOTE)",
            "Multinomial Naive Bayes",
            "DistilBERT (Transformer)",
            "DistilRoBERTa (Transformer)"
        ])

        tabs = [tab_lr, tab_nb, tab_db, tab_dr]
        for idx, m in enumerate(m_list):
            with tabs[idx]:
                st.markdown(f"### {m['Model']} Evaluation")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
                c2.metric("Precision", f"{m['Precision']*100:.2f}%")
                c3.metric("Recall", f"{m['Recall']*100:.2f}%")
                c4.metric("F1-Score", f"{m['F1-Score']*100:.2f}%")
                c5.metric("ROC-AUC", f"{m['ROC-AUC']:.4f}")

                st.markdown(f"""
                <div class="card-neutral" style="margin-top: 1rem;">
                    <div class="spec-row"><span class="spec-label">Architecture Paradigm</span><span class="spec-value">{m['Category']}</span></div>
                    <div class="spec-row"><span class="spec-label">Training Duration</span><span class="spec-value">{m['Train Time (s)']:.2f} seconds</span></div>
                    <div class="spec-row"><span class="spec-label">Inference Latency</span><span class="spec-value">{m['Inference Time (ms/sample)']:.4f} ms per message</span></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2, gap="large")

    with col_cm:
        st.markdown('<div class="section-title">Confusion Matrices (Test Partition, N=1,034)</div>', unsafe_allow_html=True)
        p_cm = get_image_path("confusion_matrices.png")
        if p_cm:
            st.image(p_cm, caption="Confusion matrices across all four models on identical test data", use_container_width=True)

    with col_roc:
        st.markdown('<div class="section-title">Receiver Operating Characteristic (ROC) Curves</div>', unsafe_allow_html=True)
        p_roc = get_image_path("roc_curves.png")
        if p_roc:
            st.image(p_roc, caption="ROC curves illustrating true positive rate vs false positive rate", use_container_width=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Special Analysis: Why SMOTE Harms Multinomial Naive Bayes</div>', unsafe_allow_html=True)
    st.markdown(r"""
    When evaluating SMOTE, machine learning practitioners often mistakenly assume balancing training data helps every algorithm. 
    However, **Multinomial Naive Bayes exhibits a known theoretical failure mode when trained with SMOTE**:

    1. **Bayesian Formulation:**  
       $$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^D P(x_j \mid y)$$
    2. **Class Prior Distortion:**  
       In the authentic population, $P(\text{Spam}) \approx 0.126$. When SMOTE oversamples the training set to 50/50, it artificially forces the prior $P(\text{Spam}) \to 0.500$.
    3. **Empirical Consequence on Held-Out Test Set:**  
       - **Standard Naive Bayes:** **2 False Positives** (Precision = 98.26%, Recall = 86.26%)
       - **SMOTE-Trained Naive Bayes:** **21 False Positives** (Precision plummeted to 85.21%!)
    
    *Conclusion:* Logistic Regression benefits substantially from SMOTE because it optimizes a linear hyper-plane distance. Naive Bayes, being a generative probabilistic model dependent on exact prior frequencies, suffers a massive spike in false alarms when balanced artificially.
    """)


# ==============================================================================
# SECTION 6: MODEL COMPARISON
# ==============================================================================
elif active_section == "6. Model Comparison":
    st.markdown('<div class="sub-tag">06 / COMPARATIVE BENCHMARK</div>', unsafe_allow_html=True)
    st.markdown('<h2>Traditional ML vs. Transformers Benchmark</h2>', unsafe_allow_html=True)

    if metadata and "metrics" in metadata:
        df_bench = pd.DataFrame(metadata["metrics"])
        st.markdown('<div class="section-title">Comparative Performance Table</div>', unsafe_allow_html=True)
        st.dataframe(
            df_bench.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.4f}",
                "Train Time (s)": "{:.2f} s",
                "Inference Time (ms/sample)": "{:.4f} ms"
            }),
            use_container_width=True,
            hide_index=True
        )

    p_comp = get_image_path("model_comparison.png")
    if p_comp:
        st.image(p_comp, caption="Comparative Performance Metric Chart across all evaluated models", use_container_width=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comprehensive Architectural Trade-Off Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    In enterprise engineering, **the "best" model is not merely the one with the highest F1-score**. Engineering trade-offs dictate architecture selection:

    | Evaluation Dimension | Traditional ML (Logistic Regression + SMOTE) | Transformer Models (DistilBERT / DistilRoBERTa) |
    | :--- | :--- | :--- |
    | **Classification Accuracy & F1** | **97.29% Acc / 89.39% F1** (High, highly competitive) | **98.84% Acc / 95.42% F1** (+6.0% F1 improvement) |
    | **Inference Latency** | **0.0008 ms / sample** (~1,250,000 predictions/sec) | **~67 - 73 ms / sample** (~15 predictions/sec on CPU) |
    | **Training Compute Time** | **< 0.1 seconds** on standard CPU | **~45 - 50 minutes** on CPU (~2 mins on GPU) |
    | **Model Artifact Size** | **~220 KB** total disk footprint | **~268 MB - 328 MB** (over 1,200x larger) |
    | **Hardware Constraints** | Runs on edge microcontrollers, Raspberry Pi, serverless functions | Requires substantial RAM (1-2 GB) or GPU acceleration |
    | **Explainability** | High: Direct feature weights map to TF-IDF n-grams | Low: Non-linear multi-head self-attention layers |

    ### When to Deploy Which Model?
    - **Deploy Logistic Regression (SMOTE) When:**
      1. Operating at telecommunications carrier scale (inspecting millions of messages/second in real-time).
      2. Deploying on embedded edge devices or low-cost serverless lambda instances.
      3. Regulatory requirements mandate transparent word-by-word interpretability for every blocked communication.
    - **Deploy DistilBERT / DistilRoBERTa When:**
      1. Dealing with sophisticated adversarial evasion (attackers using synonyms, metaphors, and complex sentence structures).
      2. High-value security environments (fraud analysis, enterprise executive protection) where every missed spam carries high cost.
      3. Serving asynchronous offline queues or secondary verification pipelines where 70ms latency is fully acceptable.
    """)


# ==============================================================================
# SECTION 7: LIVE SPAM PREDICTION
# ==============================================================================
elif active_section == "7. Live Spam Prediction":
    st.markdown('<div class="sub-tag">07 / INTERACTIVE INFERENCE</div>', unsafe_allow_html=True)
    st.markdown('<h2>Live Spam Prediction Playground</h2>', unsafe_allow_html=True)
    st.markdown('<p class="deck-text">Test real-time spam detection across both traditional ML and transformer models.</p>', unsafe_allow_html=True)

    col_input, col_presets = st.columns([3, 2], gap="large")

    sample_presets = {
        "Lottery prize award (Spam)": "URGENT! You have won a 1-week all-expenses paid vacation to Hawaii. Call 09061701461 to claim your ticket now! Code: WIN77",
        "Cash prize claim (Spam)": "WINNER!! As a valued customer you are selected for a £1,000 cash reward. Reply CLAIM to 88202 immediately.",
        "Bank account phishing (Spam)": "Your debit card has experienced unauthorized login attempts. Reply with your 6-digit PIN code immediately to avoid suspension.",
        "Academic study group (Ham)": "Hey Sujal, are we still meeting in the library at 4pm for the project discussion?",
        "Daily errand coordination (Ham)": "Can you pick up milk and bread on your way home?",
        "Course submission alert (Ham)": "Don't forget to submit your final DCS 404 assignment before midnight."
    }

    with col_presets:
        st.markdown('<div class="card-neutral">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Select Sample Preset</div>', unsafe_allow_html=True)
        selected_preset = st.radio(
            "Quick Verification Presets:",
            options=["Custom Message"] + list(sample_presets.keys()),
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)

        preset_text = ""
        if selected_preset != "Custom Message":
            preset_text = sample_presets[selected_preset]

    with col_input:
        st.markdown('<div class="section-title">Input Message Text</div>', unsafe_allow_html=True)
        user_message = st.text_area(
            "Enter your message:",
            value=preset_text,
            height=140,
            placeholder="Type or paste any SMS or email text to classify..."
        )

        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            model_options = [
                "Logistic Regression (SMOTE)",
                "Multinomial Naive Bayes",
                "DistilBERT",
                "DistilRoBERTa"
            ]
            chosen_model = st.selectbox("Select Model:", options=model_options, index=0)
        with col_btn:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            analyze_clicked = st.button("Analyze Message", type="primary", use_container_width=True)

    # Prediction Logic
    if analyze_clicked or (user_message and selected_preset != "Custom Message"):
        if not user_message.strip():
            st.error("Please enter a valid message text before analyzing.")
        else:
            pred_label = 0
            confidence = 0.0
            prob_ham = 0.0
            prob_spam = 0.0
            inference_ms = 0.0

            # Execute Model
            if chosen_model in ["Logistic Regression (SMOTE)", "Multinomial Naive Bayes"]:
                tfidf = trad_artifacts.get('tfidf')
                model_obj = trad_artifacts.get('lr') if "Logistic" in chosen_model else trad_artifacts.get('nb')
                
                if tfidf is not None and model_obj is not None:
                    t0 = time.time()
                    vec = tfidf.transform([user_message])
                    pred_label = int(model_obj.predict(vec)[0])
                    proba = model_obj.predict_proba(vec)[0]
                    inference_ms = (time.time() - t0) * 1000
                    prob_ham = float(proba[0]) * 100
                    prob_spam = float(proba[1]) * 100
                    confidence = prob_spam if pred_label == 1 else prob_ham
                else:
                    st.error("Traditional model artifacts not found. Please run train_and_evaluate.py.")

            else:
                # Transformer Model
                subpath = "distilbert" if "BERT" in chosen_model else "distilroberta"
                with st.spinner(f"Loading {chosen_model} weights for sequence classification..."):
                    tokenizer, model = load_transformer_pipeline(subpath)
                
                if tokenizer is not None and model is not None:
                    import torch
                    t0 = time.time()
                    inputs = tokenizer([user_message], padding=True, truncation=True, max_length=96, return_tensors='pt')
                    with torch.no_grad():
                        out = model(**inputs)
                        probs = torch.softmax(out.logits, dim=-1)[0].numpy()
                    inference_ms = (time.time() - t0) * 1000
                    prob_ham = float(probs[0]) * 100
                    prob_spam = float(probs[1]) * 100
                    pred_label = int(np.argmax(probs))
                    confidence = prob_spam if pred_label == 1 else prob_ham
                else:
                    st.error(f"Failed to load {chosen_model} weights. Check models/transformer_models/ directory.")

            # Render Inference Result
            st.markdown('<hr>', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1], gap="large")

            with res_col1:
                if pred_label == 1:
                    st.markdown(f"""
                    <div class="result-spam">
                        <div class="tag-spam">Prediction: SPAM</div>
                        <div class="big-stat" style="color: #D32F2F;">{confidence:.1f}%</div>
                        <div class="sub-tag">PREDICTION CONFIDENCE</div>
                        <p style="font-size: 0.9rem; line-height: 1.5; color: #111; margin-top: 0.75rem; border-top: 1px solid #E5E7EB; padding-top: 0.75rem;">
                            <strong>High-Risk Lure Detected:</strong> This message exhibits vocabulary and grammatical patterns characteristic of promotional lures, phishing vectors, or financial scam attempts.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-ham">
                        <div class="tag-ham">Prediction: HAM (Legitimate)</div>
                        <div class="big-stat" style="color: #000000;">{confidence:.1f}%</div>
                        <div class="sub-tag">PREDICTION CONFIDENCE</div>
                        <p style="font-size: 0.9rem; line-height: 1.5; color: #111; margin-top: 0.75rem; border-top: 1px solid #E5E7EB; padding-top: 0.75rem;">
                            <strong>Legitimate Conversational Text:</strong> The vocabulary, structure, and phrasing conform to natural interpersonal dialogue with minimal spam risk.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Confidence Probability Breakdown</div>', unsafe_allow_html=True)
                
                # Visual Metric Bars
                st.markdown(f"""
                <div style="margin-bottom: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 5px;">
                        <span>Ham (Legitimate) Probability</span>
                        <span style="font-variant-numeric: tabular-nums;">{prob_ham:.2f}%</span>
                    </div>
                    <div style="width: 100%; height: 16px; border: 1px solid #000000; background: #FFFFFF;">
                        <div style="width: {prob_ham:.2f}%; height: 100%; background: #000000;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 5px;">
                        <span style="color: #D32F2F;">Spam Probability</span>
                        <span style="font-variant-numeric: tabular-nums; color: #D32F2F;">{prob_spam:.2f}%</span>
                    </div>
                    <div style="width: 100%; height: 16px; border: 1px solid #000000; background: #FFFFFF;">
                        <div style="width: {prob_spam:.2f}%; height: 100%; background: #D32F2F;"></div>
                    </div>
                </div>
                <div class="spec-row" style="margin-top: 1.25rem; border-top: 1px solid #000000; padding-top: 8px;">
                    <span class="spec-label">Active Evaluator</span>
                    <span class="spec-value">{chosen_model}</span>
                </div>
                <div class="spec-row">
                    <span class="spec-label">Inference Execution Time</span>
                    <span class="spec-value">{inference_ms:.2f} ms</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Feature explainability for traditional TF-IDF models
            if chosen_model in ["Logistic Regression (SMOTE)", "Multinomial Naive Bayes"]:
                tfidf = trad_artifacts.get('tfidf')
                if tfidf is not None:
                    vec = tfidf.transform([user_message])
                    nonzero_idx = vec.nonzero()[1]
                    feature_names = tfidf.get_feature_names_out()
                    if len(nonzero_idx) > 0:
                        matched = [
                            {"Feature N-Gram": feature_names[i], "TF-IDF Weight": round(float(vec[0, i]), 4)}
                            for i in nonzero_idx
                        ]
                        matched = sorted(matched, key=lambda x: x["TF-IDF Weight"], reverse=True)
                        st.markdown('<div class="sub-tag" style="margin-top: 1.25rem;">VOCABULARY EXPLAINABILITY</div>', unsafe_allow_html=True)
                        with st.expander("Inspecting Matched TF-IDF Vocabulary Features"):
                            st.caption("Active tokens extracted from the input text present in the trained 5,000-term TF-IDF dictionary:")
                            st.dataframe(pd.DataFrame(matched), use_container_width=True, hide_index=True)
                    else:
                        st.info("Out of Vocabulary: None of the tokens in this input matched the trained 5,000-term TF-IDF dictionary.")
            else:
                st.markdown('<div class="sub-tag" style="margin-top: 1.25rem;">TRANSFORMER CONTEXTUAL EMBEDDING</div>', unsafe_allow_html=True)
                with st.expander(f"Inspecting {chosen_model} Architecture Details"):
                    st.caption("Deep transformer models process raw text through subword Byte-Pair / WordPiece tokenization and evaluate bidirectional attention context across the entire sequence.")
                    st.markdown(f"""
                    - **Architecture:** Pre-trained contextual bidirectional encoder fine-tuned on binary classification.
                    - **Max Sequence Length:** 96 tokens
                    - **Tokenization:** Subword representation capturing prefixes, morphology, and out-of-vocabulary terms.
                    """)
