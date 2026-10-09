# Machine Learning & Transformer-Based Spam Message Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive%20UI-brightgreen.svg)](https://streamlit.io/)
[![PyTorch & Transformers](https://img.shields.io/badge/PyTorch%20%7C%20Transformers-DistilBERT%20%26%20RoBERTa-red.svg)](https://huggingface.co/)
[![SMOTE Balanced](https://img.shields.io/badge/Imbalance-SMOTE%20Balanced-blueviolet.svg)](https://imbalanced-learn.org/)

**Course:** DCS 404 - Artificial Intelligence & Machine Learning  
**Institution:** Westcliff University | College of Technology & Engineering  
**Dataset:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)  
**Author:** Sujal  

---

## 1. Executive Summary & Project Updates

This repository presents an updated, production-grade Natural Language Processing (NLP) system designed to detect unsolicited, fraudulent, and promotional SMS messages (**Spam** vs. **Ham**). 

Based on professor feedback and advanced methodological requirements, the project incorporates:
1. **Class Imbalance Remediation via SMOTE:** Resolving the natural ~87.4% Ham vs. 12.6% Spam class skew in the feature space.
2. **Strict Anti-Data Leakage Protocol:** Stratified 80/20 partitioning performed *prior* to TF-IDF vectorization and SMOTE oversampling. The test split remains pristine with authentic class prevalence.
3. **Traditional Machine Learning:** **Logistic Regression** (optimized with SMOTE) and **Multinomial Naive Bayes** (with an in-depth analysis of why SMOTE degrades Bayesian prior probabilities).
4. **Deep Transformer Architectures:** Fine-tuned sequence classification with **DistilBERT** and **DistilRoBERTa**, capturing full bidirectional contextual semantics.
5. **Textual Visualizations:** High-resolution Word Clouds for both Spam triggers and Ham conversational vocabulary.
6. **Architectural Trade-Off Analysis:** Evaluating models across Accuracy, F1-score, Inference Latency, Memory Footprint, and Hardware Constraints.
7. **Production Streamlit Web App:** Simple, clean dashboard containing 7 dedicated sections and a live prediction playground.

---

## 2. Recommended Project Structure

```text
final_project/
├── app.py                          # Multi-page Streamlit web application (7 sections)
├── build_training_notebook.py      # Automated generator for Colab & local notebooks
├── create_presentation.py          # Week 8 PowerPoint presentation generator
├── requirements.txt                # Project dependencies (scikit-learn, imblearn, torch, transformers)
├── README.md                       # Comprehensive project documentation & run instructions
├── train_and_evaluate.py           # End-to-end training, SMOTE, evaluation & export script
├── update.md                       # Professor project update specification document
├── assets/                         # Visual assets & evaluation plots
│   ├── class_distribution.png      # Cleaned dataset class breakdown
│   ├── confusion_matrices.png      # 2x2 grid of test confusion matrices
│   ├── ham_wordcloud.png           # Word cloud of legitimate conversational messages
│   ├── message_length_distribution.png # Character length histogram by category
│   ├── model_comparison.png        # Comparative performance bar chart
│   ├── roc_curves.png              # Receiver Operating Characteristic (ROC) curves
│   ├── smote_class_distribution.png# Before vs. After SMOTE class balance
│   └── spam_wordcloud.png          # Word cloud of high-risk spam triggers
├── data/                           # Data storage
│   ├── SMSSpamCollection           # Original raw UCI dataset (tab-separated)
│   └── spam_messages.csv           # Cleaned and deduplicated dataset (5,169 rows)
├── figures/                        # Mirrored figures directory for backwards compatibility
├── models/                         # Serialized model artifacts & evaluation metadata
│   ├── logistic_regression.pkl     # Logistic regression estimator (joblib / pkl)
│   ├── naive_bayes.pkl             # Multinomial Naive Bayes estimator (joblib / pkl)
│   ├── tfidf_vectorizer.pkl        # Fitted 5,000-term TF-IDF vectorizer (pkl)
│   ├── model_metadata.json         # Benchmark metrics and pipeline configuration
│   └── transformer_models/         # Fine-tuned Hugging Face transformer checkpoints
│       ├── distilbert/             # DistilBERT model weights, config, and tokenizer
│       └── distilroberta/          # DistilRoBERTa model weights, config, and tokenizer
├── notebooks/                      # Interactive Jupyter / Colab notebooks
│   └── training.ipynb              # Complete end-to-end reproducible training notebook
└── presentation/                   # Presentation slide deliverables
    └── DCS404_Spam_Detection_Presentation.pptx
```

---

## 3. Strict Methodology & Anti-Leakage Protocol

To ensure academic and industry validity, the pipeline adheres to strict partitioning rules:

```text
Raw Dataset (5,169 Clean Messages)
               ↓
    80/20 Stratified Split  ──> Test Partition (1,034 samples, 12.6% Spam untouched)
               ↓
Fit TF-IDF on Training Data Only
               ↓
Transform Train & Test Splits
               ↓
Apply SMOTE to Training Features Only (3,613 Ham / 3,613 Spam balanced)
               ↓
Train Classifiers (Logistic Regression, Transformers)
               ↓
Evaluate on Pristine Held-Out Test Set (1,034 samples)
```

### Critical Rules Explained:
1. **Split First:** The dataset is partitioned before computing document frequencies. Fitting TF-IDF on the full dataset causes *data leakage* by exposing vocabulary frequencies from the test set to the training phase.
2. **Fit TF-IDF Only on Train:** The vectorizer learns vocabulary and IDF weights exclusively from the 4,135 training samples.
3. **SMOTE on Training Features Only:** Synthetic minority interpolation is performed solely within the training feature matrix. **The test set is never oversampled or balanced**, preserving real-world natural prevalence.
4. **Bayesian Prior Distortion (Why SMOTE Harms Naive Bayes):**  
   Multinomial Naive Bayes relies on prior probability:
   $$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^D P(x_j \mid y)$$
   In real data, $P(\text{Spam}) \approx 0.126$. SMOTE forces the training prior to $P(\text{Spam}) = 0.500$. As a result, Naive Bayes severely over-predicts the minority class on natural test distributions:
   - **Standard Naive Bayes:** **2 False Positives** (Precision = 98.26%)
   - **SMOTE-Trained Naive Bayes:** **21 False Positives** (Precision dropped to 85.21%!)  
   *Hence, Logistic Regression is the primary SMOTE-based classifier, while standard Naive Bayes is preserved as a baseline.*

---

## 4. Empirical Benchmark & Comparison

All models were evaluated on the identical held-out test partition ($N = 1,034$; 903 Ham, 131 Spam):

| Model | Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SMOTE)** | Traditional ML | 97.29% | 88.72% | 90.08% | **89.39%** | 0.9903 | 0.06 s | **0.0008 ms/sample** |
| **Multinomial Naive Bayes** | Traditional ML | 98.07% | 98.26% | 86.26% | **91.87%** | 0.9897 | 0.01 s | **0.0008 ms/sample** |
| **DistilBERT** | Transformer | **98.84%** | 95.42% | **95.42%** | **95.42%** | **0.9975** | ~47 min (CPU) | 67.04 ms/sample |
| **DistilRoBERTa** | Transformer | **98.84%** | **97.60%** | 93.13% | **95.31%** | **0.9993** | ~38 min (CPU) | 73.55 ms/sample |

### Multi-Dimensional Trade-Off Analysis

| Dimension | Traditional ML (Logistic Regression + SMOTE) | Transformer Models (DistilBERT / DistilRoBERTa) |
| :--- | :--- | :--- |
| **Classification F1-Score** | 89.39% (Solid, highly competitive) | **95.42%** (+6.0% improvement in minority recall) |
| **Inference Latency** | **< 0.001 ms / message** (> 1,000,000 req/sec) | **~67 - 73 ms / message** (~15 req/sec on CPU) |
| **Training Duration** | **< 0.1 seconds** | **~40 - 50 minutes** (CPU) / ~2 mins (GPU) |
| **Disk & Memory Footprint** | **~220 KB** (`.pkl` file) | **~268 MB - 328 MB** checkpoint |
| **Hardware Requirements** | Runs on microcontrollers, Lambda, Raspberry Pi | Requires multi-core CPU or GPU acceleration |
| **Feature Interpretability** | Transparent linear weights on TF-IDF n-grams | Complex non-linear self-attention matrices |

#### Operational Deployment Recommendations:
- **Deploy Logistic Regression (SMOTE)** in high-throughput network gateways, real-time carrier SMS routing, low-power edge devices, or budget-constrained serverless architectures.
- **Deploy DistilBERT / DistilRoBERTa** in asynchronous security auditing queues, financial anti-fraud verification, or high-security enterprise filters where capturing subtle evasions justifies higher computational cost.

---

## 5. Streamlit Web Application

The interactive Streamlit dashboard (`app.py`) provides 7 dedicated sections:
1. **Home:** System overview, research motivation, and pipeline architecture diagram.
2. **Dataset Overview:** Dataset statistics, 80/20 partition metrics, and an interactive searchable corpus preview.
3. **Data Exploration:** Class distribution, character length bimodality, and before/after SMOTE balancing charts.
4. **Spam and Ham Word Clouds:** Side-by-side high-resolution word clouds and vocabulary trigger analysis.
5. **Model Performance:** Individual metrics, confusion matrices, ROC curves, and Naive Bayes SMOTE limitation analysis.
6. **Model Comparison:** Comprehensive benchmark table, multi-metric bar chart, and engineering trade-off matrix.
7. **Live Spam Prediction:** Interactive playground supporting preset lures and custom text input with **Logistic Regression**, **Naive Bayes**, **DistilBERT**, and **DistilRoBERTa** model selectors, real-time confidence scores, and probability meters.

---

## 6. How to Run the Project

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Linux, macOS, or Windows

### Step 1: Clone and Set Up Virtual Environment

```bash
# Clone the repository
git clone <repository_url>
cd final_project

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run End-to-End Training & Evaluation (Optional if models exist)

To re-run data cleaning, SMOTE balancing, traditional training, transformer evaluation, figure generation, and model export:

```bash
python train_and_evaluate.py
```

### Step 3: Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Running in Google Colab

The training notebook is fully self-contained and reproducible in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/training.ipynb`.
3. Enable GPU acceleration: `Runtime` -> `Change runtime type` -> select `T4 GPU`.
4. Run all cells sequentially. The notebook will automatically fetch the UCI dataset, install dependencies, execute SMOTE, train both traditional and transformer models, and export all metrics and figures.

---

## 7. Deliverables Checklist

- [x] Preprocess text & handle missing/duplicate entries.
- [x] Stratified 80/20 train/test split.
- [x] Fit TF-IDF only on training data (anti-leakage).
- [x] SMOTE applied exclusively to training features.
- [x] Logistic Regression trained with SMOTE.
- [x] Multinomial Naive Bayes evaluated & SMOTE prior limitation explained.
- [x] DistilBERT fine-tuned and evaluated for binary sequence classification.
- [x] DistilRoBERTa fine-tuned and evaluated for binary sequence classification.
- [x] High-resolution Spam and Ham Word Clouds generated.
- [x] Class distribution & message length visualizations created.
- [x] Confusion matrices and ROC curves generated for all models.
- [x] Comprehensive comparative benchmark table & bar chart.
- [x] Multi-dimensional engineering trade-off analysis.
- [x] Interactive Streamlit app with 7 suggested sections and live prediction.
- [x] Saved models (`logistic_regression.pkl`, `naive_bayes.pkl`, `tfidf_vectorizer.pkl`, and transformer weights).
- [x] Reproducible Google Colab training notebook (`notebooks/training.ipynb`).
- [x] Updated `requirements.txt` and `README.md`.
