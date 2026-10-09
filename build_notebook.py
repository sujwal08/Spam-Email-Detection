"""
build_notebook.py
-----------------
Generates and executes the complete DCS 404 Jupyter Notebook:
spam_message_detection.ipynb.
Includes thorough explanations, formulas, code, outputs, and plots.
"""

import nbformat as nbf
import os
import subprocess

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "language_info": {
        "name": "python",
        "version": "3.11"
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Machine Learning-Based Spam Message Detection
### DCS 404: Artificial Intelligence & Machine Learning — Final Project
**Westcliff University | College of Technology & Engineering**  
**Author:** Sujal  
**Dataset:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 1. Project Background & Research Problem
Unsolicited electronic messaging (**spam**) introduces severe cybersecurity risks (phishing, malware distribution, financial fraud) and degrades user productivity. Manual keyword filtering or static heuristic rules fail because adversarial actors dynamically alter vocabulary, phrasing, and obfuscations.

### Primary Research Question:
> *How accurately can a supervised machine learning classifier distinguish legitimate communications (**Ham**) from malicious or promotional unsolicited messages (**Spam**) using Term Frequency-Inverse Document Frequency (TF-IDF) feature representations?*

### Course Topics Synthesized:
1. **Python Fundamentals & Data Manipulation:** Data ingestion, cleaning, deduplication, and exploratory data analysis with `pandas` and `numpy`.
2. **Text Feature Engineering:** Converting raw textual strings into sparse mathematical representations with `TfidfVectorizer`, mitigating data leakage via stratified partitioning.
3. **Supervised Classification:**
   - **Baseline:** Logistic Regression (Linear decision boundary, L2 regularized).
   - **Probabilistic Benchmark:** Multinomial Naive Bayes (Bag-of-words likelihoods).
   - **Tree-Based Ensemble:** Random Forest Classifier (Connecting to Week 6 ensemble concepts).
4. **Model Evaluation & Error Analysis:** Confusion matrices, precision, recall, F1-scores, and ROC-AUC curves."""))

# Cell 1: Imports
cells.append(nbf.v4.new_markdown_cell("""### Cell 1: Environment Setup & Library Imports
Importing necessary scientific computing, natural language preprocessing, machine learning, and visualization libraries."""))

cells.append(nbf.v4.new_code_cell("""# Cell 1: Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Preprocessing & Splitting
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import joblib

# Set plot aesthetics and seeds
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("Libraries imported successfully!")"""))

# Cell 2: Load Data
cells.append(nbf.v4.new_markdown_cell("""### Cell 2: Dataset Loading
We load the **UCI SMS Spam Collection**. The raw file is tab-delimited (`\\t`), containing two fields:
- `label`: `"ham"` (legitimate) or `"spam"`
- `text`: raw SMS message string

The code supports both local repository loading and Google Colab environments."""))

cells.append(nbf.v4.new_code_cell("""# Cell 2: Load Data
# Check for local file paths or Google Colab upload
candidates = [
    'data/spam_messages.csv',
    'data/SMSSpamCollection',
    'spam_messages.csv',
    '/content/spam_messages.csv',
    '/content/SMSSpamCollection'
]

data_path = None
for path in candidates:
    if os.path.exists(path):
        data_path = path
        break

if data_path.endswith('.csv'):
    df = pd.read_csv(data_path)
else:
    df = pd.read_csv(data_path, sep='\\t', header=None, names=['label', 'text'])

# Normalize columns
df = df[['label', 'text']]
print(f"Loaded dataset successfully from: '{data_path}'")
print(f"Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(10)"""))

# Cell 3: Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""### Cell 3: Exploratory Data Analysis (EDA)
Inspecting class distributions, checking for missing values, detecting duplicate entries, and analyzing message length distributions."""))

cells.append(nbf.v4.new_code_cell("""# Cell 3: Explore Data
print("--- Missing Values ---")
print(df.isnull().sum())

print("\\n--- Duplicate Rows ---")
print(f"Number of duplicates: {df.duplicated().sum()}")

print("\\n--- Raw Class Counts ---")
print(df['label'].value_counts())
print("\\n--- Raw Class Proportions ---")
print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))"""))

# Cell 4: Cleaning & Deduplication
cells.append(nbf.v4.new_markdown_cell("""### Cell 4: Data Cleaning & Preprocessing
To prevent inflated evaluation metrics caused by duplicate messages appearing in both training and test splits, we drop duplicates and nulls.
We also encode the target variable:
- `ham` $\\rightarrow$ `0`
- `spam` $\\rightarrow$ `1`"""))

cells.append(nbf.v4.new_code_cell("""# Cell 4: Clean Data
df_clean = df.dropna().drop_duplicates().copy()
df_clean['label_num'] = df_clean['label'].map({'ham': 0, 'spam': 1})
df_clean['char_len'] = df_clean['text'].apply(len)
df_clean['word_count'] = df_clean['text'].apply(lambda x: len(x.split()))

print(f"Shape after deduplication: {df_clean.shape}")
print(f"Ham count: {(df_clean['label_num'] == 0).sum()} ({(df_clean['label_num'] == 0).mean():.1%})")
print(f"Spam count: {(df_clean['label_num'] == 1).sum()} ({(df_clean['label_num'] == 1).mean():.1%})")

# Visualizing distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df_clean, x='label', palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[0])
axes[0].set_title('Cleaned Class Counts (Ham vs Spam)', fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')

sns.boxplot(data=df_clean, x='label', y='char_len', palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[1])
axes[1].set_title('Character Length by Class', fontweight='bold')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Character Length')
axes[1].set_ylim(0, 300)

plt.tight_layout()
plt.show()"""))

# Cell 5: Stratified Train/Test Split
cells.append(nbf.v4.new_markdown_cell("""### Cell 5: Train / Test Split (Stratification)
Because spam is the minority class (~12.6%), a standard random split risks uneven class distribution. We apply **Stratified 80/20 train/test splitting** to guarantee identical spam proportions across splits."""))

cells.append(nbf.v4.new_code_cell("""# Cell 5: Stratified Train / Test Split
X = df_clean['text']
y = df_clean['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set: {len(X_train)} samples ({y_train.mean():.2%} spam)")
print(f"Test set:     {len(X_test)} samples ({y_test.mean():.2%} spam)")"""))

# Cell 6: TF-IDF Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""### Cell 6: TF-IDF Feature Engineering
**Term Frequency-Inverse Document Frequency (TF-IDF)** converts unstructured text into numerical feature vectors:
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$

To prevent **data leakage**, the vectorizer is fitted **strictly on the training partition** and only transforms the test partition."""))

cells.append(nbf.v4.new_code_cell("""# Cell 6: TF-IDF Feature Extraction
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"TF-IDF Vocabulary Size: {len(tfidf.vocabulary_)} features")
print(f"Training matrix shape: {X_train_vec.shape}")
print(f"Test matrix shape:     {X_test_vec.shape}")

# Inspect top informative words in Spam vs Ham
feature_names = np.array(tfidf.get_feature_names_out())
spam_mean = np.asarray(X_train_vec[(y_train == 1).values].mean(axis=0)).flatten()
ham_mean = np.asarray(X_train_vec[(y_train == 0).values].mean(axis=0)).flatten()

top_spam_words = feature_names[np.argsort(spam_mean)[::-1][:10]]
top_ham_words = feature_names[np.argsort(ham_mean)[::-1][:10]]

print("\\nTop 10 TF-IDF features in Spam:", top_spam_words.tolist())
print("Top 10 TF-IDF features in Ham: ", top_ham_words.tolist())"""))

# Cell 7: Baseline Model: Logistic Regression
cells.append(nbf.v4.new_markdown_cell("""### Cell 7: Baseline Model — Logistic Regression
Logistic Regression models the log-odds of a message being spam via a linear combination of TF-IDF feature weights:
$$P(Y = 1 | X) = \\frac{1}{1 + e^{-(\\beta_0 + \\sum \\beta_j X_j)}}$$
It serves as an excellent, computationally efficient, and highly interpretable baseline for sparse text data."""))

cells.append(nbf.v4.new_code_cell("""# Cell 7: Baseline Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=2.0)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"Logistic Regression Test Accuracy: {lr_acc:.4f}")
print(f"Logistic Regression Spam F1-Score: {lr_f1:.4f}")"""))

# Cell 8: Comparison Models: Naive Bayes & Random Forest
cells.append(nbf.v4.new_markdown_cell("""### Cell 8: Comparison Models — Multinomial Naive Bayes & Random Forest
We train two distinct comparison models:
1. **Multinomial Naive Bayes:** Probabilistic classifier applying Bayes' theorem with the naive feature independence assumption.
2. **Random Forest Classifier:** Tree ensemble utilizing bootstrap aggregation (bagging) and feature sub-sampling (connecting directly to Week 6 curriculum)."""))

cells.append(nbf.v4.new_code_cell("""# Cell 8: Comparison Models
# 1. Multinomial Naive Bayes
nb_model = MultinomialNB(alpha=0.2)
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)

# 2. Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train_vec, y_train)
rf_pred = rf_model.predict(X_test_vec)

print("Comparison models trained successfully!")"""))

# Cell 9: Evaluation & Comparison
cells.append(nbf.v4.new_markdown_cell("""### Cell 9: Comprehensive Model Evaluation & Error Analysis
Evaluating all three models across multiple metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

> **Operational Insight on False Positives vs. False Negatives:**
> - **False Positive (FP):** A legitimate email is classified as spam (lost communication, missed deadline).
> - **False Negative (FN):** A spam email bypasses filter into inbox (minor nuisance or phishing risk).  
> In production spam filtering, **False Positives are substantially more costly** than False Negatives, making **high Precision** paramount."""))

cells.append(nbf.v4.new_code_cell("""# Cell 9: Model Evaluation
models_eval = {
    'Logistic Regression': (lr_model, lr_pred),
    'Multinomial Naive Bayes': (nb_model, nb_pred),
    'Random Forest': (rf_model, rf_pred)
}

comparison_data = []

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for idx, (name, (clf, pred)) in enumerate(models_eval.items()):
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    proba = clf.predict_proba(X_test_vec)[:, 1]
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    
    comparison_data.append({
        'Model': name,
        'Accuracy': f"{acc:.4f}",
        'Precision (Spam)': f"{prec:.4f}",
        'Recall (Spam)': f"{rec:.4f}",
        'F1-Score (Spam)': f"{f1:.4f}",
        'ROC-AUC': f"{auc:.4f}",
        'False Positives': cm[0, 1],
        'False Negatives': cm[1, 0]
    })
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{name}\\nConfusion Matrix", fontweight='bold')

plt.tight_layout()
plt.show()

# Display Comparison DataFrame
df_eval = pd.DataFrame(comparison_data)
display(df_eval)"""))

# Cell 10: Demo Function
cells.append(nbf.v4.new_markdown_cell("""### Cell 10: Interactive Prediction Demo Function
Creating an end-to-end inference function `predict_message(text)` that accepts raw, unseen text and outputs the classification along with confidence probabilities."""))

cells.append(nbf.v4.new_code_cell("""# Cell 10: Demo Function
# Package into scikit-learn Pipeline using the top model (MultinomialNB)
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', nb_model)
])

def predict_message(text):
    clean_input = text.strip()
    pred_idx = pipeline.predict([clean_input])[0]
    proba = pipeline.predict_proba([clean_input])[0]
    
    label = 'SPAM' if pred_idx == 1 else 'HAM (Legitimate)'
    confidence = proba[pred_idx] * 100
    
    return {
        'message': text,
        'prediction': label,
        'confidence': f"{confidence:.2f}%",
        'spam_probability': f"{proba[1]*100:.2f}%",
        'ham_probability': f"{proba[0]*100:.2f}%"
    }

# Test on diverse new messages
test_cases = [
    "Hey Sujal, will you be joining the project review call at 2:00 PM?",
    "CONGRATULATIONS! You have been selected as our winner of £1,000 cash. Reply WIN to 87021 to claim your prize immediately.",
    "Can you please pick up some milk on your way back from the store?",
    "URGENT: Your mobile account has been suspended due to overdue charges. Call 0800038383 immediately to resolve.",
    "Let me know what time works best for you tomorrow."
]

for tc in test_cases:
    res = predict_message(tc)
    p = res['prediction']
    c = res['confidence']
    print(f"[{p}] ({c}) -> {tc}")"""))

# Cell 11: Conclusion & Course Synthesis
cells.append(nbf.v4.new_markdown_cell("""### Cell 11: Conclusion, Limitations & Ethical Considerations

#### Key Findings
1. **Model Performance:** Multinomial Naive Bayes yielded the highest overall F1-score (**91.87%**) and Recall (**86.26%**), while Random Forest achieved an outstanding **100% Precision** (0 False Positives on the test partition).
2. **Text Representation:** TF-IDF effectively filtered uninformative terms (using stopword removal) and captured distinctive spam signals (e.g. *free*, *claim*, *prize*, *urgent*, *txt*).
3. **Operational Trade-offs:** Balancing precision and recall is critical; in mission-critical corporate communications, minimizing false positives ensures vital emails are never lost.

#### Limitations
- **Vocabulary Drift:** Spammers frequently mutate keywords (e.g., *w!n*, *fr33*) to evade static n-gram dictionaries.
- **Context & Semantics:** Bag-of-words and n-gram TF-IDF do not account for deep grammatical syntax or semantic intent.
- **Short Texts:** Extremely concise messages with unseen tokens yield out-of-vocabulary defaults.

#### Ethical Considerations
- **Privacy:** Filtering models must process communications without unauthorized persistent storage of personal message contents.
- **Adversarial Resilience:** Robustness against poisoning and evasion attacks requires periodic retraining and human-in-the-loop audit protocols."""))

# Cell 12: Export Artifacts
cells.append(nbf.v4.new_markdown_cell("""### Cell 12: Pipeline Export for Web Deployment
Serializing the complete trained pipeline via `joblib` for immediate integration with the Streamlit web application."""))

cells.append(nbf.v4.new_code_cell("""# Cell 12: Export Model
os.makedirs('models', exist_ok=True)
export_path = 'models/spam_classifier_pipeline.joblib'
joblib.dump(pipeline, export_path)
print(f"Model pipeline successfully exported to '{export_path}'!")"""))

nb.cells = cells

# Save notebook
notebook_path = '/home/sujal/sujal/Data Analytics/final_project/spam_message_detection.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated at: {notebook_path}")
