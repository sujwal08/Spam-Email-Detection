# Project Update Instructions

Update my existing **Python + Streamlit Spam Message Detection project** based on my professor’s feedback. Keep the project beginner-friendly, clean, and easy to explain in class.

The project uses the **UCI SMS Spam Collection dataset** with `spam` and `ham` labels.

## 1. Handle Class Imbalance with SMOTE

- Preprocess the text first.
- Split the dataset into **80% training and 20% testing data** using stratification.
- Fit the **TF-IDF vectorizer only on the training data** to avoid data leakage.
- Transform both training and test text using TF-IDF.
- Apply **SMOTE only to the TF-IDF training features and training labels**.
- Do **not** apply SMOTE to the test set.
- Show the class distribution before and after applying SMOTE.

Important workflow:

```text
Raw Dataset
    ↓
Train/Test Split
    ↓
Fit TF-IDF on Training Data
    ↓
Transform Train and Test Data
    ↓
Apply SMOTE to Training Data Only
    ↓
Train Machine Learning Models
    ↓
Evaluate on Original Test Data
```

## 2. Traditional Machine Learning Models

Keep at least two traditional machine learning models:

- Logistic Regression
- Multinomial Naive Bayes

Use **Logistic Regression as the main SMOTE-based model**.

For each model, calculate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

If SMOTE creates any compatibility issue with Multinomial Naive Bayes, handle it properly and explain the limitation.

## 3. Add Transformer Models

Add transformer-based text classification using:

- BERT
- RoBERTa

For easier training and faster execution, lightweight versions may be used:

- DistilBERT
- DistilRoBERTa

Requirements:

- Fine-tune the transformer model for binary classification: **Spam vs Ham**.
- Use the same train/test split where practical for a fair comparison.
- Keep the transformer pipeline separate from the TF-IDF + traditional machine learning pipeline.
- Training should be reproducible in Google Colab.

## 4. Add Word Clouds

Create two word clouds:

- Spam Word Cloud
- Ham Word Cloud

Also show:

- Number of spam messages
- Number of ham messages
- Class distribution chart

Add these visualizations to a **Data Exploration** section in Streamlit.

## 5. Compare Transformer and Machine Learning Models

Create a comparison table with:

- Logistic Regression
- Naive Bayes
- BERT / DistilBERT
- RoBERTa / DistilRoBERTa

Compare models using:

- Accuracy
- Precision
- Recall
- F1-score
- Training time, if available
- Prediction/inference time, if available

Also create a bar chart comparing model performance.

Explain the trade-offs between traditional ML and transformer models, including:

- Accuracy/performance
- Speed
- Complexity
- Hardware requirements
- Ease of deployment

Do not only say which model is best. Explain why different models may be preferred in different situations.

## 6. Improve the Streamlit UI

Keep the interface simple, clean, and professional.

Suggested sections:

1. Home
2. Dataset Overview
3. Data Exploration
4. Spam and Ham Word Clouds
5. Model Performance
6. Model Comparison
7. Live Spam Prediction

### Live Spam Prediction

Allow the user to:

- Type or paste a message
- Select a model
- Click **Analyze Message**
- See the result as **SPAM** or **HAM**
- See prediction confidence where available

Example:

```text
------------------------------------
Spam Message Detection
------------------------------------

Enter your message:

[ Congratulations! You won a prize ]

Select Model:
[ Logistic Regression ▼ ]

[ Analyze Message ]

Prediction: SPAM
Confidence: 96%
------------------------------------
```

## 7. Recommended Project Structure

```text
final_project/
├── app.py
├── data/
│   └── SMSSpamCollection
├── models/
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── tfidf_vectorizer.pkl
│   └── transformer_models/
├── notebooks/
│   └── training.ipynb
├── assets/
│   ├── spam_wordcloud.png
│   ├── ham_wordcloud.png
│   └── model_comparison.png
├── requirements.txt
├── README.md
└── update.md
```

## 8. Code Requirements

- Use beginner-friendly Python.
- Add comments explaining important steps.
- Do not over-engineer the project.
- Reuse functions where helpful.
- Keep training code separate from the Streamlit UI where possible.
- Make sure training works in Google Colab.
- Make sure the Streamlit app runs locally.
- Save traditional ML models using `joblib`.

## 9. Important Methodology Rules

To avoid data leakage:

- Split the dataset before fitting TF-IDF.
- Fit TF-IDF only on training text.
- Transform the test set using the already-fitted TF-IDF vectorizer.
- Apply SMOTE only to transformed training features.
- Never balance the test set.
- Evaluate models on the original untouched test set.

These decisions should also be explained in comments and in the README.

## 10. Final Deliverables

The completed project should include:

- Updated training notebook
- Updated Streamlit app
- TF-IDF feature extraction
- SMOTE balancing
- Logistic Regression model
- Naive Bayes model
- BERT/DistilBERT model
- RoBERTa/DistilRoBERTa model
- Spam word cloud
- Ham word cloud
- Class distribution visualization
- Confusion matrices
- Accuracy, precision, recall, and F1-score
- Model comparison table
- Model comparison chart
- Saved ML models
- Reproducible transformer training code
- Updated `requirements.txt`
- Updated `README.md`

## 11. Implementation Approach

Before changing the code:

1. Inspect the existing project files.
2. Preserve features that already work.
3. Improve the project incrementally rather than rebuilding everything.
4. Test each stage before moving to the next.
5. Keep the project simple enough to explain during a class presentation.

After implementation, provide:

- A summary of what changed
- Any new dependencies installed
- Exact commands to run the project
- Instructions for training the models
- Instructions for starting the Streamlit application

## Final Project Pipeline

```text
UCI SMS Spam Dataset
        ↓
Text Preprocessing
        ↓
80/20 Train-Test Split
        ↓
TF-IDF Vectorization
        ↓
SMOTE on Training Data
        ↓
Traditional ML Models
        ↓
Logistic Regression / Naive Bayes
        ↓
Evaluation
        ↓
--------------------------------
Transformer Pipeline
        ↓
BERT / RoBERTa
        ↓
Evaluation
        ↓
--------------------------------
Compare All Models
        ↓
Streamlit Dashboard
        ↓
Live Spam/Ham Prediction
```

The goal is to produce a **simple but complete spam message detection project** that demonstrates text preprocessing, TF-IDF, handling imbalanced data with SMOTE, traditional machine learning, transformer models, visualization, model evaluation, and deployment through Streamlit.
