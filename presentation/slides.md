# DCS 404 - Artificial Intelligence & Machine Learning
## Final Presentation: Machine Learning & Transformer-Based Spam Message Detection
**Westcliff University | College of Technology & Engineering**  
**Presenter:** Sujal  
**Course:** DCS 404 - Artificial Intelligence & Machine Learning  
**Date:** Fall 2026  

---

## Slide 1: Title & Executive Overview
### Machine Learning & Transformer-Based Spam Message Detection
#### End-to-End NLP Pipeline: SMOTE Imbalance Remediation, Traditional ML & Transformers
- **The Operational Challenge:** Unsolicited SMS and email messaging (spam, smishing, scam promotions) presents escalating cybersecurity vulnerabilities, identity theft threats, and significant user friction.
- **The Machine Learning Research Question:** How accurately can traditional linear/Bayesian classifiers and deep bidirectional transformer models differentiate legitimate communications (**Ham**) from fraudulent or unsolicited messages (**Spam**)?
- **Professor Feedback Integration (update.md):**
  - Synthesize the course curriculum (Weeks 1–7) into an advanced, modern text classification architecture.
  - Implement **SMOTE (Synthetic Minority Over-sampling Technique)** to resolve class imbalance without data leakage.
  - Fine-tune deep **Transformer architectures (DistilBERT & DistilRoBERTa)** and evaluate trade-offs against traditional models.
  - Provide exploratory **Word Clouds** and deploy a 7-section **Streamlit web dashboard**.

> **Speaker Notes (Slide 1):**  
> *"Good morning/afternoon, instructor and colleagues. Welcome to my DCS 404 Final Project presentation: 'Machine Learning and Transformer-Based Spam Message Detection'. In response to our course milestones and professor feedback, this project synthesizes end-to-end data preparation, SMOTE class imbalance remediation, traditional statistical classifiers, deep transformer fine-tuning, and production web deployment."*

---

## Slide 2: Key Learning #1 — Data Hygiene, Anti-Leakage Protocol & SMOTE Balancing
### Stratified Partitioning, Strict Leakage Controls & Synthetic Balancing (Weeks 1–3)
- **Dataset Foundation:** UCI SMS Spam Collection corpus containing 5,572 labeled records.
- **Data Quality & Hygiene:**
  - Checked for null values: 0 missing values found.
  - Removed **403 exact duplicate entries**, yielding **5,169 distinct messages** (4,516 Ham [87.4%], 653 Spam [12.6%]).
- **Strict Anti-Data Leakage Protocol:**
  - Applied stratified **80/20 train/test splitting** (4,135 training samples / 1,034 held-out test samples), preserving the natural 12.6% minority spam ratio.
  - `TfidfVectorizer` (5,000 max features, unigrams + bigrams, English stopwords) was **fitted strictly on the training partition only** before transforming test data.
- **SMOTE Oversampling on Training Data Only:**
  - In the training set, class distribution was initially 3,613 Ham vs. 522 Spam (6.9:1 imbalance).
  - Applied **SMOTE exclusively to the training TF-IDF feature space**, synthetically interpolating minority samples to reach a balanced **3,613 Ham and 3,613 Spam** (1:1 ratio).
- **Test Set Integrity:**
  - The 1,034 held-out test samples were **never subjected to SMOTE**, ensuring all empirical evaluation reflects natural, unbiased real-world message distributions.

> **Speaker Notes (Slide 2):**  
> *"Our first major learning connects directly to Weeks 1 through 3: Data preparation, feature engineering, and avoiding data leakage. Natural message datasets are inherently imbalanced—spam represented just 12.6% of unique records. To avoid data leakage, we partitioned the dataset prior to vectorization, fitted TF-IDF only on training data, and applied SMOTE strictly to the training partition. Our 1,034 test messages remained untouched, preserving real-world validation integrity."*

---

## Slide 3: Exploratory Data Analysis & Text Visualizations
### Lexical Patterns, Word Clouds & Message Length Dynamics (Week 2)
- **Visual Explainability via Word Clouds:**
  - High-resolution word clouds provide immediate qualitative insight into distinct vocabulary distributions between spam and ham.
- **High-Risk Spam Keywords:**
  - Dominated by urgency markers, monetary incentives, and mandatory response instructions:  
    `"FREE"`, `"call"`, `"claim"`, `"prize"`, `"txt"`, `"won"`, `"cash"`, `"urgent"`, `"customer"`, `"reply"`, `"mobile"`, `"stop"`.
- **Legitimate Ham Lexicon:**
  - Characterized by conversational social discourse, scheduling, and personal colloquialisms:  
    `"come"`, `"good"`, `"time"`, `"got"`, `"will"`, `"day"`, `"ok"`, `"home"`, `"later"`, `"love"`, `"going"`, `"need"`.
- **Character Length Disparity:**
  - **Spam messages:** Average **138.8 characters** (median 149), clustering closely near the 160 SMS limit due to promotional copy and opt-out terms.
  - **Ham messages:** Average **71.0 characters** (median 52), reflecting concise, informal human exchange.
- **Feature Engineering Implications:**
  - Stark token frequency differences confirm that unigram and bigram TF-IDF weights provide rich linear separability.

> **Speaker Notes (Slide 3):**  
> *"Per the professor's feedback, we integrated Spam and Ham Word Clouds into our exploratory data analysis. The visual contrast is striking: spam messages are dominated by commercial incentives like FREE, claim, win, cash, and urgent. In contrast, ham messages reflect everyday human connection—words like come, good, time, and home. Furthermore, message length distributions reveal spam messages are nearly twice as long as legitimate texts."*

---

## Slide 4: Key Learning #2 — Traditional Machine Learning & SMOTE Trade-Offs
### Logistic Regression (SMOTE) & Multinomial Naive Bayes (Weeks 4–5)
- **Logistic Regression (Primary SMOTE Model):**
  - Binary classification modeling log-odds via sigmoid activation:
    $$P(Y = 1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j X_j)}}$$
  - **Impact of SMOTE on Logistic Regression:**  
    By oversampling spam feature vectors in the high-dimensional TF-IDF space, the linear boundary shifted to prioritize minority detection.  
    **Spam Recall surged from 78.63% to 90.08%**, cutting missed spam messages (**False Negatives**) from 28 down to 13.
  - Regularized with an L2 ridge penalty ($C=2.0$) to prevent coefficient explosion on frequent terms.
- **Multinomial Naive Bayes (Standard Benchmark):**
  - Computes posterior probabilities via Bayes' theorem with Laplace smoothing ($\alpha=0.2$):
    $$P(Y=c \mid X) \propto P(Y=c) \prod_{j=1}^d P(w_j \mid Y=c)^{x_j}$$
- **Why SMOTE Degrades Naive Bayes:**
  - SMOTE generates continuous synthetic feature vectors that violate discrete word count assumptions.
  - Imposing an artificial 50/50 prior $P(\text{Spam})=0.5$ inflates false alarms on the naturally imbalanced test set. Standard MNB achieved **98.26% precision** with only 2 false positives.
- **Pedagogical Takeaway:**
  - Demonstrates that balancing algorithms are not one-size-fits-all: margin-based classifiers (LR) benefit from SMOTE, whereas generative probabilistic models (MNB) excel on natural corpus distributions.

> **Speaker Notes (Slide 4):**  
> *"In Weeks 4 and 5, we studied supervised classification. Logistic Regression serves as our primary SMOTE model. SMOTE shifted the decision hyperplane, reducing false negatives from 28 to 13 and lifting spam recall to 90.08%. Crucially, we also addressed why SMOTE hurts Multinomial Naive Bayes: synthetic features distort discrete token likelihoods. Standard Naive Bayes delivered 98.26% precision with only 2 false positives out of 903 legitimate messages."*

---

## Slide 5: Key Learning #3 — Deep Contextual Transformers
### Fine-Tuned DistilBERT & DistilRoBERTa Architectures (Week 7)
- **Overcoming Bag-of-Words Limitations:**
  - TF-IDF representations discard word sequence, syntax, and polysemy. Transformer models use multi-head self-attention mechanisms to evaluate full bidirectional context.
- **DistilBERT (Sanh et al., 2019):**
  - A 6-layer distilled transformer model retaining **97% of BERT-base language understanding** while reducing parameter count by 40% (66M parameters) and speeding up inference by 60%.
- **DistilRoBERTa (Liu et al., 2019):**
  - Distilled variant of RoBERTa (82M parameters), pre-trained with dynamic masking and byte-level BPE on 160 GB of diverse uncompressed text for enhanced linguistic robustness.
- **Fine-Tuning Methodology:**
  - Fine-tuned for binary sequence classification using Hugging Face Transformers and PyTorch.
  - Utilized `[CLS]` pooled representations connected to a linear classification head with dropout (0.1), AdamW optimizer ($\text{lr} = 2 \times 10^{-5}$), and linear warmup.
- **Reproducibility & Google Colab Acceleration:**
  - Fully runnable with free T4 GPU acceleration in Google Colab; weights exported via Hugging Face `safetensors` for CPU inference.

> **Speaker Notes (Slide 5):**  
> *"In response to the requirement to incorporate modern transformer models, we fine-tuned DistilBERT and DistilRoBERTa. Unlike traditional bag-of-words approaches, transformers evaluate words in full bidirectional context, recognizing sentence intent, nuanced sarcasm, and disguised phrasing. We chose distilled architectures to achieve state-of-the-art accuracy while maintaining a lightweight footprint suitable for classroom reproduction and local deployment."*

---

## Slide 6: Empirical Results & Benchmark Evaluation
### Held-Out Test Set Performance Benchmark ($N = 1,034$)
All models evaluated on the identical held-out test split of 1,034 messages (903 Ham, 131 Spam):

| Model Architecture | Category | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | ROC-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Transformer | **98.84%** | 95.42% | **95.42%** | **95.42%** | 0.9975 | 6 | **6** |
| **DistilRoBERTa** | Transformer | **98.84%** | 97.60% | 93.13% | 95.31% | **0.9993** | 3 | 9 |
| **Multinomial Naive Bayes** | Traditional ML | 98.07% | **98.26%** | 86.26% | 91.87% | 0.9897 | **2** | 18 |
| **Logistic Regression (SMOTE)** | Traditional ML | 97.29% | 88.72% | 90.08% | 89.39% | 0.9903 | 15 | 13 |

#### Key Analytical Observations:
1. **Transformer Superiority in Balanced F1:** DistilBERT captured 125 of 131 spam messages with only 6 false negatives, achieving a project-high **95.42% F1-score**.
2. **DistilRoBERTa Precision Excellence:** Yielded **97.60% precision** with only 3 false positives while reaching an extraordinary **0.9993 ROC-AUC**.
3. **Traditional ML Strength:** Multinomial Naive Bayes delivered **98.26% precision** (2 false positives), and Logistic Regression with SMOTE caught over **90% of spam**.

> **Speaker Notes (Slide 6):**  
> *"Slide 6 presents our full empirical benchmark across all 4 models on the held-out test set. DistilBERT and DistilRoBERTa tied for top accuracy at 98.84%, with DistilBERT achieving our highest overall F1 score of 95.42%. DistilRoBERTa achieved 97.60% precision with an extraordinary 0.9993 ROC-AUC. Meanwhile, traditional ML models remained remarkably strong, with Naive Bayes producing only 2 false positives out of 903 legitimate messages."*

---

## Slide 7: Architectural Trade-Off Analysis
### Comparing Traditional ML vs. Transformers Across Operational Dimensions
Per the professor's explicit instructions, we conducted a multi-dimensional trade-off analysis rather than declaring a single "best" model:

| Evaluation Dimension | Traditional ML (LR / MNB) | Transformer (DistilBERT / RoBERTa) | Strategic Advantage |
| :--- | :--- | :--- | :--- |
| **Detection Performance** | 97.29% – 98.07% Acc, 89.4% – 91.9% F1 | **98.84% Acc, 95.3% – 95.4% F1** | **Transformers:** +3.5% to +6.0% higher F1-score |
| **Training Execution Time** | **0.01 – 0.05 seconds** | ~2,849 seconds (~47 min CPU / 3 min GPU) | **Traditional ML:** ~50,000× faster training |
| **Inference Latency** | **0.0008 ms / message** (~1.25M msg/sec) | 70.2 – 71.3 ms / message (~14 msg/sec) | **Traditional ML:** ~90,000× lower latency |
| **Hardware Requirements** | Standard CPU, <50 MB RAM | Requires GPU for training, 2+ GB RAM | **Traditional ML:** Runs on edge/microcontrollers |
| **Deployment & Size** | **~230 KB** single `.joblib` pipeline | ~260 MB weights + PyTorch dependencies | **Traditional ML:** Minimal Docker footprint |

#### Operational Decision Framework:
- **Deploy Traditional ML (Naive Bayes / Logistic Regression):** In real-time high-throughput environments (telecom SMS switches, edge network relays) where sub-millisecond latency and low compute costs are mandatory.
- **Deploy Transformers (DistilBERT / DistilRoBERTa):** In high-security enterprise email gateways, financial fraud auditing, or second-stage triage systems where catching disguised threats justifies compute overhead.

> **Speaker Notes (Slide 7):**  
> *"A critical requirement from our professor was analyzing the trade-offs between traditional ML and transformers. No single architecture dominates all scenarios. Transformers provide superior semantic accuracy and higher F1 scores. However, traditional ML trains 50,000 times faster and executes inference 90,000 times faster with a 230-kilobyte footprint. In telecom SMS gateways handling millions of messages a second, traditional ML is ideal. In high-stakes enterprise cybersecurity, transformers are worth the investment."*

---

## Slide 8: Interactive Streamlit Dashboard & Live Predictions
### Production Web Interface, Sample Inferences & Ethical Principles (Week 8)
- **7-Section Streamlit Web Dashboard:**
  1. **Home:** System overview, pipeline architecture, and quickstart guidance.
  2. **Dataset Overview:** Raw metrics, duplicate pruning, and class distributions.
  3. **Data Exploration:** Length distributions, character dynamics, and class metrics.
  4. **Spam & Ham Word Clouds:** Interactive high-resolution word clouds.
  5. **Model Performance:** Confusion matrices, ROC curves, and class reports.
  6. **Model Comparison:** Interactive multi-metric comparative bar charts and latency tables.
  7. **Live Spam Prediction:** Interactive playground supporting message analysis across all 4 models.
- **Real-World Inference Test Cases:**
  - *Sample 1 (High-Risk Spam):*  
    `"URGENT! You won £1,000 cash. Reply WIN to claim."` $\rightarrow$ **SPAM (99.9% Confidence)**
  - *Sample 2 (Conversational Ham):*  
    `"Hi Sujal, are we still meeting in conference room B at 3:30 PM?"` $\rightarrow$ **HAM (99.8% Confidence)**
  - *Sample 3 (Disguised Marketing Spam):*  
    `"Free entry into the weekly £100 prize draw. Text WON to 85060"` $\rightarrow$ **SPAM (99.7% Confidence)**
- **Ethical & Privacy Considerations:**
  - **Data Privacy:** User messages are processed in ephemeral session memory without persistent logging.
  - **Algorithmic Fairness:** Audited false positive rates (down to 2 FP for MNB, 3 FP for RoBERTa) to prevent wrongful suppression of critical communications.

> **Speaker Notes (Slide 8):**  
> *"Finally, we deployed our complete solution in an interactive Streamlit application structured into 7 dedicated sections. In live testing, suspicious messages with prize and urgency claims trigger SPAM predictions with over 99.9% confidence. Everyday conversational messages are accurately recognized as HAM with 99.8% confidence. We also ensured strict privacy protection by processing user text ephemerally without persistent database storage. Thank you very much, and I welcome your questions!"*

---

## Presentation Delivery & Q&A Preparation
### Expected Questions & Model Answers:
1. **Q: Why not apply SMOTE to the test set?**  
   *A: Applying SMOTE to the test set causes artificial data leakage and inflates performance metrics. The test set must represent the true, unadulterated distribution of incoming real-world traffic.*
2. **Q: Why does SMOTE harm Multinomial Naive Bayes?**  
   *A: Multinomial Naive Bayes relies on discrete integer word occurrences to compute conditional likelihoods. SMOTE generates synthetic continuous numbers through interpolation and forces equal class priors, which degrades Bayesian test precision.*
3. **Q: Which model would you deploy in a real production system?**  
   *A: A hybrid two-tier architecture: deploy Naive Bayes or Logistic Regression at the edge to filter 95% of obvious messages in under 1 millisecond, and route ambiguous or borderline messages to DistilBERT for deep contextual analysis.*
