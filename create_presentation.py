"""
create_presentation.py
----------------------
Generates an 8-slide widescreen PowerPoint presentation (.pptx)
for DCS 404: Artificial Intelligence & Machine Learning Final Project.

Incorporates all updates specified in update.md:
1. SMOTE Class Imbalance Remediation & Anti-Leakage Methodology
2. Exploratory Data Analysis with Spam & Ham Word Clouds
3. Traditional Machine Learning: Logistic Regression (SMOTE) & Multinomial Naive Bayes
4. Transformer Architectures: DistilBERT & DistilRoBERTa (Colab GPU Fine-Tuning)
5. Comprehensive Empirical Benchmarking (Accuracy, Precision, Recall, F1, Latency)
6. Multi-Dimensional Architectural Trade-Off Analysis
7. Production Streamlit Web Dashboard (7 Dedicated Sections) & Live Inferences
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
OUTPUT_DIR = os.path.join(BASE_DIR, 'presentation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette - Professional Deep Tech Theme
NAVY = RGBColor(15, 23, 42)          # #0F172A (Deep Slate Navy)
PRIMARY_BLUE = RGBColor(30, 58, 138) # #1E3A8A (Classic Navy Blue)
ACCENT_BLUE = RGBColor(37, 99, 235)  # #2563EB (Royal Blue)
SLATE = RGBColor(71, 85, 105)        # #475569 (Body Slate)
DARK = RGBColor(15, 23, 42)          # #0F172A
WHITE = RGBColor(255, 255, 255)
LIGHT_BG = RGBColor(248, 250, 252)   # #F8FAFC
CARD_BG = RGBColor(241, 245, 249)    # #F1F5F9
BORDER_COLOR = RGBColor(203, 213, 225)
ACCENT_GREEN = RGBColor(22, 163, 74)
ACCENT_RED = RGBColor(220, 38, 38)
ACCENT_PURPLE = RGBColor(124, 58, 237)

blank_layout = prs.slide_layouts[6]

def add_header(slide, title_text, category_text="DCS 404: AI & MACHINE LEARNING | FINAL PROJECT"):
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_BLUE
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(25)
    p_title.font.bold = True
    p_title.font.color.rgb = PRIMARY_BLUE
    p_title.space_before = Pt(4)

def add_bullet(tf, bold_prefix, text, space=8, level=0, font_size=13):
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.space_before = Pt(space)
    p.level = level
    
    run_bold = p.add_run()
    run_bold.text = bold_prefix + " "
    run_bold.font.bold = True
    run_bold.font.size = Pt(font_size)
    run_bold.font.color.rgb = DARK
    
    run_text = p.add_run()
    run_text.text = text
    run_text.font.size = Pt(font_size)
    run_text.font.color.rgb = SLATE

# ==========================================
# SLIDE 1: Title Slide
# ==========================================
s1 = prs.slides.add_slide(blank_layout)

# Decorative background rectangle
bg_shape = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg_shape.fill.solid()
bg_shape.fill.fore_color.rgb = PRIMARY_BLUE
bg_shape.line.fill.background()

# Title text box
tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
tf1 = tbox.text_frame
tf1.word_wrap = True

p0 = tf1.paragraphs[0]
p0.text = "WESTCLIFF UNIVERSITY | DCS 404 FINAL PROJECT"
p0.font.size = Pt(14)
p0.font.bold = True
p0.font.color.rgb = RGBColor(147, 197, 253)

p1 = tf1.add_paragraph()
p1.text = "Machine Learning & Transformer-Based Spam Message Detection"
p1.font.size = Pt(34)
p1.font.bold = True
p1.font.color.rgb = WHITE
p1.space_before = Pt(14)

p2 = tf1.add_paragraph()
p2.text = "End-to-End NLP Pipeline: SMOTE Imbalance Remediation, Traditional ML & Transformers"
p2.font.size = Pt(18)
p2.font.color.rgb = RGBColor(226, 232, 240)
p2.space_before = Pt(12)

p3 = tf1.add_paragraph()
p3.text = "Presenter: Sujal  |  Course: DCS 404 - Artificial Intelligence & Machine Learning  |  Fall 2026"
p3.font.size = Pt(14)
p3.font.color.rgb = RGBColor(191, 219, 254)
p3.space_before = Pt(28)

p4 = tf1.add_paragraph()
p4.text = "Featuring: TF-IDF • SMOTE • Logistic Regression • Naive Bayes • DistilBERT • DistilRoBERTa • Streamlit UI"
p4.font.size = Pt(13)
p4.font.color.rgb = RGBColor(254, 240, 138)
p4.space_before = Pt(14)

s1.notes_slide.notes_text_frame.text = (
    "Good morning/afternoon, instructor and colleagues. Welcome to my DCS 404 Final Project presentation: "
    "'Machine Learning and Transformer-Based Spam Message Detection'. "
    "Following my professor's feedback, this updated project incorporates class imbalance remediation via SMOTE, "
    "strict anti-data leakage pipelines, fine-tuned transformer architectures (DistilBERT and DistilRoBERTa), "
    "exploratory word clouds, and a complete 7-section Streamlit deployment."
)

# ==========================================
# SLIDE 2: Key Learning #1 - Problem Framing, Data Hygiene & SMOTE
# ==========================================
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Key Learning #1: Data Hygiene, Anti-Leakage & SMOTE Balancing")

c2_left = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.6), Inches(5.3))
tf2 = c2_left.text_frame
tf2.word_wrap = True

add_bullet(tf2, "Dataset Foundation:", "UCI SMS Spam Collection corpus containing 5,572 labeled records (ham / spam).", space=0)
add_bullet(tf2, "Data Cleaning & Hygiene:", "Checked missing values (0 nulls). Pruned 403 exact duplicates to yield 5,169 distinct messages (4,516 Ham [87.4%], 653 Spam [12.6%]).")
add_bullet(tf2, "Anti-Data Leakage Protocol:", "Split dataset strictly first using stratified 80/20 train/test splitting (4,135 train / 1,034 test). The test partition preserves authentic class prevalence.")
add_bullet(tf2, "TF-IDF Vectorization:", "Fitted TfidfVectorizer (5,000 max features, unigrams + bigrams, English stop words) strictly on training text only before transforming test data.")
add_bullet(tf2, "SMOTE Applied Only to Train:", "Applied Synthetic Minority Over-sampling Technique (SMOTE) strictly to training TF-IDF features, balancing 3,613 Ham and 522 Spam to an equal 3,613:3,613 distribution.")
add_bullet(tf2, "Test Set Preservation:", "The 1,034 test messages were never oversampled or modified, ensuring real-world, unbiased empirical evaluation.")

# Right Column: SMOTE Distribution Image
smote_img = os.path.join(FIG_DIR, 'smote_class_distribution.png')
if os.path.exists(smote_img):
    s2.shapes.add_picture(smote_img, Inches(7.6), Inches(1.7), width=Inches(5.0))

s2.notes_slide.notes_text_frame.text = (
    "Our first major learning connects directly to Weeks 1 through 3: Data preparation, feature engineering, and avoiding data leakage. "
    "In natural messaging corpora, spam is heavily outnumbered—here representing only 12.6% of messages. "
    "Crucially, we avoided data leakage by fitting TF-IDF only on training data and applying SMOTE exclusively to the training set. "
    "Our test set remained 100% authentic with 903 ham and 131 spam messages, preserving real-world validity."
)

# ==========================================
# SLIDE 3: Exploratory Data Analysis & Word Clouds
# ==========================================
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Exploratory Data Analysis: Lexical Patterns & Word Clouds")

c3_left = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
tf3 = c3_left.text_frame
tf3.word_wrap = True

add_bullet(tf3, "Lexical Explainability:", "Word clouds provide immediate visual explainability, revealing distinguishing linguistic characteristics between unsolicited and legitimate messages.", space=0)
add_bullet(tf3, "Spam Keyword Triggers:", "Dominated by commercial incentives, urgent calls-to-action, and prizes: 'FREE', 'call', 'claim', 'prize', 'txt', 'won', 'cash', 'urgent', 'customer', 'reply'.")
add_bullet(tf3, "Ham Conversational Vocabulary:", "Reflects everyday personal communication, social coordination, and colloquialisms: 'come', 'good', 'time', 'got', 'will', 'day', 'ok', 'home', 'later', 'love'.")
add_bullet(tf3, "Message Length Disparity:", "Spam messages average 138.8 characters (median 149), constrained near the 160 SMS limit due to disclaimers. Ham messages average 71.0 characters (median 52).")
add_bullet(tf3, "Engineering Takeaway:", "Token frequency disparity confirms that TF-IDF n-grams provide strong discriminative power for linear and probabilistic classifiers.")

# Right Column: Side-by-side or stacked Word Clouds
spam_wc_img = os.path.join(FIG_DIR, 'spam_wordcloud.png')
ham_wc_img = os.path.join(FIG_DIR, 'ham_wordcloud.png')
if os.path.exists(spam_wc_img) and os.path.exists(ham_wc_img):
    s3.shapes.add_picture(spam_wc_img, Inches(7.0), Inches(1.7), width=Inches(5.6))
    s3.shapes.add_picture(ham_wc_img, Inches(7.0), Inches(4.3), width=Inches(5.6))

s3.notes_slide.notes_text_frame.text = (
    "Per the professor's feedback, we integrated Spam and Ham Word Clouds into our exploratory data analysis. "
    "Spam messages are packed with urgency and financial triggers like FREE, claim, win, cash, and urgent. "
    "In contrast, legitimate ham messages reflect natural social discourse—words like 'come', 'good', 'time', and 'home'. "
    "Furthermore, message length analysis shows spam messages average nearly double the character length of legitimate messages."
)

# ==========================================
# SLIDE 4: Key Learning #2 - Traditional ML & Naive Bayes SMOTE Trade-off
# ==========================================
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Key Learning #2: Traditional Machine Learning & SMOTE Trade-Offs")

c4_left = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.5), Inches(5.3))
tf4 = c4_left.text_frame
tf4.word_wrap = True

add_bullet(tf4, "Logistic Regression (Primary SMOTE Model):", "Binary log-odds model using sigmoid activation: P(Y=1|X) = 1 / (1 + e^-(beta*X)). Optimized with L2 ridge regularization (C=2.0).", space=0)
add_bullet(tf4, "Impact of SMOTE on Logistic Regression:", "SMOTE synthesis expanded the minority spam decision region, dramatically cutting False Negatives from 28 to 13 and boosting Spam Recall from 78.63% to 90.08%.")
add_bullet(tf4, "Multinomial Naive Bayes Benchmark:", "Models feature probabilities using Bayes' rule with Laplace smoothing (alpha=0.2). Highly effective for sparse discrete word frequencies.")
add_bullet(tf4, "Why SMOTE Harms Naive Bayes:", "SMOTE generates continuous fractional features that violate discrete multinomial count assumptions and artificially imposes a 50/50 prior, causing test precision to drop. Standard MNB preserves 98.26% precision.")
add_bullet(tf4, "Pedagogical Insight:", "Demonstrates that oversampling techniques are not universally beneficial; geometric classifiers (LR) benefit from SMOTE, while generative Bayesian models perform best on natural frequencies.")

# Right Column: Top Keywords / Confusion Matrix
top_kw = os.path.join(FIG_DIR, 'top_spam_keywords.png')
if os.path.exists(top_kw):
    s4.shapes.add_picture(top_kw, Inches(7.5), Inches(1.7), width=Inches(5.1))

s4.notes_slide.notes_text_frame.text = (
    "In Weeks 4 and 5, we studied supervised classification. Logistic Regression serves as our main SMOTE-based model. "
    "SMOTE dramatically improved Logistic Regression's recall from 78.6% to 90.08%, cutting missed spam messages in half. "
    "However, we also analyzed why SMOTE does not pair well with Multinomial Naive Bayes: synthetic interpolation distorts discrete token likelihoods. "
    "Standard Naive Bayes achieved 98.26% precision with only 2 false positives out of 903 legitimate messages."
)

# ==========================================
# SLIDE 5: Key Learning #3 - Deep Contextual Transformers
# ==========================================
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Key Learning #3: Transformer Architectures (DistilBERT & RoBERTa)")

c5_left = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.5), Inches(5.3))
tf5 = c5_left.text_frame
tf5.word_wrap = True

add_bullet(tf5, "Beyond Bag-of-Words:", "Traditional TF-IDF ignores syntactic order and semantic nuance. Transformers use multi-head self-attention to capture bidirectional contextual relationships.", space=0)
add_bullet(tf5, "DistilBERT Architecture:", "6-layer distilled transformer retaining 97% of BERT-base capability with 40% fewer parameters (66M) and 60% faster inference speed.")
add_bullet(tf5, "DistilRoBERTa Architecture:", "Distilled variant of RoBERTa (82M parameters), pre-trained on 160GB of text with byte-pair encoding and dynamic masking for superior semantic subtlety.")
add_bullet(tf5, "Fine-Tuning Methodology:", "Fine-tuned for binary sequence classification using Hugging Face Transformers and PyTorch with AdamW optimizer, warmup scheduling, and [CLS] pooling.")
add_bullet(tf5, "Colab GPU Training & Export:", "Reproducibly fine-tuned in Google Colab using free T4 GPU acceleration, exported to safetensors, and loaded for local CPU inference in Streamlit.")

# Right Column: ROC Curves Image
roc_img = os.path.join(FIG_DIR, 'roc_curves.png')
if os.path.exists(roc_img):
    s5.shapes.add_picture(roc_img, Inches(7.5), Inches(1.7), width=Inches(5.1))

s5.notes_slide.notes_text_frame.text = (
    "In response to the professor's requirement to add deep learning, we fine-tuned two state-of-the-art transformer models: DistilBERT and DistilRoBERTa. "
    "Unlike TF-IDF, transformers evaluate words in full bidirectional context. "
    "We chose distilled versions to keep memory footprint and training time realistic while maintaining near-perfect discrimination. "
    "Both transformer models achieved exceptional ROC-AUC scores exceeding 0.997."
)

# ==========================================
# SLIDE 6: Empirical Results & Benchmark Evaluation
# ==========================================
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Empirical Evaluation: Test Set Benchmark (N = 1,034)")

c6_left = s6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.5), Inches(5.3))
tf6 = c6_left.text_frame
tf6.word_wrap = True

add_bullet(tf6, "Test Split Integrity:", "All models evaluated on the identical, untouched held-out test split of 1,034 messages (903 Ham, 131 Spam).", space=0)
add_bullet(tf6, "DistilBERT (Top Overall):", "98.84% Accuracy | 95.42% Precision | 95.42% Recall | 95.42% F1 | 0.9975 ROC-AUC | 6 FP, 6 FN")
add_bullet(tf6, "DistilRoBERTa (Top Precision):", "98.84% Accuracy | 97.60% Precision | 93.13% Recall | 95.31% F1 | 0.9993 ROC-AUC | 3 FP, 9 FN")
add_bullet(tf6, "Multinomial Naive Bayes:", "98.07% Accuracy | 98.26% Precision | 86.26% Recall | 91.87% F1 | 0.9897 ROC-AUC | 2 FP, 18 FN")
add_bullet(tf6, "Logistic Regression (SMOTE):", "97.29% Accuracy | 88.72% Precision | 90.08% Recall | 89.39% F1 | 0.9903 ROC-AUC | 15 FP, 13 FN")
add_bullet(tf6, "Key Benchmark Takeaway:", "Transformers deliver the highest balanced detection (F1 > 95.3%). However, traditional models achieve >97% accuracy with negligible latency.")

# Right Column: Model Comparison Bar Chart
comp_img = os.path.join(FIG_DIR, 'model_comparison.png')
if os.path.exists(comp_img):
    s6.shapes.add_picture(comp_img, Inches(7.5), Inches(1.7), width=Inches(5.1))

s6.notes_slide.notes_text_frame.text = (
    "Slide 6 presents our empirical benchmark across all four models on the held-out test set of 1,034 samples. "
    "DistilBERT and DistilRoBERTa tied for the highest accuracy at 98.84%, with DistilBERT achieving an outstanding 95.42% F1 score. "
    "DistilRoBERTa achieved 97.60% precision with just 3 false positives. "
    "Meanwhile, Multinomial Naive Bayes held an exceptional 98.26% precision with only 2 false positives, and Logistic Regression with SMOTE caught 90.08% of all spam."
)

# ==========================================
# SLIDE 7: Architectural Trade-Off Analysis
# ==========================================
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Architectural Trade-Off Analysis: Traditional ML vs. Transformers")

c7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
tf7 = c7.text_frame
tf7.word_wrap = True

add_bullet(tf7, "Performance & Accuracy:", "Transformers outperform traditional ML on F1-score (+3.5% to +6.0%) and ROC-AUC (0.999 vs 0.990) by deciphering conversational nuance, typos, and disguised links.", space=0)
add_bullet(tf7, "Training Speed Disparity:", "Traditional ML trains in 0.01 to 0.05 seconds. Transformers require ~45 minutes on GPU or multiple hours on CPU (~50,000x difference).", space=10)
add_bullet(tf7, "Inference Latency & Throughput:", "Traditional ML predicts in 0.0008 ms per message (~1.25 million msg/sec throughput). Transformers require ~70 ms per message (~14 msg/sec on CPU) — a 90,000x latency advantage for ML.", space=10)
add_bullet(tf7, "Hardware & Compute Footprint:", "Traditional models require <10 MB RAM and run on microcontrollers. Transformers require 260 MB disk storage, PyTorch runtimes, and hundreds of megabytes of RAM.", space=10)
add_bullet(tf7, "Operational Decision Framework:", "High-volume telecom SMS gateways should deploy Naive Bayes or Logistic Regression for real-time edge filtering, while high-value fraud detection and forensic auditing benefit from Transformers.", space=10)

s7.notes_slide.notes_text_frame.text = (
    "A central requirement of the professor's update was comparing the trade-offs between traditional ML and transformer models. "
    "We must not simply declare one model 'the winner'. "
    "While transformers yield higher F1 scores, traditional ML trains 50,000 times faster and runs inference 90,000 times faster. "
    "In a live telecommunications gateway processing millions of SMS messages per second, traditional ML is drastically more cost-effective. "
    "For high-risk corporate email or fraud investigations, the contextual accuracy of transformers is well worth the extra latency."
)

# ==========================================
# SLIDE 8: Streamlit Dashboard & Live Predictions
# ==========================================
s8 = prs.slides.add_slide(blank_layout)
add_header(s8, "Interactive Streamlit Dashboard & Live Inferences")

c8_left = s8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.4), Inches(5.3))
tf8 = c8_left.text_frame
tf8.word_wrap = True

add_bullet(tf8, "7-Section Streamlit Architecture:", "1. Home  2. Dataset Overview  3. Data Exploration  4. Word Clouds  5. Model Performance  6. Model Comparison  7. Live Spam Prediction.", space=0)
add_bullet(tf8, "Live Inference Engine:", "Allows interactive testing across all 4 models with real-time confidence scoring and classification badges (SPAM vs. HAM).")
add_bullet(tf8, "Test Sample 1 (Spam):", "'URGENT! You won £1,000 cash. Reply WIN to claim.' -> Predicted: SPAM (99.9% Confidence).")
add_bullet(tf8, "Test Sample 2 (Ham):", "'Hi Sujal, are we still meeting in conference room B at 3:30 PM?' -> Predicted: HAM (99.8% Confidence).")
add_bullet(tf8, "Test Sample 3 (Disguised Spam):", "'Free entry into the weekly £100 prize draw. Text WON to 85060' -> Predicted: SPAM (99.7% Confidence).")
add_bullet(tf8, "Ethical & Privacy Assurance:", "Zero persistent message logging, ephemeral memory inference, and bias auditing to protect user confidentiality.")

# Right Column: Confusion Matrices Grid Image
cm_img = os.path.join(FIG_DIR, 'confusion_matrices.png')
if os.path.exists(cm_img):
    s8.shapes.add_picture(cm_img, Inches(7.4), Inches(1.8), width=Inches(5.2))

s8.notes_slide.notes_text_frame.text = (
    "Finally, we packaged the entire system into a production-grade Streamlit web dashboard structured into 7 distinct sections. "
    "In our live tests, suspicious messages with prize and urgency claims trigger SPAM predictions with over 99.9% confidence. "
    "Legitimate conversational messages are cleanly identified as HAM with 99.8% confidence. "
    "We also addressed ethics and data privacy by ensuring zero persistent message storage during inference. "
    "Thank you for your time, and I welcome any questions!"
)

# Save presentation
pptx_path = os.path.join(OUTPUT_DIR, 'DCS404_Spam_Detection_Presentation.pptx')
prs.save(pptx_path)
print(f"PowerPoint presentation generated at: {pptx_path}")
