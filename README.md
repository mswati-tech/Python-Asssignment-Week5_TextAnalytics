# Toxic Comment Classification – Text Analytics Project

## 📌 Introduction
This project focuses on detecting **toxic comments** in a text dataset using Natural Language Processing (NLP) and machine learning models.  
It follows a complete **data preprocessing → feature selection → model training → evaluation** pipeline.

The aim is to:
1. **Clean** the raw dataset by removing noise such as HTML tags, unwanted tokens like `sdata`, `edata`, and repetitive placeholders like `newlinetoken`.
2. **Refine** the cleaned dataset to remove low-importance words based on Logistic Regression coefficients.
3. **Experiment** with multiple models (Naive Bayes & Logistic Regression) and parameters (`threshold`, `max_features`, `n-grams`) to improve performance.
4. **Evaluate** models and analyze why certain settings work better or worse.

---

## 🛠 Methodology

### 1️⃣ Data Cleaning
- Implemented an **OOP-based cleaner class** `col_cleaner` with:
  - **Stopword removal** (NLTK default + extra stopwords).
  - **Tokenization** (NLTK `word_tokenize`).
  - **Noise removal**: stripped `sdata`, `edata`, `newlinetoken` patterns.
  - **Regex-based punctuation and digit removal**.
  - **Lemmatization** (WordNetLemmatizer).
  - **Stemming** for `-ing` words (PorterStemmer).

**Hit and trial:**  
- Initially only removed basic stopwords, but dataset still had non-informative terms like *padding*, *style*, *edit*, *role*.  
- Decided to apply **refining** to remove **low-importance words** using a Logistic Regression-based feature importance approach.

---

### 2️⃣ Refining Cleaned Data
- Used **TF-IDF vectorization** (`max_features = 10000`).
- Trained a Logistic Regression model.
- Extracted **low-importance words** where `-threshold < coef < threshold`.
- Experimented with `threshold`:
  - **0.02** → removed too many words, including some toxic ones (*ego*, *abusing*).
  - **0.01** → still removed useful words.
  - **0.0005** → balanced removal of noise without losing important toxic indicators.
- Re-cleaned datasets by adding these low-importance words to stopwords.

---

### 3️⃣ Model Training
Two approaches were tested:

#### **Generative Model – Naive Bayes (MultinomialNB)**
- Works well for text classification with word counts/TF-IDF.
- Advantage: fast and interpretable.
- Drawback: struggles when toxic vocabulary is rare.

#### **Discriminative Model – Logistic Regression**
- Learns decision boundaries directly from data.
- Advantage: can incorporate class weights for imbalance handling.
- Drawback: requires tuning `max_iter`, regularization, and feature space.

Both models:
- Used `max_features = 20000` in TF-IDF.
- Were tested on both unigrams and n-grams (`ngram_range=(1,2)`).

---

### 4️⃣ Experiments & Observations

#### **Unigram-only runs**
- Accuracy: ~87% (earlier run before n-grams).
- Issue: poor recall for toxic class (~0.0 in extreme case due to imbalance).
- Class imbalance caused high accuracy but low F1 for minority class.

#### **Refining with thresholds**
- Higher thresholds (0.02) → removed too many potentially toxic words.
- Lower thresholds (0.0005) → retained more toxic terms, improving model ability to detect them.

#### **Adding n-grams**
- Accuracy dropped to ~69% (NB) and ~74% (LR).
- Toxic class recall slightly improved but still low (~0.14–0.27).
- Larger feature space increased sparsity and noise.

---

## 📊 Results Summary

| Model                 | Accuracy | Toxic Precision | Toxic Recall | Toxic F1 |
|----------------------|----------|-----------------|--------------|----------|
| Naive Bayes (unigrams) | ~87%     | ~0.0–0.15       | ~0.0–0.30    | ~0.20    |
| Logistic Reg. (unigrams) | ~87%  | ~0.0–0.14       | ~0.0–0.23    | ~0.17    |
| Naive Bayes (n-grams) | ~69%     | ~0.14           | ~0.27        | ~0.19    |
| Logistic Reg. (n-grams) | ~74%  | ~0.12           | ~0.16        | ~0.14    |

---

## 📌 Conclusion & Learnings
- **Data cleaning & refining** significantly reduced noise and improved interpretability.
- **Threshold tuning** for low-importance words was critical — too high removed key toxic indicators.
- **max_features = 20000** provided a good trade-off between vocabulary size and performance.
- **N-grams** did not improve performance here due to:
  - Sparse toxic data.
  - Increase in feature space without enough corresponding examples.
- **Main limitation:** class imbalance — models favor non-toxic predictions.
- **Next steps for improvement:**
  - Apply **SMOTE** or class-weighting.
  - Use **custom toxic lexicon**.
  - Experiment with **SVM** or **transformer-based models** (BERT).

---

## 📂 File Structure
├── train.csv
├── valid.csv
├── test.csv
├── train_cleaned.csv
├── valid_cleaned.csv
├── test_cleaned.csv
├── train_cleaned_refined.csv
├── valid_cleaned_refined.csv
├── test_cleaned_refined.csv
├── low_importance_words.csv
├── AssignmentWeek_5_code.py
└── README.md
