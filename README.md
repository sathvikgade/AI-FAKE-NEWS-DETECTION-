# AI-Powered Fake News Detection System 📰🤖

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Problem Statement
Build a modular, production-ready machine learning pipeline to classify news articles as **REAL** or **FAKE**. This repository contains dedicated `.py` modules for text preprocessing, feature engineering, multi-model training, evaluation, real-time prediction, and main orchestration.

---

## 📂 Modular Repository Architecture

```text
AI_FAKE_NEWS_PROJECT/
│
├── preprocessing.py         # Module 1: Data loading, lowercasing, Regex punctuation removal & stop-words filtering
├── feature_engineering.py   # Module 2: TF-IDF vectorization (5,000 features) & train-test split
├── train.py                 # Module 3: Multi-model training (KNN, LogReg, RandomForest, Neural Net) & model saving
├── evaluate.py              # Module 4: Performance metrics calculation & IEEE comparison matrix
├── predict.py               # Module 5: Real-time news article prediction CLI
├── main.py                  # End-to-end pipeline orchestrator
├── README.md                # Full project documentation & metrics report
└── .gitignore               # Excludes large CSV datasets from Git tracking
```

---

## 🗓️ 4-Week Internship Workflow

- **Week 1: Data Loading & Preprocessing (`preprocessing.py`)**
  - Converts text to lowercase.
  - Removes non-alphabetic characters & punctuation using Regex (`re.sub`).
  - Tokenizes text and filters out English stop words using NLTK.
- **Week 2: Feature Engineering & Vectorization (`feature_engineering.py`)**
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization (5,000 max features).
  - 80/20 Stratified Train-Test split.
- **Week 3: Multi-Model Building & Training (`train.py`)**
  - **KNN** (K-Nearest Neighbors - Non-Parametric)
  - **Logistic Regression** (Parametric Linear Classifier)
  - **Random Forest** (Ensemble Classifier)
  - **Simple Neural Network** (MLP Classifier / Deep Learning)
  - Saves best model & vectorizer to `saved_models/` using `joblib`.
- **Week 4: Model Evaluation & IEEE Documentation (`evaluate.py`)**
  - Calculates Accuracy, Precision, Recall, F1-Score, and Classification Reports.

---

## 📊 Experimental Results & Performance Comparison

| Algorithm | Model Type | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **KNN** | Non-Parametric | 58.60% | 98.44% | 13.24% | 23.33% |
| **Logistic Regression** | Parametric | 96.90% | 96.45% | 97.06% | 96.75% |
| **Simple Neural Network** | Deep Learning (MLP) | 96.20% | 96.40% | 95.59% | 95.99% |
| 🏆 **Random Forest** | Ensemble | **99.40%** | **99.16%** | **99.58%** | **99.37%** |

---

## 🚀 How to Run the Modules

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
```

### 2. Run Main Pipeline
```bash
python main.py
```

### 3. Run Real-Time Interactive Prediction
```bash
python predict.py
```

---

## 📄 License
This project is open-source under the MIT License.
