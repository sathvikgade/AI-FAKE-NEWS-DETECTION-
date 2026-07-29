# AI-Powered Fake News Detection Using Text Classification 📰🤖

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Problem Statement
Build an end-to-end machine learning pipeline from scratch to classify news articles as **REAL** or **FAKE**. This project implements text preprocessing, feature extraction (TF-IDF), model training, and performance evaluation across four different machine learning algorithms.

---

## 🗓️ 30-Day Workflow Implementation

- **Week 1: Data Loading & Preprocessing**
  - Text cleaning (lowercasing, punctuation & special character removal using Regex).
  - Manual tokenization and English stop-words removal using NLTK.
- **Week 2: Feature Engineering & Vectorization**
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization (5,000 max features).
  - 80/20 Stratified Train-Test split.
- **Week 3: Model Building**
  - **KNN** (K-Nearest Neighbors - Non-Parametric)
  - **Logistic Regression** (Parametric Linear Classifier)
  - **Random Forest** (Ensemble Classifier)
  - **Simple Neural Network** (MLP Classifier / Deep Learning)
- **Week 4: Model Evaluation & Documentation**
  - Evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  - IEEE format matrix generation.

---

## 📊 Experimental Results & Performance Comparison

| Algorithm | Model Type | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **KNN** | Non-Parametric | 58.60% | 98.44% | 13.24% | 23.33% |
| **Logistic Regression** | Parametric | 96.90% | 96.45% | 97.06% | 96.75% |
| **Simple Neural Network** | Deep Learning (MLP) | 96.20% | 96.40% | 95.59% | 95.99% |
| 🏆 **Random Forest** | Ensemble | **99.40%** | **99.16%** | **99.58%** | **99.37%** |

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### 2. Run the Main Pipeline
```bash
python main.py
```

---

## 📄 License
This project is open-source under the MIT License.
