"""
================================================================================
PROJECT 1: AI-Powered Fake News Detection Using Text Classification
Path: C:\\Users\\LENOVO\\OneDrive\\Desktop\\AI_FAKE_NEWS_PROJECT
================================================================================

This script trains models directly on your 40,000+ article dataset!
- Week 1: Data Loading & Manual Text Preprocessing (Tokenization, Punctuation & Stopwords removal)
- Week 2: Feature Engineering (TF-IDF Vectorization) & Data Analysis
- Week 3: Model Building (KNN, Logistic Regression, Random Forest, Simple Neural Net / MLP)
- Week 4: Model Evaluation, Metrics Comparison & Visualizations
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Download NLTK Stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text_manual(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    clean_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

def run_pipeline():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(project_dir, "train.csv")
    
    print("="*60)
    print(" WEEK 1: DATA LOADING & PREPROCESSING")
    print("="*60)
    print(f"Loading dataset from: {train_path}")
    
    df = pd.read_csv(train_path, low_memory=False)
    print(f"Total news articles loaded: {len(df):,}")
    
    # Identify target column ('class' or 'label')
    target_col = 'class' if 'class' in df.columns else 'label'
    
    # Filter rows to only keep clean 'Fake' and 'Real' classes
    df = df[df[target_col].astype(str).str.strip().str.upper().isin(['FAKE', 'REAL'])].copy()
    
    print(f"\nCleaned Label distribution in '{target_col}':")
    print(df[target_col].value_counts())
    
    # Sample 5,000 rows for efficient multi-model execution
    if len(df) > 5000:
        print("\nSampling 5,000 articles for fast multi-model training...")
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        
    print("\nCleaning text data (lowercasing, punctuation, stopwords)...")
    df['clean_text'] = df['text'].apply(clean_text_manual)
    df['target'] = df[target_col].apply(lambda x: 1 if str(x).strip().upper() == 'REAL' else 0)
    
    print("\n" + "="*60)
    print(" WEEK 2: FEATURE ENGINEERING (TF-IDF)")
    print("="*60)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Feature matrix shape: {X.shape}")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    print("\n" + "="*60)
    print(" WEEK 3: MODEL BUILDING & TRAINING")
    print("="*60)
    models = {
        "KNN (Non-Parametric)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression (Parametric)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Simple Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    }
    
    predictions = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        print(f"  -> {name} completed!")
        
    print("\n" + "="*60)
    print(" WEEK 4: MODEL EVALUATION & IEEE METRICS MATRIX")
    print("="*60)
    results = []
    for name, preds in predictions.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results.append({
            "Algorithm": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%"
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    print("\nPipeline completed successfully! Results ready for your report.")

if __name__ == "__main__":
    run_pipeline()
