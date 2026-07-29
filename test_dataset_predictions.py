"""
================================================================================
DATASET PREDICTION VERIFICATION SCRIPT
File: test_dataset_predictions.py
================================================================================
Trains model and directly evaluates 30 articles sampled from train.csv
(15 Real Dataset Articles + 15 Fake Dataset Articles).
Verifies that Real articles are classified as REAL and Fake articles as FAKE.
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    clean_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(clean_tokens)

def run_dataset_verification():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "train.csv")
    
    print("="*75)
    print("      DATASET PREDICTION VERIFICATION TEST (REAL VS FAKE DATASET)       ")
    print("="*75)
    print(f"Loading dataset from: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    target_col = 'label' if 'label' in df.columns else 'class'
    df = df[df[target_col].astype(str).str.strip().str.upper().isin(['FAKE', 'REAL'])].copy()
    
    # 15 Real Articles and 15 Fake Articles for evaluation test
    real_sample = df[df[target_col].astype(str).str.strip().str.upper() == 'REAL'].sample(n=15, random_state=100)
    fake_sample = df[df[target_col].astype(str).str.strip().str.upper() == 'FAKE'].sample(n=15, random_state=100)
    
    test_eval_set = pd.concat([real_sample, fake_sample]).reset_index(drop=True)
    
    # Exclude test evaluation items from training set
    train_df = df.drop(test_eval_set.index).copy()
    
    print(f"Training Model on {len(train_df):,} articles...")
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    train_df['target'] = train_df[target_col].apply(lambda x: 1 if str(x).strip().upper() == 'REAL' else 0)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['target'].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model Training Completed!\n")
    
    print("="*75)
    print("                    EVALUATING DATASET ARTICLES                         ")
    print("="*75)
    print(f"{'TYPE':<8} | {'EXPECTED':<8} | {'PREDICTED':<8} | {'CONFIDENCE':<10} | {'STATUS':<10} | HEADLINE")
    print("-" * 75)
    
    correct_count = 0
    total_count = len(test_eval_set)
    
    for idx, row in test_eval_set.iterrows():
        text_content = str(row['text'])
        headline = str(row['title'])[:35] + "..." if pd.notnull(row['title']) else text_content[:35] + "..."
        expected_label = str(row[target_col]).strip().upper()
        
        cleaned_in = clean_text(text_content)
        vec_in = vectorizer.transform([cleaned_in])
        
        pred = model.predict(vec_in)[0]
        prob = model.predict_proba(vec_in)[0]
        
        predicted_label = "REAL" if pred == 1 else "FAKE"
        confidence = prob[pred] * 100
        
        is_correct = (predicted_label == expected_label)
        if is_correct:
            correct_count += 1
            status = "[SUCCESS]"
        else:
            status = "[FAILED]"
            
        print(f"{expected_label:<8} | {expected_label:<8} | {predicted_label:<8} | {confidence:6.2f}%     | {status:<10} | {headline}")
        
    print("="*75)
    print(f"VERIFICATION ACCURACY SCORE: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)")
    print("="*75)

if __name__ == "__main__":
    run_dataset_verification()
