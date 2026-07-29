"""
================================================================================
MODULE 1: DATA PREPROCESSING
File: preprocessing.py
================================================================================
Responsible for loading raw news dataset, cleaning title and body text using
lowercasing, Regex punctuation removal, and stop-words filtering.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure NLTK Stopwords are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw news headline or body text:
    1. Converts text to lowercase.
    2. Removes agency prefixes like 'WASHINGTON (Reuters) -'.
    3. Removes non-alphabetic characters and punctuation using Regex.
    4. Filters out English stop words.
    """
    if not isinstance(text, str):
        return ""
    # Strip agency markers
    text = re.sub(r'^[A-Z\s,]+\s*\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'reuters', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    clean_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(clean_tokens)

def load_and_preprocess_data(file_path, sample_size=10000):
    """
    Loads raw dataset, cleans title & text columns, maps binary target labels.
    """
    print(f"[Preprocessing] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    target_col = 'class' if 'class' in df.columns else 'label'
    df = df[df[target_col].astype(str).str.strip().str.upper().isin(['FAKE', 'REAL'])].copy()
    
    if sample_size and len(df) > sample_size:
        print(f"[Preprocessing] Sampling {sample_size:,} balanced articles for optimal training...")
        fake_df = df[df[target_col].astype(str).str.strip().str.upper() == 'FAKE'].sample(n=sample_size//2, random_state=42)
        real_df = df[df[target_col].astype(str).str.strip().str.upper() == 'REAL'].sample(n=sample_size//2, random_state=42)
        df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    print("[Preprocessing] Cleaning news titles and body text...")
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    df['target'] = df[target_col].apply(lambda x: 1 if str(x).strip().upper() == 'REAL' else 0)
    
    print(f"[Preprocessing] Preprocessing complete! Total articles: {len(df):,}")
    return df

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "train.csv")
    cleaned_df = load_and_preprocess_data(data_path, sample_size=1000)
    print(cleaned_df[['title', 'clean_title', 'target']].head(2))
