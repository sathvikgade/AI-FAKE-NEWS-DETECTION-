"""
================================================================================
MODULE 1: DATA PREPROCESSING
File: preprocessing.py
================================================================================
Responsible for loading the raw dataset, lowercasing, removing punctuation/special
characters using Regular Expressions (Regex), manual tokenization, and stop-words removal.
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
    Cleans raw news text:
    1. Converts text to lowercase.
    2. Removes non-alphabetic characters and punctuation.
    3. Tokenizes text into words.
    4. Filters out English stop words and short tokens.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    clean_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

def load_and_preprocess_data(file_path, sample_size=5000):
    """
    Loads dataset from file_path, cleans labels, applies clean_text, and returns DataFrame.
    """
    print(f"[Preprocessing] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    target_col = 'class' if 'class' in df.columns else 'label'
    df = df[df[target_col].astype(str).str.strip().str.upper().isin(['FAKE', 'REAL'])].copy()
    
    if sample_size and len(df) > sample_size:
        print(f"[Preprocessing] Sampling {sample_size:,} articles for optimal performance...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    print("[Preprocessing] Cleaning news text (lowercasing, punctuation, stopwords)...")
    df['clean_text'] = df['text'].apply(clean_text)
    df['target'] = df[target_col].apply(lambda x: 1 if str(x).strip().upper() == 'REAL' else 0)
    
    print(f"[Preprocessing] Preprocessing complete! Cleaned dataset rows: {len(df):,}")
    return df

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "train.csv")
    cleaned_df = load_and_preprocess_data(data_path, sample_size=1000)
    print(cleaned_df[['text', 'clean_text', 'target']].head(2))
