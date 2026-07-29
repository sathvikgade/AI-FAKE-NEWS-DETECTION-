"""
================================================================================
MODULE 2: FEATURE ENGINEERING
File: feature_engineering.py
================================================================================
Converts clean text into numerical feature vectors using TF-IDF (Term Frequency - 
Inverse Document Frequency) Vectorization and performs Train-Test splitting.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_tfidf_features(df, max_features=5000):
    """
    Transforms clean text into TF-IDF vector matrix.
    """
    print(f"[Feature Engineering] Extracting TF-IDF features (Max Features: {max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['target'].values
    print(f"[Feature Engineering] Feature matrix shape: {X.shape}")
    return X, y, vectorizer

def split_dataset(X, y, test_size=0.20, random_state=42):
    """
    Splits feature matrix into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Feature Engineering] Data Split -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import pandas as pd
    sample_df = pd.DataFrame({
        'clean_text': ['president trump speech election', 'scientists discover new galaxy space'],
        'target': [0, 1]
    })
    X, y, vec = extract_tfidf_features(sample_df)
    print("Vectorization test successful!")
