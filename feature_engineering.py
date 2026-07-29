"""
================================================================================
MODULE 2: FEATURE ENGINEERING
File: feature_engineering.py
================================================================================
Transforms clean news text into numerical feature vectors using TF-IDF Vectorization
(Sublinear scaling, Unigrams + Bigrams, 10,000 max features) and performs Train-Test splitting.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_tfidf_features(df, max_features=10000):
    """
    Extracts TF-IDF features from title and body text for dual-head prediction.
    """
    print(f"[Feature Engineering] Extracting TF-IDF features (Max Features: {max_features:,})...")
    
    # Title TF-IDF Vectorizer
    title_vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_title = title_vec.fit_transform(df['clean_title'])
    
    # Text TF-IDF Vectorizer
    text_vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_text = text_vec.fit_transform(df['clean_text'])
    
    y = df['target'].values
    print(f"[Feature Engineering] Feature extraction completed! Vocabulary size: {X_title.shape[1]:,}")
    return X_title, X_text, y, title_vec, text_vec

def split_dataset(X, y, test_size=0.20, random_state=42):
    """
    Splits feature matrix into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Run main.py for feature engineering.")
