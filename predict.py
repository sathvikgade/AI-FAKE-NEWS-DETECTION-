"""
================================================================================
MODULE 5: REAL-TIME PREDICTION SCRIPT
File: predict.py
================================================================================
Takes a news article text string as input and predicts whether it is REAL or FAKE
using the saved Random Forest model and TF-IDF Vectorizer.
"""

import os
import joblib
from preprocessing import clean_text

def load_prediction_artifacts(model_dir="saved_models"):
    """
    Loads trained model and vectorizer from disk.
    """
    model_path = os.path.join(model_dir, "best_model.pkl")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Trained model or vectorizer not found. Please run main.py or train.py first!")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def predict_news_article(news_text, model=None, vectorizer=None):
    """
    Predicts if a news article text is REAL or FAKE.
    Returns label string and probability score.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_prediction_artifacts()
        
    cleaned_input = clean_text(news_text)
    vectorized_input = vectorizer.transform([cleaned_input])
    
    prediction = model.predict(vectorized_input)[0]
    probabilities = model.predict_proba(vectorized_input)[0]
    
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = probabilities[prediction] * 100
    
    return label, confidence

if __name__ == "__main__":
    print("="*60)
    print("      AI-POWERED FAKE NEWS DETECTOR - REAL-TIME PREDICT      ")
    print("="*60)
    
    # Test Sample News Articles
    sample_news_1 = "The Federal Reserve announced an interest rate decision today following the monthly economic policy meeting."
    sample_news_2 = "Scientists discover secret alien base under the Antarctic ice caps hidden for centuries!"
    
    try:
        model, vec = load_prediction_artifacts()
        
        label1, conf1 = predict_news_article(sample_news_1, model, vec)
        print(f"\n[Test Article 1]: '{sample_news_1[:60]}...'")
        print(f" -> PREDICTION : {label1} (Confidence: {conf1:.2f}%)")
        
        label2, conf2 = predict_news_article(sample_news_2, model, vec)
        print(f"\n[Test Article 2]: '{sample_news_2[:60]}...'")
        print(f" -> PREDICTION : {label2} (Confidence: {conf2:.2f}%)")
        
        print("\n" + "-"*60)
        print("Enter a custom news headline/text to test (or press Enter to exit):")
        user_input = input("News Text > ").strip()
        if user_input:
            label, conf = predict_news_article(user_input, model, vec)
            print(f"\nRESULT: The input article is predicted as [{label}] with {conf:.2f}% confidence.")
    except Exception as e:
        print(f"Error during prediction: {e}")
