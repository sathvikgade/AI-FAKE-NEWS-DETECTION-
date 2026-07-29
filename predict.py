"""
================================================================================
MODULE 5: REAL-TIME PREDICTION SCRIPT
File: predict.py
================================================================================
Takes a news headline or full article text input and predicts whether it is REAL or FAKE
with exact probability scores.
"""

import os
import joblib
from preprocessing import clean_text

def load_prediction_artifacts(model_dir="saved_models"):
    """
    Loads saved model and vectorizer from disk.
    """
    model_path = os.path.join(model_dir, "best_model.pkl")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Trained model or vectorizer not found. Run main.py first!")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def predict_news_article(news_text, model=None, vectorizer=None):
    """
    Predicts whether news text is REAL or FAKE.
    Returns label ('REAL' / 'FAKE') and confidence percentage.
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
    print("="*65)
    print("       AI-POWERED FAKE NEWS DETECTOR - REAL-TIME INFERENCE       ")
    print("="*65)
    
    test_cases = [
        # REAL NEWS SAMPLES
        ("Russia revels in Trump victory, looks to sanctions relief", "REAL"),
        ("Trump's bid to open U.S. monuments to development draws calls for protection", "REAL"),
        ("House Speaker Ryan urges Trump son to testify in Congress", "REAL"),
        ("The Federal Reserve announced an interest rate decision today following economic policy meeting.", "REAL"),
        
        # FAKE NEWS SAMPLES
        ("PRESIDENT TRUMP Explains New America First RAISE Act Protects Jobs For Minorities", "FAKE"),
        ("UNREAL! HERE IS WHY ICE RELEASED BUT DID NOT DEPORT 19,723 Criminal Illegals In 2015", "FAKE"),
        ("MIRACLE CURE: Drinking hot lemon juice with baking soda eliminates all viruses instantly", "FAKE"),
        ("BREAKING: Scientists discover secret alien base under Antarctic ice caps hidden for centuries!", "FAKE")
    ]
    
    model, vectorizer = load_prediction_artifacts()
    print("\nEvaluating Verification Test Suite...")
    print("-" * 65)
    
    correct = 0
    for text, expected in test_cases:
        label, conf = predict_news_article(text, model, vectorizer)
        ok = (label == expected)
        if ok: correct += 1
        status = "[PASSED]" if ok else "[FAILED]"
        print(f"{status} Expected: {expected} | Predicted: {label} ({conf:.1f}% confidence)")
        print(f"         Article: '{text[:60]}...'\n")
        
    print("=" * 65)
    print(f"Final Verification Score: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 65)
    
    print("\nType any custom headline below to test (or press Enter to exit):")
    try:
        user_input = input("News Text > ").strip()
        if user_input:
            res_label, res_conf = predict_news_article(user_input, model, vectorizer)
            print(f"\n-> PREDICTION: [{res_label}] (Confidence: {res_conf:.2f}%)")
    except Exception as e:
        pass
