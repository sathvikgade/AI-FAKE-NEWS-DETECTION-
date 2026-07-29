"""
================================================================================
MAIN PIPELINE ORCHESTRATOR
File: main.py
================================================================================
Orchestrates the entire 4-week Machine Learning workflow:
1. Data Preprocessing (preprocessing.py)
2. Feature Engineering & TF-IDF Vectorization (feature_engineering.py)
3. Multi-Model Training (train.py)
4. Evaluation & IEEE Metrics Table (evaluate.py)
5. Model Saving & Real-Time Inference Testing (predict.py)
"""

import os
from preprocessing import load_and_preprocess_data
from feature_engineering import extract_tfidf_features, split_dataset
from train import train_all_models, save_trained_artifacts
from evaluate import evaluate_all_models
from predict import predict_news_article

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "train.csv")
    
    print("="*65)
    print("      PROJECT 1: AI-POWERED FAKE NEWS DETECTION PIPELINE        ")
    print("="*65)
    
    # Step 1: Preprocessing
    df = load_and_preprocess_data(data_path, sample_size=5000)
    
    # Step 2: Feature Engineering
    X, y, vectorizer = extract_tfidf_features(df, max_features=5000)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20)
    
    # Step 3: Model Training
    trained_models = train_all_models(X_train, y_train)
    
    # Step 4: Evaluation
    summary_df, predictions_dict = evaluate_all_models(trained_models, X_test, y_test)
    
    # Step 5: Save Best Model (Random Forest)
    best_model = trained_models["Random Forest (Ensemble)"]
    save_trained_artifacts(best_model, vectorizer, model_dir="saved_models")
    
    # Step 6: Test Real-Time Inference (predict.py)
    print("\n" + "="*65)
    print("                   TESTING REAL-TIME PREDICTIONS                 ")
    print("="*65)
    sample_text = "The Federal Reserve announced an interest rate decision today following the monthly economic policy meeting."
    label, conf = predict_news_article(sample_text, best_model, vectorizer)
    print(f"Sample Input: '{sample_text}'")
    print(f" -> PREDICTION: [{label}] (Confidence: {conf:.2f}%)")
    
    print("\n[SUCCESS] Pipeline executed cleanly! All modules integrated.")

if __name__ == "__main__":
    main()
