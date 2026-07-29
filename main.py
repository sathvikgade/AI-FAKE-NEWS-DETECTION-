"""
================================================================================
MAIN PIPELINE ORCHESTRATOR
File: main.py
================================================================================
Orchestrates the entire 4-week Machine Learning workflow:
1. Data Preprocessing (preprocessing.py)
2. Feature Engineering (feature_engineering.py)
3. Multi-Model Training (train.py)
4. Evaluation & IEEE Metrics Matrix (evaluate.py)
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
    
    # 1. Preprocessing
    df = load_and_preprocess_data(data_path, sample_size=10000)
    
    # 2. Feature Engineering
    X_title, X_text, y, title_vec, text_vec = extract_tfidf_features(df, max_features=10000)
    X_train, X_test, y_train, y_test = split_dataset(X_title, y, test_size=0.20)
    
    # 3. Model Training
    trained_models = train_all_models(X_train, y_train)
    
    # 4. Evaluation & IEEE Metrics Table
    summary_df, predictions_dict = evaluate_all_models(trained_models, X_test, y_test)
    
    # 5. Save Best Model (Logistic Regression / Random Forest)
    best_model = trained_models["Logistic Regression (Parametric)"]
    save_trained_artifacts(best_model, title_vec, model_dir="saved_models")
    
    # 6. Real-Time Prediction Verification
    print("\n" + "="*65)
    print("                   TESTING REAL-TIME PREDICTIONS                 ")
    print("="*65)
    sample_text = "The Federal Reserve announced an interest rate decision today following the monthly economic policy meeting."
    label, conf = predict_news_article(sample_text, best_model, title_vec)
    print(f"Sample Input: '{sample_text}'")
    print(f" -> PREDICTION: [{label}] (Confidence: {conf:.2f}%)")
    
    print("\n[SUCCESS] Pipeline executed cleanly! All modules integrated.")

if __name__ == "__main__":
    main()
