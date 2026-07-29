"""
================================================================================
MODULE 3: MODEL TRAINING
File: train.py
================================================================================
Trains Logistic Regression, Random Forest, Simple Neural Network (MLP), and KNN
on headline and content TF-IDF feature matrices. Saves optimal artifacts for deployment.
"""

import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_all_models(X_train, y_train):
    """
    Trains KNN, Logistic Regression, Random Forest, and Neural Network (MLP).
    """
    models = {
        "Logistic Regression (Parametric)": LogisticRegression(C=2.5, max_iter=1000, random_state=42),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Simple Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
        "KNN (Non-Parametric)": KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    print("\n[Model Training] Training all 4 Machine Learning Classifiers...")
    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"    -> {name} trained successfully!")
        
    return trained_models

def save_trained_artifacts(model, vectorizer, model_dir="saved_models"):
    """
    Saves best model and vectorizer to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    print(f"[Model Training] Best model & vectorizer saved to: {model_dir}")

if __name__ == "__main__":
    print("Run main.py to execute training pipeline.")
