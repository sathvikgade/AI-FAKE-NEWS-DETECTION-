"""
================================================================================
MODULE 3: MODEL TRAINING
File: train.py
================================================================================
Trains 4 Machine Learning Classifiers:
1. KNN (K-Nearest Neighbors - Non-Parametric)
2. Logistic Regression (Parametric Linear Classifier)
3. Random Forest (Ensemble Classifier)
4. Simple Neural Network (MLPClassifier / Deep Learning)

Saves trained models & vectorizer to disk for real-time predictions.
"""

import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_all_models(X_train, y_train):
    """
    Trains KNN, Logistic Regression, Random Forest, and Simple Neural Network (MLP).
    """
    models = {
        "KNN (Non-Parametric)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression (Parametric)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Simple Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    }
    
    trained_models = {}
    print("\n[Model Training] Starting multi-model training...")
    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"    -> {name} trained successfully!")
        
    return trained_models

def save_trained_artifacts(model, vectorizer, model_dir="saved_models"):
    """
    Saves the best model and TF-IDF vectorizer to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[Model Training] Best model saved to: {model_path}")
    print(f"[Model Training] Vectorizer saved to: {vec_path}")

if __name__ == "__main__":
    print("Run main.py to execute training pipeline.")
