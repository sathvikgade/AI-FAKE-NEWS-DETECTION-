# ==============================================================================
# PROJECT 1: SUMMER INTERSHIP PROGRAM IN AI & ML (2026)
# AI-POWERED FAKE NEWS DETECTION USING TEXT CLASSIFICATION
# ==============================================================================
# Exact implementation following the Python Code Skeleton from the Internship PDF
# ==============================================================================

# Week 1: Data Loading & Cleaning
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. Load Dataset
data = pd.read_csv("train.csv", low_memory=False)

# Clean null values and filter valid labels
data = data.dropna(subset=['text', 'label']).copy()
data = data[data['label'].astype(str).str.strip().str.upper().isin(['FAKE', 'REAL'])].copy()

X = data['text']
y = data['label']

# 2. Text Preprocessing Function (Remove Punctuation & Lowercase)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text) # remove punctuation & non-word chars
    text = text.lower()             # lowercase
    return text

print("Cleaning text dataset... Please wait.")
X = X.apply(clean_text)

# Week 2: Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer

print("Extracting TF-IDF features (max_features=5000)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
print(f"Dataset split -> Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Week 3: Model Building
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
}

# Train & Evaluate
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "="*60)
print("             TRAINING & EVALUATING ALL 4 ALGORITHMS             ")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc * 100:.2f}%")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, preds))

print("\n" + "="*60)
print("Pipeline Execution Completed Successfully!")
print("="*60)
