"""
================================================================================
MODULE 4: MODEL EVALUATION
File: evaluate.py
================================================================================
Evaluates trained classifiers using Accuracy, Precision, Recall, F1-Score,
Classification Reports, and Confusion Matrices. Formats output for IEEE reports.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluates predictions of all models and outputs IEEE metrics comparison table.
    """
    print("\n" + "="*65)
    print("                MODEL EVALUATION & IEEE METRICS MATRIX           ")
    print("="*65)
    
    results = []
    predictions_dict = {}
    
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        predictions_dict[name] = preds
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        results.append({
            "Algorithm": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%"
        })
        
        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1-Score: {f1*100:.2f}%")
        print("Classification Report:\n", classification_report(y_test, preds, target_names=['Fake', 'Real']))
        
    summary_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("                  FINAL IEEE SUMMARY TABLE                      ")
    print("="*65)
    print(summary_df.to_string(index=False))
    print("="*65)
    
    return summary_df, predictions_dict

if __name__ == "__main__":
    print("Run main.py to evaluate models.")
