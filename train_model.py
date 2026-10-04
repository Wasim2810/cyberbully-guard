"""
train_model.py
---------------
Trains multiple machine-learning models on the cyberbullying dataset,
compares their performance, automatically selects the best model, and
saves the winning model + TF-IDF vectorizer to disk using Joblib.

Run:
    python train_model.py

Outputs:
    model/model.pkl            -> best trained classifier
    model/tfidf.pkl             -> fitted TF-IDF vectorizer
    model/label_encoder.pkl     -> label encoder (class name <-> id)
    model/metrics.json          -> accuracy / precision / recall / F1 for all models
    reports/confusion_matrix.png
    reports/roc_curve.png
    reports/model_comparison.csv
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from utils.preprocessing import clean_text, get_pipeline_status
from utils.visualization import plot_confusion_matrix, plot_roc_curves

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cyberbullying.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the labeled dataset from CSV. Expects columns: text, label."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    return df.reset_index(drop=True)


def main():
    print(f"[INFO] NLP backend in use: {get_pipeline_status()}")
    print(f"[INFO] Loading dataset from {DATASET_PATH} ...")
    df = load_dataset(DATASET_PATH)
    print(f"[INFO] Loaded {len(df)} rows across {df['label'].nunique()} classes.")
    print(df["label"].value_counts())

    # ---------------------------------------------------------------------
    # 1. Preprocess all text using the shared NLP pipeline
    # ---------------------------------------------------------------------
    print("[INFO] Cleaning text (lowercase, remove URLs/HTML/numbers/emojis/"
          "punctuation, stopword removal, tokenization, lemmatization)...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 2. Encode labels
    # ---------------------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    class_names = list(label_encoder.classes_)
    print(f"[INFO] Classes: {class_names}")

    # ---------------------------------------------------------------------
    # 3. Train / test split (stratified to preserve class balance)
    # ---------------------------------------------------------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------------------
    # 4. TF-IDF feature extraction
    # ---------------------------------------------------------------------
    print("[INFO] Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    # ---------------------------------------------------------------------
    # 5. Train multiple candidate models
    # ---------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5.0),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"[INFO] Training {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_sec": elapsed,
        }
        trained_models[name] = model

        print(f"    -> Accuracy={acc:.4f}  Precision={prec:.4f}  "
              f"Recall={rec:.4f}  F1={f1:.4f}  ({elapsed:.2f}s)")

    # ---------------------------------------------------------------------
    # 6. Select best model based on accuracy
    # ---------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_name]
    print(f"\n[RESULT] Best model: {best_name} "
          f"(Accuracy={results[best_name]['accuracy']:.4f})")

    print("\n[INFO] Classification report for best model:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

    # ---------------------------------------------------------------------
    # 7. Generate evaluation visuals (confusion matrix + ROC curve)
    # ---------------------------------------------------------------------
    print("[INFO] Generating confusion matrix and ROC curve...")
    plot_confusion_matrix(
        y_test, y_pred_best, labels=list(range(len(class_names))),
        save_path=os.path.join(REPORTS_DIR, "confusion_matrix.png"),
    )
    # Re-label axis ticks with class names for readability by re-plotting with names
    plot_confusion_matrix(
        [class_names[i] for i in y_test],
        [class_names[i] for i in y_pred_best],
        labels=class_names,
        save_path=os.path.join(REPORTS_DIR, "confusion_matrix.png"),
    )

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)
            plot_roc_curves(
                y_test_bin, y_score, class_names,
                save_path=os.path.join(REPORTS_DIR, "roc_curve.png"),
            )
    except Exception as exc:
        print(f"[WARN] Could not generate ROC curve: {exc}")

    # ---------------------------------------------------------------------
    # 8. Persist artifacts with Joblib
    # ---------------------------------------------------------------------
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    metrics_out = {
        "best_model": best_name,
        "classes": class_names,
        "results": results,
        "trained_on_rows": len(df),
        "test_size": 0.2,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=4)

    pd.DataFrame(results).T.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    print(f"\n[SUCCESS] Saved best model ('{best_name}') to model/model.pkl")
    print("[SUCCESS] Saved TF-IDF vectorizer to model/tfidf.pkl")
    print("[SUCCESS] Saved label encoder to model/label_encoder.pkl")
    print("[SUCCESS] Saved metrics.json and evaluation reports to reports/")


if __name__ == "__main__":
    main()
