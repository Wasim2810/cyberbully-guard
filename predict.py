"""
predict.py
----------
Loads the trained model + TF-IDF vectorizer + label encoder and exposes a
single `predict_text()` function used by both the Streamlit app and any
command-line testing.

Run directly for a quick CLI test:
    python predict.py "You are useless and everyone hates you."
"""

import os
import sys
import joblib
import numpy as np

from utils.preprocessing import clean_text
from utils.helper import (
    detect_harmful_words,
    get_risk_level,
    get_reason,
    get_suggestion,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

_model = None
_tfidf = None
_label_encoder = None


def _load_artifacts():
    """Lazily load model artifacts (cached after first call)."""
    global _model, _tfidf, _label_encoder
    if _model is None or _tfidf is None or _label_encoder is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH)
                and os.path.exists(ENCODER_PATH)):
            raise FileNotFoundError(
                "Model artifacts not found. Please run 'python train_model.py' first."
            )
        _model = joblib.load(MODEL_PATH)
        _tfidf = joblib.load(TFIDF_PATH)
        _label_encoder = joblib.load(ENCODER_PATH)
    return _model, _tfidf, _label_encoder


def predict_text(raw_text: str) -> dict:
    """
    Run the full inference pipeline on a single piece of raw user text.

    Returns a dictionary with:
        prediction, confidence, risk_level, reason, suggestion,
        harmful_words (list of (word, category)), all_class_probabilities
    """
    model, tfidf, label_encoder = _load_artifacts()

    # 1. Preprocess text using the exact same pipeline used during training
    cleaned = clean_text(raw_text)

    # 2. Vectorize
    vector = tfidf.transform([cleaned])

    # 3. Predict class + probabilities
    pred_id = model.predict(vector)[0]
    prediction = label_encoder.inverse_transform([pred_id])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
    else:
        # Fallback for models without predict_proba: uniform pseudo-confidence
        probabilities = np.zeros(len(label_encoder.classes_))
        probabilities[pred_id] = 1.0

    confidence = float(probabilities[pred_id])
    class_probs = {
        label_encoder.classes_[i]: float(probabilities[i])
        for i in range(len(label_encoder.classes_))
    }

    # 4. Detect specific harmful words/phrases for explainability
    harmful_words = detect_harmful_words(raw_text)

    # 5. Risk scoring, reason, and recommended action
    risk_level = get_risk_level(prediction, confidence)
    reason = get_reason(prediction, harmful_words)
    suggestion = get_suggestion(prediction)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "reason": reason,
        "suggestion": suggestion,
        "harmful_words": harmful_words,
        "all_class_probabilities": class_probs,
        "cleaned_text": cleaned,
    }


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "You are useless and everyone hates you."
    result = predict_text(text)
    print(f"Input Text : {text}")
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']*100:.2f}%")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Reason     : {result['reason']}")
    print(f"Suggestion : {result['suggestion']}")
    print(f"Detected Words: {[w for w, c in result['harmful_words']]}")
