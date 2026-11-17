"""Shared inference: loads the saved scaler + ANN once, used by app.py and api.py."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import keras
import pandas as pd

MODELS = Path(__file__).parent / "models"
CLASSES = ["Low", "Medium", "High"]

_scaler = joblib.load(MODELS / "scaler.joblib")
_model = keras.models.load_model(MODELS / "student_performance.keras")
FEATURES = list(_scaler.feature_names_in_)  # column order the model was trained on


def predict(student: dict) -> dict:
    x = _scaler.transform(pd.DataFrame([student])[FEATURES])
    proba = _model.predict(x, verbose=0)[0]
    i = int(proba.argmax())
    return {
        "predicted_performance": CLASSES[i],
        "confidence": round(float(proba[i]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
    }
