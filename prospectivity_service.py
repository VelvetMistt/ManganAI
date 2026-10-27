import json
from pathlib import Path

import joblib
import pandas as pd

from services.data_service import load_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "prospectivity_model.pkl"


def load_model_payload():
    return joblib.load(MODEL_PATH)


def predict_prospectivity(features: dict):
    payload = load_model_payload()
    model = payload["model"]
    feature_names = payload["feature_names"]
    mapping = payload["inverse_label_mapping"]
    data = pd.DataFrame([features], columns=feature_names)
    probs = model.predict_proba(data)[0]
    class_index = int(model.predict(data)[0])
    label = mapping[class_index]
    return {
        "predicted_class": label,
        "probabilities": {key: round(float(value), 4) for key, value in zip(model.classes_, probs)},
        "confidence": round(float(max(probs)), 4),
        "model_metrics": payload["metrics"],
    }


def locations() -> list[dict]:
    df = load_dataset()
    df = df[[
        "sample_id", "state_region", "latitude", "longitude", "manganese_ppm",
        "magnetic_intensity_nT", "prospectivity_score", "prospectivity_label"
    ]].copy()
    return df.to_dict(orient="records")


def prospectivity_summary() -> dict:
    df = load_dataset()
    return {
        "High": int((df["prospectivity_label"] == "High").sum()),
        "Medium": int((df["prospectivity_label"] == "Medium").sum()),
        "Low": int((df["prospectivity_label"] == "Low").sum()),
    }
