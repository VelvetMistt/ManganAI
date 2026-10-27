from pathlib import Path

import joblib
import pandas as pd

from services.data_service import load_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "production_model.pkl"


def load_model_payload():
    return joblib.load(MODEL_PATH)


def predict_production(features: dict):
    payload = load_model_payload()
    model = payload["model"]
    feature_names = payload["feature_names"]
    data = pd.DataFrame([features], columns=feature_names)
    prediction = float(model.predict(data)[0])
    target = float(features.get("target_production_tonnes", 0.0))
    shortfall = max(target - prediction, 0.0)
    shortfall_pct = (shortfall / target * 100.0) if target > 0 else 0.0

    if shortfall_pct < 5:
        risk = "Low"
    elif shortfall_pct < 12:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "predicted_monthly_production": round(prediction, 2),
        "shortfall_tonnes": round(shortfall, 2),
        "shortfall_percentage": round(shortfall_pct, 2),
        "production_risk": risk,
        "model_metrics": payload["metrics"],
    }


def production_summary() -> dict:
    df = load_dataset()
    return {
        "average_target_production": round(float(df["target_production_tonnes"].mean()), 2),
        "average_predicted_production": round(float(df["monthly_production_tonnes"].mean()), 2),
        "average_shortfall_pct": round(float(df["production_shortfall_pct"].mean()), 2),
        "risk_distribution": {
            "Low": int((df["production_risk"] == "Low").sum()),
            "Medium": int((df["production_risk"] == "Medium").sum()),
            "High": int((df["production_risk"] == "High").sum()),
        },
    }
