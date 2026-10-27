from fastapi import APIRouter
import joblib
from pathlib import Path

router = APIRouter()

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@router.get("/models/metrics")
def model_metrics():
    prospectivity = joblib.load(MODEL_DIR / "prospectivity_model.pkl")
    production = joblib.load(MODEL_DIR / "production_model.pkl")
    return {
        "prospectivity": prospectivity["metrics"],
        "production": production["metrics"],
    }
