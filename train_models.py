from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score, recall_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "manganese_synthetic_training_dataset_1000_rows (1).csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PROSPECTIVITY_FEATURES = [
    "manganese_ppm",
    "iron_pct",
    "silica_pct",
    "magnetic_intensity_nT",
    "distance_to_fault_km",
    "elevation_m",
    "ndvi",
    "rainfall_mm",
    "land_surface_temperature_c",
    "latitude",
    "longitude",
]

PRODUCTION_FEATURES = [
    "target_production_tonnes",
    "equipment_downtime_hours",
    "blasting_delay_hours",
    "working_hours_per_day",
    "rainfall_mm",
    "land_surface_temperature_c",
    "ndvi",
    "prospectivity_score",
]


def safe_isna(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum().to_frame(name="missing_values")


def train_prospectivity_model(df: pd.DataFrame):
    X = df[PROSPECTIVITY_FEATURES].copy()
    y = df["prospectivity_label"].copy()
    le = {label: idx for idx, label in enumerate(sorted(y.unique().tolist()))}
    y_encoded = y.map(le)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    pred_labels = pd.Series(preds).map({v: k for k, v in le.items()})
    actual_labels = pd.Series(y_test).map({v: k for k, v in le.items()})

    metrics = {
        "algorithm": "Random Forest Classifier",
        "accuracy": round(float(accuracy_score(actual_labels, pred_labels)), 4),
        "precision": round(float(precision_score(actual_labels, pred_labels, average='weighted', zero_division=0)), 4),
        "recall": round(float(recall_score(actual_labels, pred_labels, average='weighted', zero_division=0)), 4),
        "f1_score": round(float(f1_score(actual_labels, pred_labels, average='weighted', zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(actual_labels, pred_labels, labels=["High", "Medium", "Low"]).tolist(),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "feature_count": int(len(PROSPECTIVITY_FEATURES)),
        "trained_at": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        "label_mapping": le,
        "classes": ["High", "Medium", "Low"],
    }

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    metrics["feature_importance"] = importance.round(4).to_dict()

    payload = {
        "model": model,
        "feature_names": PROSPECTIVITY_FEATURES,
        "label_mapping": le,
        "inverse_label_mapping": {v: k for k, v in le.items()},
        "metrics": metrics,
    }
    joblib.dump(payload, MODEL_DIR / "prospectivity_model.pkl")
    return payload


def train_production_model(df: pd.DataFrame):
    X = df[PRODUCTION_FEATURES].copy()
    y = df["monthly_production_tonnes"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "algorithm": "Random Forest Regressor",
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "feature_count": int(len(PRODUCTION_FEATURES)),
        "trained_at": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    metrics["feature_importance"] = importance.round(4).to_dict()

    payload = {
        "model": model,
        "feature_names": PRODUCTION_FEATURES,
        "metrics": metrics,
    }
    joblib.dump(payload, MODEL_DIR / "production_model.pkl")
    return payload


def save_preprocessor_metadata():
    metadata = {
        "prospectivity_features": PROSPECTIVITY_FEATURES,
        "production_features": PRODUCTION_FEATURES,
        "label_order": ["High", "Medium", "Low"],
        "synthetic_dataset_notice": "Prototype uses a synthetic dataset for development and demonstration because proprietary MOIL operational data is not publicly available.",
    }
    joblib.dump(metadata, MODEL_DIR / "preprocessors.pkl")


def main():
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isna().sum())
    print("Prospectivity labels:\n", df["prospectivity_label"].value_counts())
    print("Production risk:\n", df["production_risk"].value_counts())

    train_prospectivity_model(df)
    train_production_model(df)
    save_preprocessor_metadata()

    print("Prospectivity model saved to:", MODEL_DIR / "prospectivity_model.pkl")
    print("Production model saved to:", MODEL_DIR / "production_model.pkl")
    print("Preprocessor metadata saved to:", MODEL_DIR / "preprocessors.pkl")


if __name__ == "__main__":
    main()
