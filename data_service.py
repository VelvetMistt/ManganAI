import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "manganese_synthetic_training_dataset_1000_rows (1).csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def dataset_summary() -> dict:
    df = load_dataset()
    return {
        "dataset_name": DATA_PATH.name,
        "records": int(len(df)),
        "features": int(len(df.columns)),
        "feature_categories": {
            "geological": ["elevation_m", "distance_to_fault_km"],
            "geochemical": ["manganese_ppm", "iron_pct", "silica_pct"],
            "geophysical": ["magnetic_intensity_nT"],
            "space_environmental": ["ndvi", "rainfall_mm", "land_surface_temperature_c", "latitude", "longitude", "elevation_m"],
            "operational": ["equipment_downtime_hours", "blasting_delay_hours", "working_hours_per_day"],
            "production": ["target_production_tonnes", "monthly_production_tonnes", "production_shortfall_pct"],
        },
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "synthetic_dataset_disclaimer": "Prototype uses a synthetic dataset for development and demonstration because proprietary MOIL operational data is not publicly available.",
        "feature_groups": {
            "Geological": ["elevation_m", "distance_to_fault_km"],
            "Geochemical": ["manganese_ppm", "iron_pct", "silica_pct"],
            "Geophysical": ["magnetic_intensity_nT"],
            "Space/Environmental": ["ndvi", "rainfall_mm", "land_surface_temperature_c", "elevation_m", "latitude", "longitude"],
            "Operational": ["equipment_downtime_hours", "blasting_delay_hours", "working_hours_per_day"],
            "Production": ["target_production_tonnes", "monthly_production_tonnes", "production_shortfall_pct"],
        },
    }


def prospectivity_distribution() -> dict:
    df = load_dataset()
    counts = df["prospectivity_label"].value_counts().to_dict()
    return {
        "High": int(counts.get("High", 0)),
        "Medium": int(counts.get("Medium", 0)),
        "Low": int(counts.get("Low", 0)),
        "total": int(len(df)),
    }


def overview_summary() -> dict:
    df = load_dataset()
    distr = prospectivity_distribution()
    return {
        "total_records": int(len(df)),
        "high_prospectivity_locations": int(distr["High"]),
        "medium_prospectivity_locations": int(distr["Medium"]),
        "low_prospectivity_locations": int(distr["Low"]),
        "average_prospectivity_score": round(float(df["prospectivity_score"].mean()), 2),
        "average_monthly_production": round(float(df["monthly_production_tonnes"].mean()), 2),
        "high_production_risk_count": int((df["production_risk"] == "High").sum()),
        "target_vs_production": {
            "total_target": round(float(df["target_production_tonnes"].sum()), 2),
            "total_actual": round(float(df["monthly_production_tonnes"].sum()), 2),
            "average_target": round(float(df["target_production_tonnes"].mean()), 2),
            "average_actual": round(float(df["monthly_production_tonnes"].mean()), 2),
        },
        "production_risk_distribution": {
            "Low": int((df["production_risk"] == "Low").sum()),
            "Medium": int((df["production_risk"] == "Medium").sum()),
            "High": int((df["production_risk"] == "High").sum()),
        },
    }


def regional_summary() -> list[dict]:
    df = load_dataset()
    summary = df.groupby("state_region").agg(
        records=("sample_id", "count"),
        avg_prospectivity_score=("prospectivity_score", "mean"),
        avg_monthly_production=("monthly_production_tonnes", "mean"),
        avg_target_production=("target_production_tonnes", "mean"),
        high_count=("prospectivity_label", lambda s: (s == "High").sum()),
        medium_count=("prospectivity_label", lambda s: (s == "Medium").sum()),
        low_count=("prospectivity_label", lambda s: (s == "Low").sum()),
    ).reset_index()
    summary = summary.sort_values("avg_prospectivity_score", ascending=False)
    return summary.to_dict(orient="records")
