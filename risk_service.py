from services.data_service import load_dataset


def risk_thresholds():
    return {
        "Low": {"min": 0, "max": 5, "description": "shortfall < 5%"},
        "Medium": {"min": 5, "max": 12, "description": "5% <= shortfall < 12%"},
        "High": {"min": 12, "max": 100, "description": "shortfall >= 12%"},
    }


def analyze_risk(payload: dict):
    shortfall_pct = float(payload.get("shortfall_percentage", 0.0))
    if shortfall_pct < 5:
        risk = "Low"
    elif shortfall_pct < 12:
        risk = "Medium"
    else:
        risk = "High"

    drivers = {
        "equipment_downtime_hours": float(payload.get("equipment_downtime_hours", 0.0)),
        "blasting_delay_hours": float(payload.get("blasting_delay_hours", 0.0)),
        "rainfall_mm": float(payload.get("rainfall_mm", 0.0)),
        "working_hours_per_day": float(payload.get("working_hours_per_day", 0.0)),
        "prospectivity_score": float(payload.get("prospectivity_score", 0.0)),
    }

    recommendations = []
    if drivers["equipment_downtime_hours"] > 7:
        recommendations.append("Review equipment availability and maintenance scheduling.")
    if drivers["blasting_delay_hours"] > 5:
        recommendations.append("Review blasting schedule and operational dependencies.")
    if drivers["rainfall_mm"] > 1000:
        recommendations.append("Consider weather-aware production planning.")
    if drivers["working_hours_per_day"] < 18:
        recommendations.append("Review shift utilization and scheduling.")
    if drivers["prospectivity_score"] > 75:
        recommendations.append("Prospectivity remains favorable; ensure resource allocation aligns with mineral potential.")

    if not recommendations:
        recommendations.append("Maintain current operating plan and monitor daily production variance.")

    return {
        "risk_level": risk,
        "risk_score": round(min(100, max(0, shortfall_pct * 5)), 2),
        "shortfall_percentage": round(shortfall_pct, 2),
        "risk_drivers": drivers,
        "recommended_actions": recommendations,
        "thresholds": risk_thresholds(),
        "decision_support_note": "Prototype decision-support thresholds only; not a guaranteed optimization result.",
    }


def risk_summary() -> dict:
    df = load_dataset()
    risk_levels = {
        "Low": int((df["production_risk"] == "Low").sum()),
        "Medium": int((df["production_risk"] == "Medium").sum()),
        "High": int((df["production_risk"] == "High").sum()),
    }
    return {
        "distribution": risk_levels,
        "thresholds": risk_thresholds(),
    }
