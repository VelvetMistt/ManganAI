from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.risk_service import analyze_risk, risk_summary

router = APIRouter()


class RiskInput(BaseModel):
    equipment_downtime_hours: float = Field(..., ge=0)
    blasting_delay_hours: float = Field(..., ge=0)
    rainfall_mm: float = Field(..., ge=0)
    working_hours_per_day: float = Field(..., ge=0)
    prospectivity_score: float = Field(..., ge=0)
    shortfall_percentage: float = Field(..., ge=0)


@router.get("/risk/summary")
def get_summary():
    return risk_summary()


@router.post("/risk/analyze")
def analyze(input_data: RiskInput):
    return analyze_risk(input_data.model_dump())
