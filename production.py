from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.production_service import predict_production, production_summary

router = APIRouter()


class ProductionInput(BaseModel):
    target_production_tonnes: float = Field(..., ge=0)
    equipment_downtime_hours: float = Field(..., ge=0)
    blasting_delay_hours: float = Field(..., ge=0)
    working_hours_per_day: float = Field(..., ge=0)
    rainfall_mm: float = Field(..., ge=0)
    land_surface_temperature_c: float = Field(..., ge=-100)
    ndvi: float = Field(...)
    prospectivity_score: float = Field(..., ge=0)


@router.post("/production/predict")
def predict(input_data: ProductionInput):
    payload = input_data.model_dump()
    return predict_production(payload)


@router.get("/production/summary")
def get_summary():
    return production_summary()
