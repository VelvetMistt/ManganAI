from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.data_service import prospectivity_distribution
from services.prospectivity_service import locations, predict_prospectivity, prospectivity_summary

router = APIRouter()


class ProspectivityInput(BaseModel):
    manganese_ppm: float = Field(..., ge=0)
    iron_pct: float = Field(..., ge=0)
    silica_pct: float = Field(..., ge=0)
    magnetic_intensity_nT: float = Field(..., ge=0)
    elevation_m: float = Field(..., ge=0)
    ndvi: float = Field(...)
    rainfall_mm: float = Field(..., ge=0)
    land_surface_temperature_c: float = Field(..., ge=-100)
    distance_to_fault_km: float = Field(..., ge=0)
    latitude: float = Field(...)
    longitude: float = Field(...)


@router.get("/prospectivity/locations")
def get_locations():
    return {"locations": locations()}


@router.get("/prospectivity/distribution")
def get_distribution():
    return prospectivity_distribution()


@router.get("/prospectivity/summary")
def get_summary():
    return prospectivity_summary()


@router.post("/prospectivity/predict")
def predict(input_data: ProspectivityInput):
    payload = input_data.model_dump()
    result = predict_prospectivity(payload)
    return result
