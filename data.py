from fastapi import APIRouter

from services.data_service import dataset_summary

router = APIRouter()


@router.get("/data/sources")
def data_sources():
    return dataset_summary()
