from fastapi import APIRouter

from services.data_service import overview_summary, regional_summary

router = APIRouter()


@router.get("/overview")
def overview():
    return {
        "summary": overview_summary(),
        "regional_summary": regional_summary(),
    }
