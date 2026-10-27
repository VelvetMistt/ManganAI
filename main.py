from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.data import router as data_router
from routes.models import router as models_router
from routes.overview import router as overview_router
from routes.prospectivity import router as prospectivity_router
from routes.production import router as production_router
from routes.risk import router as risk_router

app = FastAPI(
    title="MOIL Manganese Prospectivity & Production Intelligence",
    version="1.0.0",
    description="AI-based manganese prospectivity and production shortfall decision support dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(overview_router, prefix="/api")
app.include_router(prospectivity_router, prefix="/api")
app.include_router(production_router, prefix="/api")
app.include_router(risk_router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "MOIL manganese intelligence API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
