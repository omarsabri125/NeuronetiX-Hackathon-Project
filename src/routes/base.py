from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import logging
import yaml

base_router = APIRouter(
    prefix="/api/v1",
    tags=["Churn-Prediction"],
)
logger = logging.getLogger("uvicorn.error")

def get_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

@base_router.get("/health")
async def health_check(params=Depends(get_params)):
    try:
        logger.info("Health check endpoint called.")
        return {"status": "ok",
                "app_name": params["prepare"]["app_name"]}
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

