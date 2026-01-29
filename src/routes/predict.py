from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from ..routes import PredictRequest
from ..inference.inference_churn import inference
import logging

predict_router = APIRouter(
    prefix="/api/v1",
    tags=["Churn-Prediction"],
)
logger = logging.getLogger("uvicorn.error")

@predict_router.post("/predict")
async def predict_churn(request: Request, predict_request: PredictRequest):

    try:
        logger.info("Predict endpoint called.")

        # Access the model and encoders from the app state
        scalar = request.app.scalar
        label_encoder = request.app.label_encoder
        model = request.app.model

        churn_prediction, churn_probability = inference(model, scalar, label_encoder, predict_request)

        return JSONResponse(
            status_code=200,
            content={
                "churn_prediction": churn_prediction,
                "churn_probability": churn_probability
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})