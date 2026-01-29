from fastapi import FastAPI
from .routes.base import base_router
from .routes.predict import predict_router
import pickle

app = FastAPI(title="Churn Prediction API")

async def startup_span():

    model_path = "models/without_smoteenn/random_forest.pkl"
    label_encoders_path = "models/without_smoteenn/label_encoders.pkl"
    scaler_path = "models/without_smoteenn/scaler.pkl"

    with open(model_path, 'rb') as f:
        app.model = pickle.load(f)
    
    with open(label_encoders_path, 'rb') as f:
        app.label_encoder = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        app.scalar = pickle.load(f)

async def shutdown_span():

    if hasattr(app, 'model'):
        del app.model

    if hasattr(app, 'label_encoder'):
        del app.label_encoder

    if hasattr(app, 'scalar'):
        del app.scalar

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base_router)
app.include_router(predict_router)



