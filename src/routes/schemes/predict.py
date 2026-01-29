from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    SeniorCitizen: Literal[0, 1] = Field(..., description="0 or 1")

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(..., ge=0, description="Number of months like 5 or 29")

    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
