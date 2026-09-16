from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.model_service import predict_transaction


app = FastAPI(
    title="Fraud Detection & Risk Scoring API",
    description="API for real-time fraud detection and transaction risk scoring.",
    version="1.0.0"
)


class Transaction(BaseModel):

    Time: float = Field(..., ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(..., ge=0)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection & Risk Scoring API is running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(transaction: Transaction):

    result = predict_transaction(
        transaction.model_dump()
    )

    return result