from pydantic import BaseModel


class PredictRequest(BaseModel):
    complaint: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float | None = None
