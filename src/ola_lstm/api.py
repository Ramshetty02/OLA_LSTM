"""FastAPI inference server for ride demand forecasting."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ola_lstm.config import Config
from ola_lstm.predict import RideDemandPredictor, PredictionResult

app = FastAPI(
    title="Ola Ride Demand Forecast API",
    description="Stacked LSTM inference for hourly bike ride demand prediction",
    version="1.0.0",
)

_predictor: RideDemandPredictor | None = None


class PredictRequest(BaseModel):
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(ge=0, le=6, description="0=Monday … 6=Sunday")
    month: int = Field(ge=1, le=12, description="Month (1-12)")


class PredictResponse(BaseModel):
    ride_count: int
    demand_level: str
    hour: int
    day_name: str
    month_name: str
    next_6_hours: list[dict[str, int]]


def _to_response(result: PredictionResult) -> PredictResponse:
    return PredictResponse(
        ride_count=result.ride_count,
        demand_level=result.demand_level,
        hour=result.hour,
        day_name=result.day_name,
        month_name=result.month_name,
        next_6_hours=[{"hour": h, "rides": c} for h, c in result.next_6_hours],
    )


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    artifacts = Path("artifacts")
    if not (artifacts / Config().model_filename).exists():
        raise RuntimeError(
            "No trained model found. Run `ola-lstm train` first."
        )
    _predictor = RideDemandPredictor.from_artifacts(artifacts)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _predictor.predict(request.hour, request.day_of_week, request.month)
    return _to_response(result)
