"""FastAPI prediction service for MP047 (Indore used two-wheeler price)."""

from __future__ import annotations

import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.bike_price.cleaning import normalize_brand_tier
from src.bike_price.config import load_params


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "prediction", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mp047_api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = setup_logger()


class BikeFeatures(BaseModel):
    age_years: float = Field(..., ge=0, le=30, description="Age of bike in years (<= 30)")
    km_driven: float = Field(..., ge=0, description="Odometer reading in km")
    engine_cc: float = Field(..., ge=50, description="Engine displacement in cc (>= 50)")
    owners: int = Field(..., ge=0, description="Number of previous owners")
    service_records: int = Field(..., ge=0, description="Number of service visits")
    accidents: int = Field(..., ge=0, description="Reported accidents")
    brand_tier: str = Field(..., description="Brand tier (budget, mid, premium)")
    city_zone: str = Field(..., description="Dealer city zone (North, South, Central)")


class PredictionResponse(BaseModel):
    prediction: float


@lru_cache(maxsize=1)
def get_model():
    params = load_params()
    model_path = Path(params["paths"]["model_path"])
    return joblib.load(model_path)


app = FastAPI(title="MP047 Used Two-Wheeler Price API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BikeFeatures) -> PredictionResponse:
    start = time.perf_counter()
    model = get_model()

    row = pd.DataFrame([features.model_dump()])
    row["brand_tier"] = normalize_brand_tier(row["brand_tier"])
    row["city_zone"] = row["city_zone"].astype(str).str.strip()

    raw_pred = float(model.predict(row)[0])
    prediction = round(max(0.0, raw_pred), 2)
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

    logger.info(
        "prediction_served",
        extra={
            "event": "predict",
            "prediction": prediction,
            "latency_ms": latency_ms,
        },
    )
    return PredictionResponse(prediction=prediction)
