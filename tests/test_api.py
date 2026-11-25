"""FastAPI service and JSON logging tests using TestClient."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.main import JsonFormatter, app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_request():
    payload = {
        "age_years": 5.0,
        "km_driven": 25000.0,
        "engine_cc": 150.0,
        "owners": 1,
        "service_records": 8,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
    assert body["prediction"] >= 0.0


def test_predict_rejects_age_above_30():
    payload = {
        "age_years": 31.0,
        "km_driven": 25000.0,
        "engine_cc": 150.0,
        "owners": 1,
        "service_records": 8,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_engine_cc_below_50():
    payload = {
        "age_years": 5.0,
        "km_driven": 25000.0,
        "engine_cc": 45.0,
        "owners": 1,
        "service_records": 8,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_json_formatter_outputs_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="mp047_api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="prediction_served",
        args=(),
        exc_info=None,
    )
    record.event = "predict"
    record.prediction = 95.4
    record.latency_ms = 1.25

    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["event"] == "predict"
    assert parsed["prediction"] == 95.4
    assert parsed["latency_ms"] == 1.25
