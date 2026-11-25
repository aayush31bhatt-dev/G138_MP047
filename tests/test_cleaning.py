"""Unit tests for data audit and cleaning without reading raw CSV files."""

from __future__ import annotations

import pandas as pd

from src.bike_price.cleaning import clean_dataframe, normalize_brand_tier
from src.bike_price.config import load_params
from src.bike_price.data_audit import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_unit_mixup,
)


def _sample_params() -> dict:
    return load_params("params.yaml")


def test_normalize_brand_tier():
    raw = pd.Series(["Budget", "bud.", " budget ", "MID", "mid.", "pre.", "Premium "])
    cleaned = normalize_brand_tier(raw)
    assert cleaned.tolist() == [
        "budget",
        "budget",
        "budget",
        "mid",
        "mid",
        "premium",
        "premium",
    ]


def test_clean_dataframe_fixes_issues():
    params = _sample_params()
    df = pd.DataFrame(
        [
            {
                "record_id": "R1",
                "date": "2026-01-05",
                "age_years": 5.0,
                "km_driven": 25.0,
                "engine_cc": 150,
                "owners": 1,
                "service_records": 6,
                "accidents": 0,
                "brand_tier": "bud.",
                "city_zone": "South",
                "final_price_from_rc_transfer": 90.0,
                "resale_thousand_rs": 91.0,
            },
            {
                "record_id": "R2",
                "date": "2026-01-06",
                "age_years": 4.0,
                "km_driven": 30000.0,
                "engine_cc": 200,
                "owners": 2,
                "service_records": -1,
                "accidents": 1,
                "brand_tier": "Pre.",
                "city_zone": "North",
                "final_price_from_rc_transfer": 110.0,
                "resale_thousand_rs": 111.0,
            },
            {
                "record_id": "R3",
                "date": "2026-02-15",
                "age_years": 6.0,
                "km_driven": 40000.0,
                "engine_cc": 180,
                "owners": 2,
                "service_records": 8,
                "accidents": 1,
                "brand_tier": "mid ",
                "city_zone": "Central",
                "final_price_from_rc_transfer": 80.0,
                "resale_thousand_rs": 82.0,
            },
        ]
    )

    cleaned, median_sr = clean_dataframe(df, params, is_training=True)

    assert "final_price_from_rc_transfer" not in cleaned.columns
    # Row R3 during stuck sensor window (2026-02-10..2026-02-23) is removed
    assert len(cleaned) == 2
    # South km_driven converted from thousand km to km
    assert cleaned.loc[0, "km_driven"] == 25000.0
    # North km_driven stays untouched
    assert cleaned.loc[1, "km_driven"] == 30000.0
    # Sentinel -1 replaced by median of remaining rows (6 and 8 -> 7.0)
    assert median_sr == 7.0
    assert cleaned.loc[1, "service_records"] == 7.0
    assert cleaned["brand_tier"].tolist() == ["budget", "premium"]


def test_audit_helpers_on_synthetic_data():
    train = pd.DataFrame(
        {
            "record_id": ["1", "2", "3", "4", "5", "6"],
            "date": ["2026-01-01"] * 6,
            "km_driven": [40.0, 50.0, 40000.0, 42000.0, 45000.0, 48000.0],
            "service_records": [-1, -1, -1, -1, -1, 5],
            "brand_tier": ["budget", "bud.", "mid", "Mid", "premium", "pre."],
            "city_zone": ["South", "South", "North", "North", "Central", "Central"],
            "leak": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [10.1, 20.2, 29.9, 40.1, 50.0, 59.8],
        }
    )
    later = pd.DataFrame(
        {
            "record_id": ["7", "8", "9"],
            "date": ["2026-07-01"] * 3,
            "km_driven": [41000.0, 42000.0, 43000.0],
            "service_records": [0, 5, 10],
            "brand_tier": ["budget", "mid", "premium"],
            "city_zone": ["South", "North", "Central"],
            "target": [10.0, 30.0, 50.0],
        }
    )
    holdout = later.drop(columns=["target"])

    assert check_leaking_column(train, holdout, "target") == {"present": True, "column": "leak"}
    assert check_unit_mixup(train, ["km_driven"]) == {
        "present": True,
        "column": "km_driven",
        "group_value": "South",
        "factor": 0.001,
    }
    assert check_sentinel_value(train, later, ["service_records"]) == {
        "present": True,
        "column": "service_records",
        "value": -1,
        "count": 5,
    }
    assert check_duplicate_rows(train) == {"present": False, "count": None}
    assert check_spelling_variants(train, later, ["brand_tier"]) == {
        "present": True,
        "column": "brand_tier",
        "true_levels": 3,
    }
