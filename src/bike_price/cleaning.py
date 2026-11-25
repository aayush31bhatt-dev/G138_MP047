"""Data cleaning transformations for training and inference inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bike_price.config import load_params


def normalize_brand_tier(
    series: pd.Series,
    aliases: dict[str, str] | None = None,
) -> pd.Series:
    """Strip spaces, lowercase, and map shorthand variants to canonical tiers."""
    mapping = aliases or {"bud.": "budget", "pre.": "premium", "mid.": "mid"}
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.replace(mapping)


def clean_dataframe(
    df: pd.DataFrame,
    params: dict[str, Any],
    service_median: float | None = None,
    is_training: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Clean raw bike records and return (cleaned_df, service_records_median)."""
    data_cfg = params["data"]
    cleaned = df.drop_duplicates().copy()

    leaking_col = data_cfg["leaking_col"]
    if leaking_col in cleaned.columns:
        cleaned = cleaned.drop(columns=[leaking_col])

    # Fix unit mixup where South recorded km_driven in thousand km (factor 0.001)
    unit_cfg = data_cfg["unit_fix"]
    unit_col = unit_cfg["column"]
    unit_zone = unit_cfg["zone"]
    unit_factor = float(unit_cfg["factor"])
    affected = (cleaned["city_zone"] == unit_zone) & (cleaned[unit_col] < 500)
    cleaned.loc[affected, unit_col] = cleaned.loc[affected, unit_col] / unit_factor

    # Normalize spelling variants in brand_tier
    cleaned["brand_tier"] = normalize_brand_tier(
        cleaned["brand_tier"], data_cfg.get("brand_aliases")
    )

    # Replace sentinel value (-1) in service_records with training median
    sent_cfg = data_cfg["sentinel"]
    sent_col = sent_cfg["column"]
    sent_val = sent_cfg["value"]
    cleaned.loc[cleaned[sent_col] == sent_val, sent_col] = np.nan

    if service_median is None:
        service_median = float(cleaned[sent_col].median())
    cleaned[sent_col] = cleaned[sent_col].fillna(service_median)

    # Drop training rows captured during the stuck-sensor period for owners
    if is_training and "date" in cleaned.columns:
        stuck_cfg = data_cfg["stuck_sensor"]
        in_stuck_window = (cleaned["date"] >= stuck_cfg["start_date"]) & (
            cleaned["date"] <= stuck_cfg["end_date"]
        )
        cleaned = cleaned.loc[~in_stuck_window].copy()

    return cleaned.reset_index(drop=True), service_median


def main() -> None:
    params = load_params()
    raw_df = pd.read_csv(params["paths"]["raw_train"])
    cleaned_df, median_val = clean_dataframe(raw_df, params, is_training=True)

    out_path = Path(params["paths"]["cleaned_train"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned train rows: {len(raw_df)} -> {len(cleaned_df)} (service_records median={median_val})")


if __name__ == "__main__":
    main()
