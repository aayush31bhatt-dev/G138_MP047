"""Programmatic audit for the seven data problems in train.csv."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.bike_price.config import load_params


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """Find columns missing at prediction time that strongly correlate with target."""
    ignore = {"record_id", "date", target_col}
    extra_cols = [c for c in train_df.columns if c not in holdout_df.columns and c not in ignore]
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            corr = abs(train_df[col].corr(train_df[target_col]))
            if corr > 0.95:
                return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_unit_mixup(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    group_col: str = "city_zone",
) -> dict[str, Any]:
    """Check if one city_zone recorded a numeric column in a different unit."""
    standard_factors = [1000, 100, 60, 12, 10, 0.1, 0.01, 0.001]
    for col in numeric_cols:
        means = train_df.groupby(group_col)[col].mean()
        if len(means) < 2:
            continue
        overall_med = means.median()
        if overall_med == 0:
            continue
        for zone, zone_mean in means.items():
            ratio = zone_mean / overall_med
            if ratio > 5 or ratio < 0.2:
                factor = min(standard_factors, key=lambda f: abs(f - ratio))
                return {
                    "present": True,
                    "column": col,
                    "group_value": str(zone),
                    "factor": factor,
                }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, Any]:
    """Detect repeated impossible sentinel codes (e.g. -1, -999, 9999)."""
    candidates = [-9999, -999, -99, -1, 999, 9999]
    for col in numeric_cols:
        later_min = later_df[col].min()
        later_max = later_df[col].max()
        counts = train_df[col].value_counts()
        for val in candidates:
            cnt = int(counts.get(val, 0))
            if cnt >= 5 and (val < later_min or val > later_max):
                return {
                    "present": True,
                    "column": col,
                    "value": int(val),
                    "count": cnt,
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train_df: pd.DataFrame) -> dict[str, Any]:
    """Count exact duplicate rows."""
    dup_count = int(train_df.duplicated().sum())
    return {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None,
    }


def check_stuck_sensor(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    min_run_length: int = 20,
) -> dict[str, Any]:
    """Detect long consecutive runs of the exact same value when sorted by date."""
    sorted_df = train_df.sort_values(["date", "record_id"]).reset_index(drop=True)
    for col in numeric_cols:
        run_ids = (sorted_df[col] != sorted_df[col].shift()).cumsum()
        runs = sorted_df.groupby(run_ids).agg(
            count=(col, "size"),
            start_date=("date", "first"),
            end_date=("date", "last"),
        )
        longest = runs.sort_values("count", ascending=False).iloc[0]
        if int(longest["count"]) >= min_run_length:
            return {
                "present": True,
                "column": col,
                "start_date": str(longest["start_date"]),
                "end_date": str(longest["end_date"]),
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Detect target values typed with an extra zero (large 10x gap at the top)."""
    values = np.sort(train_df[target_col].dropna().to_numpy())
    if len(values) < 10:
        return {"present": False, "count": None}
    median_val = float(np.median(values))
    diffs = np.diff(values)
    max_gap_idx = int(np.argmax(diffs))
    high_group = values[max_gap_idx + 1 :]
    low_top = values[max_gap_idx]
    if len(high_group) <= 0.1 * len(values) and high_group[0] > 4 * low_top and high_group[0] > 5 * median_val:
        return {"present": True, "count": int(len(high_group))}
    return {"present": False, "count": None}


def check_spelling_variants(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    cat_cols: list[str],
) -> dict[str, Any]:
    """Find categorical columns whose raw spellings exceed the clean reference levels."""
    for col in cat_cols:
        train_unique = train_df[col].dropna().astype(str).nunique()
        clean_levels = later_df[col].dropna().astype(str).nunique()
        if train_unique > clean_levels:
            return {
                "present": True,
                "column": col,
                "true_levels": int(clean_levels),
            }
    return {"present": False, "column": None, "true_levels": None}


def run_audit(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all seven data-quality checks and return structured findings."""
    data_cfg = params["data"]
    num_cols = data_cfg["numeric_features"]
    cat_cols = data_cfg["categorical_features"]
    target_col = data_cfg["target_col"]

    return {
        "leaking_column": check_leaking_column(train_df, holdout_df, target_col),
        "unit_mixup": check_unit_mixup(train_df, num_cols),
        "sentinel_value": check_sentinel_value(train_df, later_df, num_cols),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df, num_cols),
        "target_typos": check_target_typos(train_df, target_col),
        "spelling_variants": check_spelling_variants(train_df, later_df, cat_cols),
    }


def main() -> None:
    params = load_params()
    train_df = pd.read_csv(params["paths"]["raw_train"])
    later_df = pd.read_csv(params["paths"]["raw_later"])
    holdout_df = pd.read_csv(params["paths"]["holdout_features"])

    issues = run_audit(train_df, later_df, holdout_df, params)
    for name, detail in issues.items():
        print(f"{name}: {detail}")


if __name__ == "__main__":
    main()
