"""Drift monitoring (PSI), group MAE evaluation, and v2 retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.bike_price.cleaning import clean_dataframe
from src.bike_price.config import file_sha256, load_params
from src.bike_price.train import build_pipeline


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index using quantile bins on expected values."""
    eps = 1e-6
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def should_promote(mae_v1: float, mae_v2: float, min_improvement: float = 0.08) -> bool:
    """Return True if v2 MAE improves over v1 by at least min_improvement fraction."""
    if mae_v1 <= 0:
        return False
    return mae_v2 <= mae_v1 * (1.0 - min_improvement)


def run_monitoring_and_retrain(params: dict[str, Any]) -> dict[str, Any]:
    """Compute PSI, group MAE, retrain v2 on train + 70% later, and apply promotion rule."""
    paths = params["paths"]
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    train_cfg = params["training"]
    mon_cfg = params["monitoring"]

    raw_train_path = Path(paths["raw_train"])
    data_sha256 = file_sha256(raw_train_path)

    raw_train = pd.read_csv(raw_train_path)
    clean_train, _ = clean_dataframe(raw_train, params, is_training=True)
    later_df = pd.read_csv(paths["raw_later"])

    num_features = data_cfg["numeric_features"]
    cat_features = data_cfg["categorical_features"]
    feature_cols = num_features + cat_features
    target_col = data_cfg["target_col"]

    # 1. PSI per numeric column
    psi_bins = int(mon_cfg["psi_bins"])
    psi_scores = {
        col: round(
            compute_psi(
                clean_train[col].to_numpy(dtype=float),
                later_df[col].to_numpy(dtype=float),
                bins=psi_bins,
            ),
            4,
        )
        for col in num_features
    }
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 2. Load v1 from MLflow registry and evaluate per city_zone on later.csv
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    reg_name = mlflow_cfg["registered_model_name"]

    v1_model = mlflow.sklearn.load_model(f"models:/{reg_name}/1")

    mae_by_group: dict[str, float] = {}
    for zone in ["North", "South", "Central"]:
        sub = later_df.loc[later_df["city_zone"] == zone]
        preds = np.clip(v1_model.predict(sub[feature_cols]), 0.0, None)
        mae_by_group[zone] = round(float(mean_absolute_error(sub[target_col], preds)), 4)

    worst_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Sort later.csv by date, split 70/30, and retrain v2
    later_sorted = later_df.sort_values(["date", "record_id"]).reset_index(drop=True)
    split_idx = int(len(later_sorted) * float(mon_cfg["retrain_train_fraction"]))
    later_train = later_sorted.iloc[:split_idx].copy()
    later_check = later_sorted.iloc[split_idx:].copy()

    combined_train = pd.concat([clean_train, later_train], ignore_index=True)

    v2_estimator = GradientBoostingRegressor(
        n_estimators=int(mon_cfg["v2_n_estimators"]),
        learning_rate=float(mon_cfg["v2_learning_rate"]),
        max_depth=int(mon_cfg["v2_max_depth"]),
        random_state=int(train_cfg["random_state"]),
    )
    v2_model = build_pipeline(v2_estimator, num_features, cat_features)
    v2_model.fit(combined_train[feature_cols], combined_train[target_col])

    v1_check_preds = np.clip(v1_model.predict(later_check[feature_cols]), 0.0, None)
    v2_check_preds = np.clip(v2_model.predict(later_check[feature_cols]), 0.0, None)

    mae_v1_check = round(float(mean_absolute_error(later_check[target_col], v1_check_preds)), 4)
    mae_v2_check = round(float(mean_absolute_error(later_check[target_col], v2_check_preds)), 4)

    # 4. Log v2 to MLflow, register version 2, and apply 8% promotion rule
    promote = should_promote(
        mae_v1_check,
        mae_v2_check,
        min_improvement=float(mon_cfg["promotion_min_improvement"]),
    )
    decision = "promote" if promote else "keep"

    with mlflow.start_run(run_name="v2_retrain_gbr") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("stage", "v2")
        mlflow.log_param("model_family", "GradientBoostingRegressor")
        mlflow.log_param("retrain_rows", len(combined_train))
        mlflow.log_param("decision", decision)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("mae_later_check", mae_v2_check)
        mlflow.sklearn.log_model(
            v2_model,
            name="model",
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    mv2 = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=reg_name)
    client = MlflowClient()

    if promote:
        client.set_registered_model_alias(reg_name, "champion", mv2.version)
        joblib.dump(v2_model, paths["model_path"])
        champion_version = int(mv2.version)
    else:
        champion_version = 1

    return {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
        "champion_version": champion_version,
        "v2_run_id": v2_run_id,
    }


def main() -> None:
    params = load_params()
    summary = run_monitoring_and_retrain(params)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
