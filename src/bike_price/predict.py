"""Generate final holdout predictions using the @champion model."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.bike_price.cleaning import normalize_brand_tier
from src.bike_price.config import load_params


def generate_holdout_predictions() -> Path:
    params = load_params()
    paths = params["paths"]
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    model_uri = f"models:/{mlflow_cfg['registered_model_name']}@champion"
    model = mlflow.sklearn.load_model(model_uri)

    holdout_df = pd.read_csv(paths["holdout_features"])
    feature_cols = data_cfg["numeric_features"] + data_cfg["categorical_features"]

    features = holdout_df[feature_cols].copy()
    features["brand_tier"] = normalize_brand_tier(
        features["brand_tier"], data_cfg.get("brand_aliases")
    )

    preds = np.clip(model.predict(features), 0.0, None)
    out_df = pd.DataFrame(
        {
            "record_id": holdout_df["record_id"],
            "prediction": np.round(preds, 2),
        }
    )

    out_path = Path(paths["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} predictions to {out_path}")
    return out_path


def main() -> None:
    generate_holdout_predictions()


if __name__ == "__main__":
    main()
