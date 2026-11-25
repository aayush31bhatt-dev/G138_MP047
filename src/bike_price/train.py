"""Model training and MLflow experiment tracking for MP047."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bike_price.cleaning import clean_dataframe
from src.bike_price.config import file_sha256, load_params


def build_pipeline(
    estimator: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create a preprocessing + regression scikit-learn pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def get_candidate_models(training_cfg: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Return candidate (run_name, estimator, hyperparams) tuples."""
    seed = int(training_cfg["random_state"])
    candidates: list[tuple[str, Any, dict[str, Any]]] = []

    for alpha in training_cfg["ridge_alphas"]:
        candidates.append(
            (
                f"ridge_alpha_{alpha}",
                Ridge(alpha=float(alpha)),
                {"model_family": "Ridge", "alpha": float(alpha)},
            )
        )

    candidates.append(
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=int(training_cfg["rf_n_estimators"]),
                max_depth=int(training_cfg["rf_max_depth"]),
                random_state=seed,
            ),
            {
                "model_family": "RandomForestRegressor",
                "n_estimators": int(training_cfg["rf_n_estimators"]),
                "max_depth": int(training_cfg["rf_max_depth"]),
                "random_state": seed,
            },
        )
    )

    candidates.append(
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=int(training_cfg["gbr_n_estimators"]),
                learning_rate=float(training_cfg["gbr_learning_rate"]),
                max_depth=int(training_cfg["gbr_max_depth"]),
                random_state=seed,
            ),
            {
                "model_family": "GradientBoostingRegressor",
                "n_estimators": int(training_cfg["gbr_n_estimators"]),
                "learning_rate": float(training_cfg["gbr_learning_rate"]),
                "max_depth": int(training_cfg["gbr_max_depth"]),
                "random_state": seed,
            },
        )
    )
    return candidates


def train_v1(params: dict[str, Any]) -> dict[str, Any]:
    """Train candidate v1 models on cleaned train.csv and register the best one."""
    paths = params["paths"]
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    train_cfg = params["training"]

    raw_train_path = Path(paths["raw_train"])
    data_sha256 = file_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    clean_df, service_median = clean_dataframe(raw_df, params, is_training=True)

    clean_out = Path(paths["cleaned_train"])
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_out, index=False)

    num_features = data_cfg["numeric_features"]
    cat_features = data_cfg["categorical_features"]
    feature_cols = num_features + cat_features
    target_col = data_cfg["target_col"]

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    cv = KFold(
        n_splits=int(train_cfg["cv_folds"]),
        shuffle=True,
        random_state=int(train_cfg["random_state"]),
    )

    best_run_id = ""
    best_cv_mae = float("inf")
    best_pipeline: Pipeline | None = None
    best_name = ""

    for run_name, estimator, hyperparams in get_candidate_models(train_cfg):
        pipeline = build_pipeline(estimator, num_features, cat_features)
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        cv_mae = float(-scores.mean())

        pipeline.fit(X, y)
        train_mae = float(mean_absolute_error(y, pipeline.predict(X)))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("stage", "v1")
            mlflow.log_param("service_median_imputed", service_median)
            mlflow.log_params(hyperparams)
            mlflow.log_metric("cv_mae", cv_mae)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            run_id = run.info.run_id

        print(f"{run_name:22s} | cv_mae={cv_mae:.4f} | train_mae={train_mae:.4f} | run_id={run_id}")

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_run_id = run_id
            best_pipeline = pipeline
            best_name = run_name

    reg_name = mlflow_cfg["registered_model_name"]
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=reg_name)

    client = MlflowClient()
    client.set_registered_model_alias(reg_name, "champion", mv.version)

    model_path = Path(paths["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    return {
        "experiment_name": mlflow_cfg["experiment_name"],
        "best_run_id": best_run_id,
        "best_model_name": best_name,
        "best_cv_mae": best_cv_mae,
        "registered_model_name": reg_name,
        "registered_version": str(mv.version),
    }


def main() -> None:
    params = load_params()
    result = train_v1(params)
    print("Registered v1 model:", result)


if __name__ == "__main__":
    main()
