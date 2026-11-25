# MP047 — Used Two-Wheeler Price (Indore)

**Group ID:** G138  
**Problem ID:** MP047  
**Batch:** 37  
**Faculty:** Dr. Ravi Pratap  

## Group Members

| Enrolment No. | Name |
|---|---|
| S24CSEU1622 | PrinceTomar |
| S24CSEU1636 | PrathamKishor |
| S24CSEU1628 | DakshRathore |
| S24CSEU1651 | AayushBhatt |

## Overview

End-to-end MLOps pipeline for predicting second-hand two-wheeler resale prices (`resale_thousand_rs`, in thousand INR) in Indore, evaluated with MAE.

## Repository Layout

- `params.yaml`: Pipeline paths, cleaning rules, MLflow, training, and monitoring settings
- `src/bike_price/`: Modular Python package (`config.py`, `data_audit.py`, `cleaning.py`, `train.py`, `monitoring.py`, `predict.py`)
- `app/main.py`: FastAPI prediction service (`/health`, `/predict`) with Pydantic validation and JSON logging
- `tests/`: Unit and API tests (`pytest` + `TestClient`)
- `data/raw/`: DVC-tracked raw data (`train.csv.dvc`, `later.csv.dvc`) and `holdout_features.csv`
- `mlflow.db`: SQLite MLflow tracking store and model registry
- `models/model.joblib`: Active champion model bundle used by the API
- `k8s/`: Kubernetes Deployment and Service manifests
- `Dockerfile` & `.dockerignore`: Container packaging for `mp047-api:1.0`
- `.github/workflows/ci.yml`: Automated pytest workflow on push
- `reports/findings.yaml`: Data audit, tracking, and drift/retraining findings
- `predictions/holdout_predictions.csv`: Final holdout predictions

## Running the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

1. **Audit raw training data for the 7 data issues:**

```bash
python -m src.bike_price.data_audit
```

2. **Clean training data:**

```bash
python -m src.bike_price.cleaning
```

3. **Train candidate v1 models, log to MLflow, and register champion v1:**

```bash
python -m src.bike_price.train
```

4. **Run drift monitoring (PSI), group MAE analysis, and v2 retraining:**

```bash
python -m src.bike_price.monitoring
```

5. **Generate holdout predictions with the `@champion` model:**

```bash
python -m src.bike_price.predict
```

6. **Run tests:**

```bash
pytest
```

7. **Run the FastAPI service locally or with Docker:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
docker build -t mp047-api:1.0 .
docker run -p 8000:8000 mp047-api:1.0
```
