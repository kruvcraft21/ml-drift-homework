from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager, suppress
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import random
import os
from app.logger import get_logger
from datetime import datetime, timezone
import asyncio
from dataclasses import dataclass, field

from app.mlflow_loader import load_production_model, load_staging_model, load_model_by_version
from csv_logger import CsvLogger

# ---- CONFIG ----
MODEL_NAME = os.getenv("MODEL_NAME", "prod_model")
TRAFFIC_TO_B = float(os.getenv("TRAFFIC_TO_B", "0.3"))  # Default: 30% traffic to B
LOG_FILE = "logs/router_logs.log"
AB_FILE = "logs/ab_stats.csv"
MODEL_REFRESH_INTERVAL = int(os.getenv("MODEL_REFRESH_INTERVAL", "300"))  # seconds

# ---- GLOBAL VARIABLES ----
@dataclass
class ModelsManager():
    prod_models: list = field(default_factory=list)
    staging_models: list = field(default_factory=list)
    traffic_split: float = TRAFFIC_TO_B

    def clear(self):
        self.prod_models.clear()
        self.staging_models.clear()
    
    def load_models(self, model_name):
        self.prod_models = load_production_model(model_name)
        self.staging_models = load_staging_model(model_name)

    def refresh_traffic(self, traffic_to_b):
        self.traffic_split = traffic_to_b

models_manger = ModelsManager()
TRAFFIC_SPLIT = TRAFFIC_TO_B
model_refresh_task = None

# ---- SETUP LOGGING ----
logger = get_logger(LOG_FILE)
ab_logger = CsvLogger(AB_FILE, 
                      datefmt="%Y-%m-%dT%H:%M:%S%z", 
                      header=["timestamp", "user_id", "assigned_group", "model_stage", "prediction", "features"]
                      )

# ---- INPUT PAYLOAD SCHEMA ----
class PredictionRequest(BaseModel):
    user_id: int
    features: Dict[str, float]


# ---- UTILS ----
def log_request(user_id, assigned_group, model_stage, features: dict, prediction):
    """
    Log request for A/B testing purposes.
    """
    _features = "|".join(f"{key}={val}" for key, val in features.items())
    message = f'{user_id},{assigned_group},{model_stage},{prediction},"{_features}"'
    logger.info(message)
    ab_logger.info(message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models from MLflow Registry on service startup and refresh them in background.
    """

    # initial load
    models_manger.load_models(MODEL_NAME)
    logger.info("Init models.")
    async def refresh_models_periodically():
        while True:
            logger.info("Refresh models")
            await asyncio.sleep(MODEL_REFRESH_INTERVAL)
            models_manger.clear()
            models_manger.load_models(MODEL_NAME)

    model_refresh_task = asyncio.create_task(refresh_models_periodically())
    
    yield

    if model_refresh_task:
        model_refresh_task.cancel()
        with suppress(asyncio.CancelledError):
            await model_refresh_task


app = FastAPI(lifespan=lifespan, )

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    A/B traffic (Production vs Staging) prediction API.
    """
    if len(models_manger.prod_models) == 0:
        raise HTTPException(status_code=500, detail="No models available for prediction.")

    # Choose A or B
    assigned_group = "A"
    model_version = models_manger.prod_models[0]
    if random.random() < models_manger.traffic_split and len(models_manger.staging_models) != 0:
        assigned_group = "B"
        model_version = models_manger.staging_models[0]

    model = load_model_by_version(model_version)
    # Predict
    try:
        features_df = pd.DataFrame([request.features])
        prediction = model.predict(features_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Log request
    log_request(
        user_id=request.user_id,
        assigned_group=assigned_group,
        model_stage="Production" if assigned_group == "A" else "Staging",
        features=request.features,
        prediction=prediction.tolist(),
    )
    return {
        "user_id": request.user_id,
        "assigned_group": assigned_group,
        "model_stage": "Production" if assigned_group == "A" else "Staging",
        "prediction": prediction.tolist(),
    }


@app.post("/ab/config")
async def update_traffic_split(traffic_to_b: float):
    if not (0.0 <= traffic_to_b <= 1.0):
        raise HTTPException(status_code=400, detail="Traffic split must be between 0.0 and 1.0")
    models_manger.refresh_traffic(traffic_to_b)
    return {"traffic_to_b": models_manger.traffic_split}


@app.post("/model/refresh")
async def refresh_models_endpoint(model_name: str = MODEL_NAME):
    """
    Принудительная перезагрузка списка моделей из MLflow.
    Опционально можно передать model_name в query (например: /model/refresh?model_name=prod_model).
    """
    models_manger.clear()
    models_manger.load_models(model_name)
    return {
        "status": "ok",
        "model_name": model_name,
        "prod_model": len(models_manger.prod_models),
        "staging_models": len(models_manger.staging_models)
    }

@app.get("/ab/stats")
async def get_stats():
    try:
        df = await asyncio.to_thread(pd.read_csv, AB_FILE)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{AB_FILE} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read stats: {e}")
    return df.to_dict('records')