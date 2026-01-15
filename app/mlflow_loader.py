import mlflow
import logging
from typing import Optional
from mlflow.pyfunc import _PythonModelPyfuncWrapper
from mlflow.entities.model_registry import ModelVersion
import os

logger = logging.getLogger("mlflow_loader")
MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_ENDPOINT)

lazy_loader : dict[str, _PythonModelPyfuncWrapper] = {}


def load_last_model(model_name: str, stage: str) -> list[ModelVersion]:
    """
    Load a model by its tag stage (production/staging) from MLflow Registry.
    """
    return mlflow.search_model_versions(
        filter_string=f'name="{model_name}" and tags.stage = "{stage}"'
        , order_by=['last_updated_timestamp DESC']
        , max_results=1
        )

def load_model_by_version(model_version: ModelVersion) -> _PythonModelPyfuncWrapper:
    if model_version.source in lazy_loader.keys():
        return lazy_loader[model_version.source]
    model = mlflow.pyfunc.load_model(model_version.source)
    lazy_loader[model_version.source] = model
    return model


def load_production_model(model_name: str) -> list[ModelVersion]:
    """
    Load LAST the Production model for a given name.
    """
    return load_last_model(model_name, "production")


def load_staging_model(model_name: str) -> list[ModelVersion]:
    """
    Load LAST the Staging model for a given name.
    """
    return load_last_model(model_name, "staging")