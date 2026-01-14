import argparse
import os

import mlflow

import mlflow.sklearn
import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, remove_metric
# from get_data import get_dataset

mlflow.set_tracking_uri("http://mlflow:5000/")


def train_model(train: pd.DataFrame):
    # train = get_dataset()
    # Создаем эксперимент
    clf1 = setup(train, target = 'target', log_experiment = True, experiment_name = 'clf1', log_plots=True, log_data=True)
    # Сравниваем модели
    best = compare_models(exclude=['roc_auc_score'])
    # Отдельно создадим финальную модель, чтобы проще найти её потом
    final_model = finalize_model(best)
    

def register_model():
    # Ищем модель в нашем эксперименте, pycaret автоматически выбрал лучшую
    runs = mlflow.search_runs(
        experiment_names=['clf1'], 
        filter_string='tags.Source = "finalize_model"', 
        order_by=['end_time DESC'], 
        max_results=1)
    
    run_id = runs.loc[0, "run_id"]
    # Сразу регистрируем её
    # Механика stage в mlflow помечена как устаревшей, поэтому я решил использовать теги
    # На мой взгляд теги удобнее чем алиасы
    ver = mlflow.register_model(f"runs:/{run_id}/model", "prod_model", tags={"stage": 'staging'})


def count_register_model():
    return len(mlflow.search_registered_models())

def main():
    train_model()
    register_model()


if __name__ == "__main__":
    main()
