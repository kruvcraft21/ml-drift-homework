from airflow.sdk import dag, task
from airflow.sdk import Variable
from get_data import get_dataset
from drift_check import compute_psi_for_features
from mlflow_worker import count_register_model, train_model, register_model
import pandas as pd
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

def obtain_drifted_data() -> pd.DataFrame:
    """
    Получает набор данных с применённым искусственным дрейфом признаков на основе параметров конфигурации.
    Функционал:
    - Читает параметры дрифта из Airflow Variable с ключами:
        - "DRIFT_TYPE" (тип дрифта, например "none", "concept", "feature"),
        - "DRIFT_MAGNITUDE" (величина дрифта, строковое представление числа),
        - "DRIFT_FRACTION" (доля объектов, к которым применяется дрифт, строковое представление числа).
    - Устанавливает соответствующие переменные окружения (os.environ) на основе считанных значений, чтобы они были доступны для последующей обработки.
    - Вызывает внутрненнюю функцию get_dataset(), которая формирует и возвращает результирующий DataFrame с учётом указанных параметров дрифта.
    Побочные эффекты:
    - Модифицирует os.environ для ключей 'DRIFT_TYPE', 'DRIFT_MAGNITUDE', 'DRIFT_FRACTION'.
    :return: DataFrame с данными, в которых смоделирован указанный дрейф.
    :rtype: pandas.DataFrame
    """
    # Тут указываем параметры дрифта, по умолчанию все none, но для тестов мы можем изменить параметры
    os.environ['DRIFT_TYPE'] = Variable.get("DRIFT_TYPE", default='none')
    os.environ['DRIFT_MAGNITUDE'] = Variable.get("DRIFT_MAGNITUDE", default="0.5")
    os.environ['DRIFT_FRACTION'] = Variable.get("DRIFT_FRACTION", default="0.2")
    return get_dataset()


# Будем запускать даг каждые 30 минут
@dag(
    tags=['ml'],
    schedule="*/30 * * * *",
    catchup=False
)
def my_flow():

    @task()
    def save_dataset():
        obtain_drifted_data().to_csv("dataset.csv")
        log.info("Write dataset")

    dataset = save_dataset()

    def has_drift_occurred(context) -> bool:
        """
        Проверяем был ли дрифт
        """
        if count_register_model() == 0 or not os.path.exists("dataset.csv"):
            # Когда запускаемся всегда True, чтобы запустить обучение моделей
            log.info("Init model")
            return True
        
        ref_df = pd.read_csv("dataset.csv")
        
        cur_df = obtain_drifted_data()

        features = ref_df.drop("target", axis=1).columns.to_list()
        psi_values = compute_psi_for_features(ref_df, cur_df, features)

        max_psi = max(v for v in psi_values.values() if not np.isnan(v)) if psi_values else 0.0
        drift_detected = bool(max_psi > 0.2)

        log.info("PSI value: %s", max_psi)
        if drift_detected:
            log.info("Drift detected")
            return True

        return False
    
    
    @task.run_if(has_drift_occurred)
    @task()
    def execute_model_training():
        log.info("Run train model")
        train_model(obtain_drifted_data())

    run_train = execute_model_training()

    @task
    def execute_model_registration():
        log.info("Run register last model")
        register_model()

    run_reg = execute_model_registration()

    run_train >> dataset >> run_reg
    

my_flow = my_flow()