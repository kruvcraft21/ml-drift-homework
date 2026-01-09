from airflow.sdk import dag, task
from airflow.sdk import Variable
from get_data import get_dataset
from drift_check import compute_psi_for_features
from mlflow_worker import count_register_model, train_register_model
import pandas as pd
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

# Будем запускать даг каждые 30 минут
@dag(
    tags=['ml'],
    schedule="*/30 * * * *",
    catchup=False
)
def my_flow():

    @task()
    def save_dataset():
        was_drift = Variable.get("was_drift", default='false') == 'true'
        # Сохраняем датасет, если его нет (например когда мы запускаемся)
        # Также когда был дрифт и мы обучили модель на новых данных
        if not os.path.exists("dataset.csv") or was_drift:
            get_dataset().to_csv("dataset.csv")
            Variable.set("was_drift", "false")

    dataset = save_dataset()

    def is_drift() -> bool:
        """
        Проверяем был ли дрифт
        """
        if count_register_model() == 0:
            # Когда запускаемся всегда True, чтобы запустить обучение моделей
            log.info("Init model")
            return True
        
        ref_df = pd.read_csv("dataset.csv")
        # Тут указываем параметры дрифта, по умолчанию все none, но для тестов мы можем изменить параметры
        os.environ['DRIFT_TYPE'] = Variable.get("DRIFT_TYPE", default='none')
        os.environ['DRIFT_MAGNITUDE'] = Variable.get("DRIFT_MAGNITUDE", default="0.5")
        os.environ['DRIFT_FRACTION'] = Variable.get("DRIFT_FRACTION", default="0.2")
        cur_df = get_dataset()

        features = ref_df.drop("target", axis=1).columns.to_list()
        psi_values = compute_psi_for_features(ref_df, cur_df, features)

        max_psi = max(v for v in psi_values.values() if not np.isnan(v)) if psi_values else 0.0
        drift_detected = bool(max_psi > 0.2)

        if drift_detected:
            log.info("Drift detected")
            Variable.set("was_drift", "true")
            return True

        return False
    
    
    @task.run_if(lambda x: is_drift())
    @task()
    def train_model():
        log.info("Run train model")
        train_register_model()

    run_train = train_model()

    dataset >> run_train
    

my_flow = my_flow()