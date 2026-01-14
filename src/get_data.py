from sklearn.datasets import load_iris
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Подгружаем переменные из .env
# чтобы контролировать и создавать дрифт, для тестов
load_dotenv()

def generate_data() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Генерация данных, 10000 объектов, на 20 фичей, где только 5 инфморативны
    Также будут 3 класса
    
    :rtype: tuple[DataFrame, ndarray]
    """
    data = make_classification(n_samples=10000, n_features=10, n_informative=5, n_classes=3, random_state=0)
    return pd.DataFrame(data[0], columns=[f"f{n}" for n in range(data[0].shape[1])]), data[1]

def get_dataset() -> pd.DataFrame:
    """
    Возвращает датасет с возможным дрейфом, управляемым через env:
      DRIFT_TYPE: none|mean_shift|scale|feature_swap|label_noise (default: none)
      DRIFT_MAGNITUDE: float, масштаб дрейфа (default: 0.5)
      DRIFT_FRACTION: доля строк, затронутых для feature_swap/label_noise (default: 0.2)
    """
    data, y = generate_data()

    drift_type = os.getenv('DRIFT_TYPE', 'none').lower()
    magnitude = float(os.getenv('DRIFT_MAGNITUDE', '0.5'))
    frac = float(os.getenv('DRIFT_FRACTION', '0.2'))
    rng = np.random.default_rng(int(42))
        
    if drift_type == 'mean_shift':
        # аддитивный шум пропорционально std признака
        noise = rng.normal(loc=0, scale=0.1 * magnitude, size=data.shape) * data.std(axis=0).to_numpy()
        data = data + noise

    elif drift_type == 'scale':
        # масштабирование признаков (умножение)
        scales = 1.0 + rng.normal(loc=magnitude, scale=0.1 * magnitude, size=data.shape[1])
        data = data * scales

    elif drift_type == 'feature_swap':
        # поменять местами значения двух признаков в части строк
        n = max(1, int(len(data) * frac))
        cols = data.columns.tolist()
        if len(cols) >= 2:
            a, b = rng.choice(len(cols), size=2, replace=False)
            idx = rng.choice(data.index, size=n, replace=False)
            tmp = data.loc[idx, cols[a]].copy()
            data.loc[idx, cols[a]] = data.loc[idx, cols[b]].values
            data.loc[idx, cols[b]] = tmp.values

    elif drift_type == 'label_noise':
        # изменить метки для части объектов на случайные другие классы
        n = int(len(y) * frac)
        if n > 0:
            idx = rng.choice(len(y), size=n, replace=False)
            classes = np.unique(y)
            y = y.copy()
            for i in idx:
                choices = classes[classes != y.iloc[i]]
                y.iloc[i] = rng.choice(choices)

    data['target'] = y
    return data


def main():
    drift_type = os.getenv('DRIFT_TYPE', 'none').lower()
    magnitude = float(os.getenv('DRIFT_MAGNITUDE', '0.5'))
    frac = float(os.getenv('DRIFT_FRACTION', '0.2'))
    print(drift_type, magnitude, frac)


if __name__ == "__main__":
    main()