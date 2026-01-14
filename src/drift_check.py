import json
from typing import Dict, List

import numpy as np
import pandas as pd
import os
from get_data import get_dataset


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / expected_counts.sum()
    actual_perc = actual_counts / actual_counts.sum()

    # защита от деления на ноль
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    return float(np.sum(psi_values))


def compute_psi_for_features(
    train: pd.DataFrame,
    current: pd.DataFrame,
    features: List[str],
    bins: int = 10,
) -> Dict[str, float]:
    psi_dict = {}
    for f in features:
        if f not in train.columns or f not in current.columns:
            psi_dict[f] = float("nan")
            continue
        psi_dict[f] = calculate_psi(train[f].values.astype(float), current[f].values.astype(float), bins=bins)
    return psi_dict


def main():
    # Создаем управляемый дрифт, чтобы посмотреть как это работает
    os.environ['DRIFT_TYPE'] = 'mean_shift'
    os.environ['DRIFT_MAGNITUDE'] = '50.0'
    train_df = get_dataset()
    os.environ['DRIFT_TYPE'] = 'none'
    # os.environ['DRIFT_MAGNITUDE'] = '50.0'
    test_df = get_dataset()

    features = train_df.drop("target", axis=1).columns.to_list()
    psi_values = compute_psi_for_features(train_df, test_df, features)

    max_psi = max(v for v in psi_values.values() if not np.isnan(v)) if psi_values else 0.0
    drift_detected = bool(max_psi > 0.2)

    result = {
        "drift_detected": drift_detected,
        "psi": psi_values,
        "max_psi": max_psi,
        "threshold": 0.2,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()