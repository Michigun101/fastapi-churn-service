import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fastapi_churn_day13 as churn


def make_churn_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Синтетический датасет: небольшой, воспроизводимый (фиксированный seed)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "monthly_fee": rng.choice([9.99, 19.99, 29.99], n),
            "usage_hours": rng.uniform(1, 100, n).round(2),
            "support_requests": rng.integers(0, 8, n),
            "account_age_months": rng.integers(1, 48, n),
            "failed_payments": rng.integers(0, 5, n),
            "region": rng.choice(["europe", "america", "asia"], n),
            "device_type": rng.choice(["mobile", "desktop"], n),
            "payment_method": rng.choice(["card", "paypal"], n),
            "autopay_enabled": rng.integers(0, 2, n),
        }
    )
    # Вероятность ухода зависит от признаков, чтобы у модели было что выучить.
    logit = -1.5 + 0.8 * df["failed_payments"] - 0.03 * df["usage_hours"] + 0.5 * (1 - df["autopay_enabled"])
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


GOOD_CLIENT = {
    "monthly_fee": 9.99,
    "usage_hours": 5.0,
    "support_requests": 6,
    "account_age_months": 1,
    "failed_payments": 3,
    "region": "america",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 0,
}


@pytest.fixture
def df():
    return make_churn_df()


@pytest.fixture
def good_client():
    return dict(GOOD_CLIENT)


@pytest.fixture
def app_env(tmp_path, monkeypatch, df):
    """Изолированное окружение: свой CSV, свои файлы модели и истории, чистое состояние."""
    csv_path = tmp_path / "churn_dataset.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(churn, "DATASET_PATH", csv_path)
    monkeypatch.setattr(churn, "dataset", churn.ChurnDataset(csv_path, churn.DatasetRowChurn))
    monkeypatch.setattr(churn, "MODEL_PATH", tmp_path / "models" / "churn_model.joblib")
    monkeypatch.setattr(churn, "HISTORY_PATH", tmp_path / "models" / "history.json")
    monkeypatch.setattr(churn, "model", None)
    monkeypatch.setattr(churn, "model_info", None)
    monkeypatch.setattr(churn, "training_history", [])
    return tmp_path


@pytest.fixture
def client(app_env):
    # with — запускает lifespan (загрузку модели и истории при старте приложения).
    with TestClient(churn.app) as c:
        yield c
