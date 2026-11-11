"""Настройки и константы сервиса."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method", "autopay_enabled"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

ALLOWED_HYPERPARAMETERS = {
    "logreg": {"C", "solver", "max_iter", "class_weight", "random_state"},
    "random_forest": {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "class_weight", "random_state",
    },
}
MAX_N_ESTIMATORS = 1000

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class Settings:
    """Пути к файлам. Читаются в момент использования, поэтому тесты могут их подменять."""

    dataset_path: Path = BASE_DIR / "data" / "churn_dataset.csv"
    model_path: Path = BASE_DIR / "models" / "churn_model.joblib"
    history_path: Path = BASE_DIR / "models" / "training_history.json"


settings = Settings()
