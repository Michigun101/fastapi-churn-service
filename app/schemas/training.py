"""Модели обучения: конфигурация, метрики, история."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TrainingConfigChurn(BaseModel):
    """Настройки обучения: тип модели и её гиперпараметры."""

    model_type: Literal["logreg", "random_forest"] = "logreg"
    hyperparameters: dict[str, Any] = {}

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"model_type": "random_forest",
                 "hyperparameters": {"n_estimators": 200, "max_depth": 6}}
            ]
        }
    )


class TrainingMetricsChurn(BaseModel):
    """Метрики одного обучения, посчитанные на тестовой выборке."""

    accuracy: float
    f1: float
    roc_auc: float | None   # None, если в тесте только один класс
    train_size: int
    test_size: int


class TrainingRecordChurn(BaseModel):
    """Запись в истории обучений."""

    id: int
    timestamp: str
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: TrainingMetricsChurn


class MetricsResponseChurn(BaseModel):
    """Ответ GET /model/metrics."""

    total: int
    latest: TrainingRecordChurn | None
    best_by_f1: TrainingRecordChurn | None
    best_by_roc_auc: TrainingRecordChurn | None
    history: list[TrainingRecordChurn]


TRAIN_EXAMPLES = {
    "logreg": {
        "summary": "Логистическая регрессия",
        "value": {"model_type": "logreg", "hyperparameters": {"C": 0.5}},
    },
    "random_forest": {
        "summary": "Случайный лес",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 6},
        },
    },
}
