"""Эндпоинты работы с моделью: обучение, статус, схема признаков, метрики."""

import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Body, Query

from app.core.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from app.core.errors import ChurnError
from app.core.logging_config import logger
from app.core.state import state
from app.ml.dataset import get_dataset
from app.ml.persistence import add_history_record, best_record, save_churn_model
from app.ml.preprocessing import prepare_data, split_data
from app.ml.training import effective_hyperparameters, evaluate_model, train_churn_model
from app.schemas.errors import (
    EX_DATASET_EMPTY,
    EX_DATASET_NOT_FOUND,
    EX_TRAINING_FAILED,
    EX_VALIDATION_TYPE,
    error_doc,
)
from app.schemas.features import FeatureVectorChurn
from app.schemas.training import TRAIN_EXAMPLES, MetricsResponseChurn, TrainingConfigChurn

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    summary="Обучить модель",
    responses={
        404: error_doc("Нет файла датасета", no_dataset=EX_DATASET_NOT_FOUND),
        400: error_doc("Датасет пуст", empty=EX_DATASET_EMPTY),
        422: error_doc(
            "Неверная конфигурация обучения или данные не подходят для обучения",
            wrong_config=EX_TRAINING_FAILED,
            wrong_type=EX_VALIDATION_TYPE,
        ),
    },
)
def train_model(
    config: TrainingConfigChurn | None = Body(default=None, openapi_examples=TRAIN_EXAMPLES),
):
    config = config or TrainingConfigChurn()
    df = get_dataset().df
    if df.empty:
        raise ChurnError(400, "DATASET_EMPTY", "Датасет пуст: нет данных для обучения")

    started = time.perf_counter()
    logger.info("Обучение запущено: model_type=%s, hyperparameters=%s, строк в датасете=%d",
                config.model_type, config.hyperparameters, len(df))

    try:
        X, y, _, _ = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
    except (KeyError, ValueError, TypeError) as e:
        raise ChurnError(
            422, "DATA_PREPARATION_FAILED", f"Не удалось подготовить данные: {e}"
        )

    try:
        if y_train.nunique() < 2:
            raise ValueError("в обучающей выборке только один класс churn")
        trained = train_churn_model(X_train, y_train, config)
    except (ValueError, TypeError) as e:
        raise ChurnError(422, "TRAINING_FAILED", f"Невозможно обучить модель: {e}")

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        **evaluate_model(trained, X_test, y_test),
    }
    hyperparameters = effective_hyperparameters(trained, config.model_type)
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": config.model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }

    try:
        save_churn_model(trained, info)
    except OSError as e:
        raise ChurnError(
            500, "MODEL_SAVE_FAILED", f"Модель обучена, но не сохранена: {e}"
        )
    state.model, state.model_info = trained, info
    add_history_record(info)
    logger.info("Обучение завершено за %.2f с: accuracy=%s, f1=%s, roc_auc=%s",
                time.perf_counter() - started, metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
    return {"model_type": config.model_type, "hyperparameters": hyperparameters, **metrics}


@router.get("/status")
def model_status():
    if state.model is None or state.model_info is None:
        return {
            "trained": False, "trained_at": None, "model_type": None,
            "hyperparameters": None, "metrics": None,
        }
    info = state.model_info
    return {
        "trained": True,
        "trained_at": info["trained_at"],
        "model_type": info.get("model_type"),
        "hyperparameters": info.get("hyperparameters"),
        "metrics": info["metrics"],
    }


@router.get("/schema")
def model_schema():
    model = state.model
    allowed: dict[str, list] = {}
    if model is not None:
        encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
        allowed = {
            name: cats.tolist() for name, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
        }
    fields = FeatureVectorChurn.model_fields
    order = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return {
        "model_trained": model is not None,
        "feature_order": order,
        "features": [
            {
                "name": name,
                "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
                "type": fields[name].annotation.__name__,
                "known_values": allowed.get(name),
            }
            for name in order
        ],
        "target": TARGET,
    }


@router.get(
    "/metrics",
    response_model=MetricsResponseChurn,
    summary="Метрики и история обучений",
    description=(
        "Возвращает последнее обучение, лучшие записи по f1 и roc_auc и список "
        "последних обучений (новые первыми). Параметр `model_type` оставляет "
        "в расчёте только модели этого типа — так удобно сравнивать настройки."
    ),
)
def model_metrics(
    limit: int = Query(5, ge=1, le=100, description="Сколько последних записей вернуть"),
    model_type: Literal["logreg", "random_forest"] | None = Query(
        None, description="Фильтр по типу модели"
    ),
):
    records = [
        r for r in state.training_history if model_type is None or r["model_type"] == model_type
    ]
    return {
        "total": len(records),
        "latest": records[-1] if records else None,
        "best_by_f1": best_record(records, "f1"),
        "best_by_roc_auc": best_record(records, "roc_auc"),
        # История хранится от старых к новым, а отдаём последние limit штук, новые первыми.
        "history": records[::-1][:limit],
    }
