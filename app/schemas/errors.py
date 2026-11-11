"""Модель ошибки и примеры ошибок для документации /docs."""

from typing import Any

from pydantic import BaseModel


class ErrorResponseChurn(BaseModel):
    """Формат любой ошибки сервиса."""

    code: str
    message: str
    details: Any = None


EX_VALIDATION_MISSING = {
    "summary": "Не хватает признаков",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 2)",
        "details": [
            {"field": "usage_hours", "message": "Field required", "type": "missing"},
            {"field": "region", "message": "Field required", "type": "missing"},
        ],
    },
}
EX_VALIDATION_TYPE = {
    "summary": "Неверный тип значения",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 1)",
        "details": [
            {
                "field": "monthly_fee",
                "message": "Input should be a valid number, unable to parse string as a number",
                "type": "float_parsing",
            }
        ],
    },
}
EX_VALIDATION_EXTRA = {
    "summary": "Лишний признак",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 1)",
        "details": [
            {"field": "age", "message": "Extra inputs are not permitted", "type": "extra_forbidden"}
        ],
    },
}
EX_MODEL_NOT_TRAINED = {
    "summary": "Модель не обучена",
    "value": {
        "code": "MODEL_NOT_TRAINED",
        "message": "Модель ещё не обучена. Сначала вызовите POST /model/train",
        "details": None,
    },
}
EX_EMPTY_REQUEST = {
    "summary": "Пустой список клиентов",
    "value": {"code": "EMPTY_REQUEST", "message": "Список клиентов пуст", "details": None},
}
EX_DATASET_NOT_FOUND = {
    "summary": "Нет файла датасета",
    "value": {
        "code": "DATASET_NOT_FOUND",
        "message": "Файл датасета не найден: churn_dataset.csv",
        "details": {"expected_path": "data/churn_dataset.csv"},
    },
}
EX_DATASET_EMPTY = {
    "summary": "Пустой датасет",
    "value": {
        "code": "DATASET_EMPTY",
        "message": "Датасет пуст: нет данных для обучения",
        "details": None,
    },
}
EX_TRAINING_FAILED = {
    "summary": "Неверные гиперпараметры",
    "value": {
        "code": "TRAINING_FAILED",
        "message": "Невозможно обучить модель: n_estimators должен быть целым от 1 до 1000",
        "details": None,
    },
}


def error_doc(description: str, **examples) -> dict:
    """Описание ответа-ошибки для параметра responses эндпоинта."""
    return {
        "model": ErrorResponseChurn,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }
