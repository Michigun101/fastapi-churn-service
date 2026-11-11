"""Сохранение и загрузка модели и истории обучений."""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.logging_config import logger
from app.core.state import state


def save_churn_model(model: Pipeline, info: dict, path: Path | None = None) -> None:
    """Сохраняет модель вместе с информацией об обучении (время, метрики)."""
    path = path or settings.model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump({"model": model, "info": info}, tmp)
    tmp.replace(path)


def load_churn_model(path: Path | None = None):
    """Загружает модель из файла. Возвращает (model, info) или None."""
    path = path or settings.model_path
    if not path.exists():
        return None
    try:
        saved = joblib.load(path)
        return saved["model"], saved["info"]
    except Exception as e:
        logger.warning("Не удалось загрузить модель: %s", e)
        return None


def load_history(path: Path | None = None) -> list[dict]:
    """Читает историю из JSON-файла; при отсутствии или ошибке возвращает пустой список."""
    path = path or settings.history_path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError) as e:
        logger.warning("Не удалось прочитать историю: %s", e)
        return []


def save_history(history: list[dict], path: Path | None = None) -> None:
    """Записывает историю в JSON (через временный файл, чтобы не испортить старую)."""
    path = path or settings.history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_history_record(info: dict) -> dict:
    """Добавляет запись об обучении в историю и сохраняет файл."""
    record = {
        "id": len(state.training_history) + 1,
        "timestamp": info["trained_at"],
        "model_type": info["model_type"],
        "hyperparameters": info["hyperparameters"],
        "metrics": info["metrics"],
    }
    state.training_history.append(record)
    try:
        save_history(state.training_history)
    except OSError as e:
        # Модель уже обучена и сохранена — сбой записи истории не должен
        # превращать успешное обучение в ошибку.
        logger.warning("Не удалось сохранить историю: %s", e)
    return record


def best_record(records: list[dict], metric: str) -> dict | None:
    """Запись с наибольшим значением метрики (записи без значения пропускаются)."""
    candidates = [r for r in records if r["metrics"].get(metric) is not None]
    return max(candidates, key=lambda r: r["metrics"][metric], default=None)
