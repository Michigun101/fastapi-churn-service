"""Служебные эндпоинты: проверка работы и состояния сервиса."""

from fastapi import APIRouter

from app.core.config import settings
from app.core.state import state

router = APIRouter()


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/health", summary="Состояние сервиса")
def health():
    # Эндпоинт всегда отвечает 200, пока процесс жив: по нему Docker проверяет
    # работоспособность. Готовность к работе описывают поля ниже.
    model_loaded = state.model is not None
    dataset_file_found = settings.dataset_path.exists()
    return {
        "status": "ok" if model_loaded and dataset_file_found else "degraded",
        "model_loaded": model_loaded,
        "model_type": state.model_info.get("model_type") if state.model_info else None,
        "dataset_loaded": state.dataset is not None and not state.dataset.df.empty,
        "dataset_file_found": dataset_file_found,
    }
