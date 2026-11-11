"""Точка входа: создание приложения FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import api_router
from app.core.errors import register_exception_handlers
from app.core.logging_config import logger, setup_logging
from app.core.state import state
from app.ml.persistence import load_churn_model, load_history

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_churn_model()
    if loaded is not None:
        state.model, state.model_info = loaded
        logger.info("Модель загружена из файла: %s", state.model_info.get("model_type"))
    else:
        logger.info("Сохранённой модели нет, нужно вызвать POST /model/train")
    state.training_history = load_history()
    yield


app = FastAPI(title="Churn service", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(api_router)
