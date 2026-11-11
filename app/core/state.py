"""Состояние приложения в памяти: обученная модель, история обучений, датасет."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from app.ml.dataset import ChurnDataset


class AppState:
    def __init__(self) -> None:
        self.model: "Pipeline | None" = None
        self.model_info: dict | None = None
        self.training_history: list[dict] = []
        self.dataset: "ChurnDataset | None" = None


state = AppState()
