"""Чтение churn датасета и доступ к нему."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.errors import ChurnError
from app.core.logging_config import logger
from app.core.state import state
from app.schemas.features import DatasetRowChurn


class ChurnDataset:
    """Читает CSV, валидирует строки через Pydantic и хранит данные в DataFrame."""

    def __init__(self, path: Path, row_model: type[BaseModel]):
        self.path = path
        self.row_model = row_model
        self.df = pd.DataFrame()

    def load(self) -> "ChurnDataset":
        df = pd.read_csv(self.path)
        rows = [self.row_model(**record) for record in df.to_dict("records")]
        self.df = pd.DataFrame([row.model_dump() for row in rows])
        return self

    def preview(self, n: int) -> list[dict]:
        return self.df.head(n).to_dict("records")

    def info(self) -> dict:
        counts = self.df["churn"].value_counts().sort_index()
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "features": [c for c in self.df.columns if c != "churn"],
            "churn_distribution": {str(k): int(v) for k, v in counts.items()},
        }


def get_dataset() -> ChurnDataset:
    """Возвращает загруженный датасет; при первом обращении читает файл."""
    path = settings.dataset_path
    if state.dataset is None or state.dataset.path != path:
        state.dataset = ChurnDataset(path, DatasetRowChurn)
    dataset = state.dataset
    if dataset.df.empty:
        if not path.exists():
            raise ChurnError(
                404,
                "DATASET_NOT_FOUND",
                f"Файл датасета не найден: {path.name}",
                {"expected_path": f"data/{path.name}"},
            )
        try:
            dataset.load()
        except pd.errors.EmptyDataError:
            raise ChurnError(400, "DATASET_EMPTY", "Файл датасета пуст")
        except ValidationError as e:
            details = [
                {"field": ".".join(map(str, err["loc"])), "message": err["msg"], "type": err["type"]}
                for err in e.errors()[:5]
            ]
            raise ChurnError(
                422, "DATASET_INVALID", "Строки датасета не соответствуют схеме", details
            )
        except Exception as e:
            raise ChurnError(500, "DATASET_LOAD_FAILED", f"Не удалось прочитать датасет: {e}")
        logger.info("Датасет загружен: %s (строк: %d, столбцов: %d)", path.name, *dataset.df.shape)
    return dataset
