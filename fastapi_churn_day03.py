"""День 3 — Загрузка и предварительный просмотр churn датасета"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"

app = FastAPI()


class FeatureVectorChurn(BaseModel):

    monthly_fee: float        
    usage_hours: float         
    support_requests: int      
    account_age_months: int    
    failed_payments: int       
    region: str                
    device_type: str           
    payment_method: str        
    autopay_enabled: int  


class DatasetRowChurn(FeatureVectorChurn):
    """Строка тренировочного датасета"""

    churn: int 


class ChurnDataset:
    """Читает CSV, валидирует строки через Pydantic и хранит данные в DataFrame."""

    def __init__(self, path: Path, row_model: type[BaseModel]):
        self.path = path
        self.row_model = row_model
        self.df = pd.DataFrame()           
        self.rows: list[BaseModel] = []    

    def load(self) -> "ChurnDataset":
        df = pd.read_csv(self.path)
        # Каждый словарь превращаем в объект DatasetRowChurn.
        self.rows = [self.row_model(**record) for record in df.to_dict("records")]
        self.df = pd.DataFrame([row.model_dump() for row in self.rows])
        return self

    def preview(self, n: int) -> list[dict]:
        # Первые n строк как список словарей — FastAPI превратит его в JSON.
        return self.df.head(n).to_dict("records")

    def info(self) -> dict:
        counts = self.df["churn"].value_counts().sort_index()
        return {
            "rows": int(self.df.shape[0]),       
            "columns": int(self.df.shape[1]),
            "features": [c for c in self.df.columns if c != "churn"],
            "churn_distribution": {str(k): int(v) for k, v in counts.items()},
        }


dataset = ChurnDataset(DATASET_PATH, DatasetRowChurn)


def get_dataset() -> ChurnDataset:
    if dataset.df.empty:
        if not DATASET_PATH.exists():
            raise HTTPException(404, f"Файл датасета не найден: {DATASET_PATH.name}")
        try:
            dataset.load()
        except Exception as e:
            raise HTTPException(500, f"Не удалось загрузить датасет: {e}")
    return dataset


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1, le=1000)):
    return get_dataset().preview(n)


@app.get("/dataset/info")
def dataset_info():
    return get_dataset().info()
