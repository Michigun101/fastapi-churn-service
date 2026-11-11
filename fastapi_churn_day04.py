"""День 4 — Предобработка churn данных и разбиение на train и test"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method", "autopay_enabled"]
TARGET = "churn"

TEST_SIZE = 0.2       
RANDOM_STATE = 42     

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
    """Читает CSV, валидирует строки через Pydantic и хранит данные в DataFrame"""

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


def prepare_data(df: pd.DataFrame):
    """Готовит данные к обучению"""
    df = df.dropna(subset=[TARGET])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)


    for col in NUMERIC_FEATURES:
        X[col] = X[col].fillna(X[col].median())
    for col in CATEGORICAL_FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode().iloc[0])

    return X, y, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def split_data(X: pd.DataFrame, y: pd.Series):
    """Делит данные на train и test"""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def class_distribution(y: pd.Series) -> dict:
    """Количество и доля каждого класса churn"""
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return {
        str(k): {"count": int(v), "share": round(int(v) / total, 4)}
        for k, v in counts.items()
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


@app.get("/dataset/split-info")
def dataset_split_info():
    X, y, numeric, categorical = prepare_data(get_dataset().df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "train_churn_distribution": class_distribution(y_train),
        "test_churn_distribution": class_distribution(y_test),
    }

