"""День 6 — Сохранение и загрузка churn модели.

Запуск:
    uvicorn fastapi_churn_day06:app --reload
Датасет: data/churn_dataset.csv
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"

TEST_SIZE = 0.2       
RANDOM_STATE = 42    


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до yield выполняется один раз при старте приложения
    global model, model_info
    loaded = load_churn_model(MODEL_PATH)
    if loaded is not None:
        model, model_info = loaded
    yield


app = FastAPI(lifespan=lifespan)


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

def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Обучает pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def save_churn_model(model: Pipeline, info: dict, path: Path = MODEL_PATH) -> None:
    """Сохраняет модель вместе с информацией об обучении"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Пишем сначала во временный файл, потом переименовываем
    tmp = path.with_suffix(".tmp")
    joblib.dump({"model": model, "info": info}, tmp)
    tmp.replace(path)


def load_churn_model(path: Path = MODEL_PATH):
    """Загружает модель из файла"""
    if not path.exists():
        return None
    try:
        saved = joblib.load(path)
        return saved["model"], saved["info"]
    except Exception:
        return None


dataset = ChurnDataset(DATASET_PATH, DatasetRowChurn)

model: Pipeline | None = None
model_info: dict | None = None


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


@app.post("/model/train")
def train_model():
    global model, model_info
    df = get_dataset().df
    if df.empty:
        raise HTTPException(400, "Датасет пуст: нет данных для обучения")
    try:
        X, y, _, _ = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        if y_train.nunique() < 2:
            raise ValueError("в обучающей выборке только один класс churn")
        trained = train_churn_model(X_train, y_train)
    except ValueError as e:
        raise HTTPException(422, f"Невозможно обучить модель: {e}")

    y_pred = trained.predict(X_test)
    result = {
        "model": "LogisticRegression",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }
    info = {"trained_at": datetime.now(timezone.utc).isoformat(), "metrics": result}
    try:
        save_churn_model(trained, info)
    except OSError as e:
        raise HTTPException(500, f"Модель обучена, но не сохранена: {e}")
    model, model_info = trained, info
    return result


@app.get("/model/status")
def model_status():
    if model is None or model_info is None:
        return {"trained": False, "trained_at": None, "metrics": None}
    return {
        "trained": True,
        "trained_at": model_info["trained_at"],
        "metrics": model_info["metrics"],
    }

