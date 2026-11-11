"""День 7 — Эндпоинт предсказания churn"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "monthly_fee": 19.99,
                    "usage_hours": 21.5,
                    "support_requests": 2,
                    "account_age_months": 14,
                    "failed_payments": 1,
                    "region": "europe",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                }
            ]
        }
    )


class PredictionResponseChurn(BaseModel):
    """Предсказание модели для одного клиента"""

    predicted_churn: int            
    probability_no_churn: float     
    probability_churn: float        

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predicted_churn": 0,
                    "probability_no_churn": 0.8123,
                    "probability_churn": 0.1877,
                }
            ]
        }
    )


class DatasetRowChurn(FeatureVectorChurn):
    """Строка обучающего датасета"""

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


# Примеры запроса для /docs
PREDICT_EXAMPLES = {
    "single": {
        "summary": "Один клиент",
        "value": {
            "monthly_fee": 19.99,
            "usage_hours": 21.5,
            "support_requests": 2,
            "account_age_months": 14,
            "failed_payments": 1,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    },
    "list": {
        "summary": "Список клиентов",
        "value": [
            {
                "monthly_fee": 9.99,
                "usage_hours": 5.0,
                "support_requests": 6,
                "account_age_months": 1,
                "failed_payments": 3,
                "region": "america",
                "device_type": "desktop",
                "payment_method": "card",
                "autopay_enabled": 0,
            },
            {
                "monthly_fee": 29.99,
                "usage_hours": 80.0,
                "support_requests": 0,
                "account_age_months": 36,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1,
            },
        ],
    },
}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Предсказать churn",
    description=(
        "Принимает одного или несколько клиентов. "
        "Для одного клиента возвращает один объект, для списка — список "
    ),
    responses={503: {"description": "Модель ещё не обучена (вызовите POST /model/train)"}},
)
def predict(
    payload: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        openapi_examples=PREDICT_EXAMPLES
    ),
):
    if model is None:
        raise HTTPException(503, "Модель ещё не обучена. Сначала вызовите POST /model/train")

    is_single = isinstance(payload, FeatureVectorChurn)
    items = [payload] if is_single else payload
    if not items:
        raise HTTPException(422, "Список клиентов пуст")

    X = pd.DataFrame([item.model_dump() for item in items])[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    classes = list(model.classes_)
    predicted = model.predict(X)
    proba = model.predict_proba(X)
    i0, i1 = classes.index(0), classes.index(1)

    results = [
        PredictionResponseChurn(
            predicted_churn=int(pred),
            probability_no_churn=round(float(p[i0]), 4),
            probability_churn=round(float(p[i1]), 4),
        )
        for pred, p in zip(predicted, proba)
    ]
    return results[0] if is_single else results


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

