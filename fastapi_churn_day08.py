"""День 8 — Конфигурация обучения и выбор модели churn"""

from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

ALLOWED_HYPERPARAMETERS = {
    "logreg": {"C", "solver", "max_iter", "class_weight", "random_state"},
    "random_forest": {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "class_weight", "random_state",
    },
}
MAX_N_ESTIMATORS = 1000   

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


class TrainingConfigChurn(BaseModel):
    """Настройки обучения"""

    # Literal ограничивает допустимые значения
    model_type: Literal["logreg", "random_forest"] = "logreg"
    hyperparameters: dict[str, Any] = {}

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"model_type": "random_forest",
                 "hyperparameters": {"n_estimators": 200, "max_depth": 6}}
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


def build_classifier(config: TrainingConfigChurn):
    params = dict(config.hyperparameters)
    unknown = set(params) - ALLOWED_HYPERPARAMETERS[config.model_type]
    if unknown:
        allowed = sorted(ALLOWED_HYPERPARAMETERS[config.model_type])
        raise ValueError(
            f"неизвестные гиперпараметры {sorted(unknown)} для {config.model_type}; "
            f"допустимы: {allowed}"
        )
    n_estimators = params.get("n_estimators")
    if n_estimators is not None and not (
        isinstance(n_estimators, int) and 1 <= n_estimators <= MAX_N_ESTIMATORS
    ):
        raise ValueError(f"n_estimators должен быть целым от 1 до {MAX_N_ESTIMATORS}")

    if config.model_type == "logreg":
        return LogisticRegression(**{"max_iter": 1000, "random_state": RANDOM_STATE, **params})
    return RandomForestClassifier(**{"random_state": RANDOM_STATE, **params})


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfigChurn | None = None,
) -> Pipeline:
    config = config or TrainingConfigChurn()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", build_classifier(config)),
        ]
    )
    model.fit(X_train, y_train)
    return model



def effective_hyperparameters(model: Pipeline, model_type: str) -> dict:
    """Фактические значения гиперпараметров обученной модели"""
    params = model.named_steps["classifier"].get_params()
    return {k: params[k] for k in sorted(ALLOWED_HYPERPARAMETERS[model_type])}


def save_churn_model(model: Pipeline, info: dict, path: Path = MODEL_PATH) -> None:
    """Сохраняет модель вместе с информацией об обучении (время, метрики)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Пишем сначала во временный файл, потом переименовываем: если процесс
    # прервётся посреди записи, старый файл модели не будет испорчен.
    tmp = path.with_suffix(".tmp")
    # joblib.dump сериализует объект на диск (эффективно для массивов numpy).
    joblib.dump({"model": model, "info": info}, tmp)
    tmp.replace(path)


def load_churn_model(path: Path = MODEL_PATH):
    """Загружает модель из файла. Возвращает (model, info) или None."""
    if not path.exists():
        return None
    try:
        # Внимание: joblib.load использует pickle — загружать можно только
        # собственные файлы, которым доверяешь.
        saved = joblib.load(path)
        return saved["model"], saved["info"]
    except Exception as e:
        # Причину пишем в лог, чтобы ошибку было видно в терминале.
        logging.getLogger("uvicorn.error").warning("Не удалось загрузить модель: %s", e)
        # Файл повреждён или несовместим — приложение должно запуститься
        # без модели, а не упасть.
        return None


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


# Примеры запроса для /docs: один клиент и список из двух.
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
        "Принимает одного клиента (объект) или несколько (список). "
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


TRAIN_EXAMPLES = {
    "logreg": {
        "summary": "Логистическая регрессия",
        "value": {"model_type": "logreg", "hyperparameters": {"C": 0.5}},
    },
    "random_forest": {
        "summary": "Случайный лес",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 6},
        },
    },
}


@app.post("/model/train")
def train_model(
    config: TrainingConfigChurn | None = Body(default=None, openapi_examples=TRAIN_EXAMPLES),
):
    global model, model_info
    config = config or TrainingConfigChurn()
    df = get_dataset().df
    if df.empty:
        raise HTTPException(400, "Датасет пуст: нет данных для обучения")
    try:
        X, y, _, _ = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        if y_train.nunique() < 2:
            raise ValueError("в обучающей выборке только один класс churn")
        trained = train_churn_model(X_train, y_train, config)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Невозможно обучить модель: {e}")

    y_pred = trained.predict(X_test)
    result = {
        "model_type": config.model_type,
        "hyperparameters": effective_hyperparameters(trained, config.model_type),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }

    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": result["model_type"],
        "hyperparameters": result["hyperparameters"],
        "metrics": {k: result[k] for k in ("train_size", "test_size", "accuracy", "f1")},
    }

    try:
        save_churn_model(trained, info)
    except OSError as e:
        raise HTTPException(500, f"Модель обучена, но не сохранена: {e}")
    model, model_info = trained, info
    return result


@app.get("/model/status")
def model_status():
    if model is None or model_info is None:
        return {
            "trained": False, "trained_at": None, "model_type": None,
            "hyperparameters": None, "metrics": None,
        }
    return {
        "trained": True,
        "trained_at": model_info["trained_at"],
        "model_type": model_info.get("model_type"),
        "hyperparameters": model_info.get("hyperparameters"),
        "metrics": model_info["metrics"],
    }

