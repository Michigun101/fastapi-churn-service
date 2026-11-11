"""День 13 — Мониторинг churn сервиса и Docker"""

from contextlib import asynccontextmanager
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from starlette.exceptions import HTTPException as StarletteHTTPException

# Формат логов: время, уровень, имя логгера, сообщение. 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("churn")

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method", "autopay_enabled"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"
HISTORY_PATH = Path(__file__).parent / "models" / "training_history.json"

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
    global model, model_info
    loaded = load_churn_model(MODEL_PATH)
    if loaded is not None:
        model, model_info = loaded
        logger.info("Модель загружена из файла: %s", model_info.get("model_type"))
    else:
        logger.info("Сохранённой модели нет, нужно вызвать POST /model/train")
    training_history[:] = load_history(HISTORY_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class ErrorResponseChurn(BaseModel):
    """Формат любой ошибки сервиса."""

    code: str
    message: str
    details: Any = None


class ChurnError(Exception):
    """Ожидаемая ошибка сервиса: HTTP-статус + код + сообщение + подробности."""

    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_response(status_code: int, code: str, message: str, details: Any = None):
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "details": details},
    )


HTTP_ERROR_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}


@app.exception_handler(ChurnError)
async def churn_error_handler(request: Request, exc: ChurnError):
    # Серверные ошибки (5xx) логируем как ERROR, клиентские (4xx) — как WARNING.
    level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(level, "%s %s -> %s %s: %s", request.method, request.url.path,
               exc.status_code, exc.code, exc.message)
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = HTTP_ERROR_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")
    logger.warning("%s %s -> %s %s", request.method, request.url.path, exc.status_code, code)
    return error_response(exc.status_code, code, str(exc.detail))


UNION_BRANCHES = ("FeatureVectorChurn", "list[FeatureVectorChurn]")


def clean_validation_errors(exc: RequestValidationError) -> list[dict]:
    """Превращает ошибки Pydantic в список"""
    errors = exc.errors()
    branch = None
    if isinstance(exc.body, dict):
        branch = UNION_BRANCHES[0]
    elif isinstance(exc.body, list):
        branch = UNION_BRANCHES[1]
    if branch and any(len(e["loc"]) > 1 and e["loc"][1] == branch for e in errors):
        errors = [e for e in errors if len(e["loc"]) > 1 and e["loc"][1] == branch]

    result = []
    for e in errors:
        loc = e["loc"]
        skip = 2 if len(loc) > 1 and loc[1] in UNION_BRANCHES else 1
        path = ""
        for part in loc[skip:]:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        if e["type"] == "json_invalid":
            path = ""
        result.append({"field": path or None, "message": e["msg"], "type": e["type"]})
    return result


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = clean_validation_errors(exc)
    logger.warning("%s %s -> 422 VALIDATION_ERROR: %s", request.method, request.url.path, details)
    return error_response(
        422,
        "VALIDATION_ERROR",
        f"Входные данные не прошли проверку (ошибок: {len(details)})",
        details,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error("%s %s -> 500 INTERNAL_ERROR", request.method, request.url.path, exc_info=exc)
    return error_response(500, "INTERNAL_ERROR", "Внутренняя ошибка сервиса")


EX_VALIDATION_MISSING = {
    "summary": "Не хватает признаков",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 2)",
        "details": [
            {"field": "usage_hours", "message": "Field required", "type": "missing"},
            {"field": "region", "message": "Field required", "type": "missing"},
        ],
    },
}
EX_VALIDATION_TYPE = {
    "summary": "Неверный тип значения",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 1)",
        "details": [
            {
                "field": "monthly_fee",
                "message": "Input should be a valid number, unable to parse string as a number",
                "type": "float_parsing",
            }
        ],
    },
}
EX_VALIDATION_EXTRA = {
    "summary": "Лишний признак",
    "value": {
        "code": "VALIDATION_ERROR",
        "message": "Входные данные не прошли проверку (ошибок: 1)",
        "details": [
            {"field": "age", "message": "Extra inputs are not permitted", "type": "extra_forbidden"}
        ],
    },
}
EX_MODEL_NOT_TRAINED = {
    "summary": "Модель не обучена",
    "value": {
        "code": "MODEL_NOT_TRAINED",
        "message": "Модель ещё не обучена. Сначала вызовите POST /model/train",
        "details": None,
    },
}
EX_EMPTY_REQUEST = {
    "summary": "Пустой список клиентов",
    "value": {"code": "EMPTY_REQUEST", "message": "Список клиентов пуст", "details": None},
}
EX_DATASET_NOT_FOUND = {
    "summary": "Нет файла датасета",
    "value": {
        "code": "DATASET_NOT_FOUND",
        "message": "Файл датасета не найден: churn_dataset.csv",
        "details": {"expected_path": "data/churn_dataset.csv"},
    },
}
EX_DATASET_EMPTY = {
    "summary": "Пустой датасет",
    "value": {
        "code": "DATASET_EMPTY",
        "message": "Датасет пуст: нет данных для обучения",
        "details": None,
    },
}
EX_TRAINING_FAILED = {
    "summary": "Неверные гиперпараметры",
    "value": {
        "code": "TRAINING_FAILED",
        "message": "Невозможно обучить модель: n_estimators должен быть целым от 1 до 1000",
        "details": None,
    },
}


def error_doc(description: str, **examples) -> dict:
    """Описание ответа-ошибки для параметра responses эндпоинта"""
    return {
        "model": ErrorResponseChurn,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }


class FeatureVectorChurn(BaseModel):
    """Признаки одного клиента """

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
        extra="forbid",
        allow_inf_nan=False,
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


class TrainingMetricsChurn(BaseModel):
    """Метрики одного обучения"""

    accuracy: float
    f1: float
    roc_auc: float | None   
    train_size: int
    test_size: int


class TrainingRecordChurn(BaseModel):
    """Запись в истории обучений."""

    id: int                          
    timestamp: str                   
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: TrainingMetricsChurn


class MetricsResponseChurn(BaseModel):
    """Ответ GET /model/metrics."""

    total: int                                  
    latest: TrainingRecordChurn | None          
    best_by_f1: TrainingRecordChurn | None      
    best_by_roc_auc: TrainingRecordChurn | None 
    history: list[TrainingRecordChurn]          


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


assert set(FEATURE_COLUMNS) == set(FeatureVectorChurn.model_fields), (
    "NUMERIC_FEATURES + CATEGORICAL_FEATURES не совпадают с полями FeatureVectorChurn"
)


class DatasetRowChurn(FeatureVectorChurn):
    """Строка обучающего датасета"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=True)

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
        self.rows = [self.row_model(**record) for record in df.to_dict("records")]
        self.df = pd.DataFrame([row.model_dump() for row in self.rows])
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


def prepare_data(df: pd.DataFrame):
    """Готовит данные к обучению"""
    df = df.dropna(subset=[TARGET])

    X = df[FEATURE_COLUMNS].copy()
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
    """Создаёт модель нужного типа с заданными гиперпараметрами"""
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


def build_preprocessor() -> ColumnTransformer:
    """Предобработка признаков"""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfigChurn | None = None,
) -> Pipeline:
    """Обучает единый pipeline"""
    config = config or TrainingConfigChurn()
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_classifier(config)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def features_to_frame(items: list[FeatureVectorChurn], model: Pipeline) -> pd.DataFrame:
    """Превращает объекты запроса в таблицу"""
    columns = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return pd.DataFrame([item.model_dump() for item in items])[columns]


# История обучений в памяти; дублируется в HISTORY_PATH.
training_history: list[dict] = []


# Пути по умолчанию в функциях ниже берутся при вызове (path=None), а не при
# определении функции. Благодаря этому тесты могут подменить MODEL_PATH и
# HISTORY_PATH на временные файлы и не трогать настоящую модель.
def load_history(path: Path | None = None) -> list[dict]:
    """Читает историю из JSON-файла"""
    path = path or HISTORY_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError) as e:
        logging.getLogger("uvicorn.error").warning("Не удалось прочитать историю: %s", e)
        return []


def save_history(history: list[dict], path: Path | None = None) -> None:
    """Записывает историю в JSON"""
    path = path or HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_history_record(info: dict) -> dict:
    """Добавляет запись об обучении в историю и сохраняет файл"""
    record = {
        "id": len(training_history) + 1,
        "timestamp": info["trained_at"],
        "model_type": info["model_type"],
        "hyperparameters": info["hyperparameters"],
        "metrics": info["metrics"],
    }
    training_history.append(record)
    try:
        save_history(training_history)
    except OSError as e:
        # Модель уже обучена и сохранена — сбой записи истории не должен
        logging.getLogger("uvicorn.error").warning("Не удалось сохранить историю: %s", e)
    return record


def effective_hyperparameters(model: Pipeline, model_type: str) -> dict:
    """Фактические значения гиперпараметров обученной модели"""
    params = model.named_steps["classifier"].get_params()
    return {k: params[k] for k in sorted(ALLOWED_HYPERPARAMETERS[model_type])}


def save_churn_model(model: Pipeline, info: dict, path: Path | None = None) -> None:
    """Сохраняет модель вместе с информацией об обучении"""
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump({"model": model, "info": info}, tmp)
    tmp.replace(path)


def load_churn_model(path: Path | None = None):
    """Загружает модель из файла"""
    path = path or MODEL_PATH
    if not path.exists():
        return None
    try:
        saved = joblib.load(path)
        return saved["model"], saved["info"]
    except Exception as e:
        logging.getLogger("uvicorn.error").warning("Не удалось загрузить модель: %s", e)
        return None


dataset = ChurnDataset(DATASET_PATH, DatasetRowChurn)

model: Pipeline | None = None
model_info: dict | None = None


def get_dataset() -> ChurnDataset:
    if dataset.df.empty:
        if not DATASET_PATH.exists():
            raise ChurnError(
                404,
                "DATASET_NOT_FOUND",
                f"Файл датасета не найден: {DATASET_PATH.name}",
                {"expected_path": f"data/{DATASET_PATH.name}"},
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
        logger.info("Датасет загружен: %s (строк: %d, столбцов: %d)",
                    DATASET_PATH.name, *dataset.df.shape)
    return dataset


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


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
    responses={
        422: error_doc(
            "Некорректные входные данные",
            missing=EX_VALIDATION_MISSING,
            wrong_type=EX_VALIDATION_TYPE,
            extra=EX_VALIDATION_EXTRA,
            empty_list=EX_EMPTY_REQUEST,
        ),
        503: error_doc("Модель ещё не обучена", not_trained=EX_MODEL_NOT_TRAINED),
    },
)
def predict(
    payload: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        openapi_examples=PREDICT_EXAMPLES
    ),
):
    if model is None:
        raise ChurnError(
            503, "MODEL_NOT_TRAINED",
            "Модель ещё не обучена. Сначала вызовите POST /model/train",
        )

    is_single = isinstance(payload, FeatureVectorChurn)
    items = [payload] if is_single else payload
    if not items:
        raise ChurnError(422, "EMPTY_REQUEST", "Список клиентов пуст")

    try:
        X = features_to_frame(items, model)
        classes = list(model.classes_)
        predicted = model.predict(X)
        proba = model.predict_proba(X)
        i0, i1 = classes.index(0), classes.index(1)
    except (KeyError, ValueError, TypeError) as e:
        raise ChurnError(
            422, "PREDICTION_FAILED", f"Не удалось получить предсказание: {e}"
        )

    results = [
        PredictionResponseChurn(
            predicted_churn=int(pred),
            probability_no_churn=round(float(p[i0]), 4),
            probability_churn=round(float(p[i1]), 4),
        )
        for pred, p in zip(predicted, proba)
    ]
    # В лог пишем только количество и итог, без самих значений признаков клиентов.
    logger.info("predict: клиентов=%d, предсказано churn=1: %d",
                len(results), sum(r.predicted_churn for r in results))
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


@app.post(
    "/model/train",
    summary="Обучить модель",
    responses={
        404: error_doc("Нет файла датасета", no_dataset=EX_DATASET_NOT_FOUND),
        400: error_doc("Датасет пуст", empty=EX_DATASET_EMPTY),
        422: error_doc(
            "Неверная конфигурация обучения или данные не подходят для обучения",
            wrong_config=EX_TRAINING_FAILED,
            wrong_type=EX_VALIDATION_TYPE,
        ),
    },
)
def train_model(
    config: TrainingConfigChurn | None = Body(default=None, openapi_examples=TRAIN_EXAMPLES),
):
    global model, model_info
    config = config or TrainingConfigChurn()
    df = get_dataset().df
    if df.empty:
        raise ChurnError(400, "DATASET_EMPTY", "Датасет пуст: нет данных для обучения")

    started = time.perf_counter()
    logger.info("Обучение запущено: model_type=%s, hyperparameters=%s, строк в датасете=%d",
                config.model_type, config.hyperparameters, len(df))

    try:
        X, y, _, _ = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
    except (KeyError, ValueError, TypeError) as e:
        raise ChurnError(
            422, "DATA_PREPARATION_FAILED", f"Не удалось подготовить данные: {e}"
        )

    try:
        if y_train.nunique() < 2:
            raise ValueError("в обучающей выборке только один класс churn")
        trained = train_churn_model(X_train, y_train, config)
    except (ValueError, TypeError) as e:
        raise ChurnError(422, "TRAINING_FAILED", f"Невозможно обучить модель: {e}")

    y_pred = trained.predict(X_test)
    # roc_auc считаем по вероятности класса 1; он невычислим, если в тесте один класс.
    roc_auc = None
    if y_test.nunique() == 2:
        proba_churn = trained.predict_proba(X_test)[:, list(trained.classes_).index(1)]
        roc_auc = round(float(roc_auc_score(y_test, proba_churn)), 4)
    result = {
        "model_type": config.model_type,
        "hyperparameters": effective_hyperparameters(trained, config.model_type),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": roc_auc,
    }
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": result["model_type"],
        "hyperparameters": result["hyperparameters"],
        "metrics": {k: result[k] for k in ("train_size", "test_size", "accuracy", "f1", "roc_auc")},
    }

    try:
        save_churn_model(trained, info)
    except OSError as e:
        raise ChurnError(
            500, "MODEL_SAVE_FAILED", f"Модель обучена, но не сохранена: {e}"
        )
    model, model_info = trained, info
    # Каждое успешное обучение добавляется в историю.
    add_history_record(info)
    logger.info("Обучение завершено за %.2f с: accuracy=%s, f1=%s, roc_auc=%s",
                time.perf_counter() - started, result["accuracy"], result["f1"], result["roc_auc"])
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


@app.get("/model/schema")
def model_schema():
    allowed: dict[str, list] = {}
    if model is not None:
        encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
        allowed = {
            name: cats.tolist() for name, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
        }
    fields = FeatureVectorChurn.model_fields
    order = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return {
        "model_trained": model is not None,
        "feature_order": order,
        "features": [
            {
                "name": name,
                "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
                "type": fields[name].annotation.__name__,
                "known_values": allowed.get(name),
            }
            for name in order
        ],
        "target": TARGET,
    }


def best_record(records: list[dict], metric: str) -> dict | None:
    """Запись с наибольшим значением метрики (записи без значения пропускаются)."""
    candidates = [r for r in records if r["metrics"].get(metric) is not None]
    return max(candidates, key=lambda r: r["metrics"][metric], default=None)


@app.get(
    "/model/metrics",
    response_model=MetricsResponseChurn,
    summary="Метрики и история обучений",
    description=(
        "Возвращает последнее обучение, лучшие записи по f1 и roc_auc и список "
        "последних обучений (новые первыми). Параметр `model_type` оставляет "
        "в расчёте только модели этого типа — так удобно сравнивать настройки."
    ),
)
def model_metrics(
    limit: int = Query(5, ge=1, le=100, description="Сколько последних записей вернуть"),
    model_type: Literal["logreg", "random_forest"] | None = Query(
        None, description="Фильтр по типу модели"
    ),
):
    records = [
        r for r in training_history if model_type is None or r["model_type"] == model_type
    ]
    return {
        "total": len(records),
        "latest": records[-1] if records else None,
        "best_by_f1": best_record(records, "f1"),
        "best_by_roc_auc": best_record(records, "roc_auc"),
        "history": records[::-1][:limit],
    }


@app.get("/health", summary="Состояние сервиса")
def health():
    # Эндпоинт всегда отвечает 200, пока процесс жив: по нему Docker проверяет работоспособность
    model_loaded = model is not None
    dataset_file_found = DATASET_PATH.exists()
    return {
        "status": "ok" if model_loaded and dataset_file_found else "degraded",
        "model_loaded": model_loaded,
        "model_type": model_info.get("model_type") if model_info else None,
        "dataset_loaded": not dataset.df.empty,
        "dataset_file_found": dataset_file_found,
    }

