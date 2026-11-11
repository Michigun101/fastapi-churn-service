"""Создание, обучение и оценка модели."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from app.core.config import ALLOWED_HYPERPARAMETERS, MAX_N_ESTIMATORS, RANDOM_STATE
from app.ml.preprocessing import build_preprocessor
from app.schemas.training import TrainingConfigChurn


def build_classifier(config: TrainingConfigChurn):
    """Создаёт модель нужного типа с заданными гиперпараметрами."""
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
    """Обучает единый pipeline: предобработка + выбранная модель."""
    config = config or TrainingConfigChurn()
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_classifier(config)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def effective_hyperparameters(model: Pipeline, model_type: str) -> dict:
    """Фактические значения гиперпараметров обученной модели (с учётом умолчаний)."""
    params = model.named_steps["classifier"].get_params()
    return {k: params[k] for k in sorted(ALLOWED_HYPERPARAMETERS[model_type])}


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Метрики на тестовой выборке: accuracy, f1 и roc_auc."""
    y_pred = model.predict(X_test)
    # roc_auc считаем по вероятности класса 1; он невычислим, если в тесте один класс.
    roc_auc = None
    if y_test.nunique() == 2:
        proba_churn = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        roc_auc = round(float(roc_auc_score(y_test, proba_churn)), 4)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": roc_auc,
    }
