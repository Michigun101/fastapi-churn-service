"""Подготовка данных: разделение признаков и цели, train/test, предобработка."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from app.schemas.features import FeatureVectorChurn


def prepare_data(df: pd.DataFrame):
    """Готовит данные к обучению.

    Возвращает X (признаки), y (целевая переменная `churn`),
    а также списки числовых и категориальных столбцов.
    """
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
    """Делит данные на train и test (80% / 20%)."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def class_distribution(y: pd.Series) -> dict:
    """Количество и доля каждого класса churn."""
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return {
        str(k): {"count": int(v), "share": round(int(v) / total, 4)}
        for k, v in counts.items()
    }


def build_preprocessor() -> ColumnTransformer:
    """Предобработка признаков: масштабирование чисел + one-hot для категорий."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def features_to_frame(items: list[FeatureVectorChurn], model) -> pd.DataFrame:
    """Превращает объекты запроса в таблицу с теми же столбцами и порядком, что при обучении."""
    columns = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return pd.DataFrame([item.model_dump() for item in items])[columns]
