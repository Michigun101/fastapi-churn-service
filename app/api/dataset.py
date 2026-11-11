"""Эндпоинты для просмотра датасета."""

from fastapi import APIRouter, Query

from app.ml.dataset import get_dataset
from app.ml.preprocessing import class_distribution, prepare_data, split_data

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(n: int = Query(5, ge=1, le=1000)):
    return get_dataset().preview(n)


@router.get("/info")
def dataset_info():
    return get_dataset().info()


@router.get("/split-info")
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
