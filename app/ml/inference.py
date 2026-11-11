"""Предсказание churn обученной моделью."""

from sklearn.pipeline import Pipeline

from app.ml.preprocessing import features_to_frame
from app.schemas.features import FeatureVectorChurn
from app.schemas.prediction import PredictionResponseChurn


def predict_churn(model: Pipeline, items: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    """Класс и вероятности для каждого клиента (в том же порядке, что и items)."""
    X = features_to_frame(items, model)
    classes = list(model.classes_)
    predicted = model.predict(X)
    proba = model.predict_proba(X)
    i0, i1 = classes.index(0), classes.index(1)
    return [
        PredictionResponseChurn(
            predicted_churn=int(pred),
            probability_no_churn=round(float(p[i0]), 4),
            probability_churn=round(float(p[i1]), 4),
        )
        for pred, p in zip(predicted, proba)
    ]
