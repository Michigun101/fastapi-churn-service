"""Эндпоинт предсказания churn."""

from fastapi import APIRouter, Body

from app.core.errors import ChurnError
from app.core.logging_config import logger
from app.core.state import state
from app.ml.inference import predict_churn
from app.schemas.errors import (
    EX_EMPTY_REQUEST,
    EX_MODEL_NOT_TRAINED,
    EX_VALIDATION_EXTRA,
    EX_VALIDATION_MISSING,
    EX_VALIDATION_TYPE,
    error_doc,
)
from app.schemas.features import FeatureVectorChurn
from app.schemas.prediction import PREDICT_EXAMPLES, PredictionResponseChurn

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Предсказать churn",
    description=(
        "Принимает одного клиента (объект) или несколько (список). "
        "Для одного клиента возвращает один объект, для списка — список "
        "предсказаний в том же порядке. Если модель не обучена — ошибка 503."
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
    if state.model is None:
        raise ChurnError(
            503, "MODEL_NOT_TRAINED",
            "Модель ещё не обучена. Сначала вызовите POST /model/train",
        )

    is_single = isinstance(payload, FeatureVectorChurn)
    items = [payload] if is_single else payload
    if not items:
        raise ChurnError(422, "EMPTY_REQUEST", "Список клиентов пуст")

    try:
        results = predict_churn(state.model, items)
    except (KeyError, ValueError, TypeError) as e:
        raise ChurnError(
            422, "PREDICTION_FAILED", f"Не удалось получить предсказание: {e}"
        )

    # В лог пишем только количество и итог, без самих значений признаков клиентов.
    logger.info("predict: клиентов=%d, предсказано churn=1: %d",
                len(results), sum(r.predicted_churn for r in results))
    return results[0] if is_single else results
