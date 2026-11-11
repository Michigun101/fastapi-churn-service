"""Модель ответа /predict и примеры запросов для документации."""

from pydantic import BaseModel, ConfigDict


class PredictionResponseChurn(BaseModel):
    """Предсказание модели для одного клиента."""

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
