"""Модели признаков клиента и строки датасета."""

from pydantic import BaseModel, ConfigDict

from app.core.config import FEATURE_COLUMNS


class FeatureVectorChurn(BaseModel):
    """Признаки одного клиента — вход модели предсказания."""

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
        },
    )


assert set(FEATURE_COLUMNS) == set(FeatureVectorChurn.model_fields), (
    "NUMERIC_FEATURES + CATEGORICAL_FEATURES не совпадают с полями FeatureVectorChurn"
)


class DatasetRowChurn(FeatureVectorChurn):
    """Строка обучающего датасета: признаки + целевая переменная."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=True)

    churn: int
