"""День 2 — Pydantic-модели для признаков churn.

Запуск:
    uvicorn fastapi_churn_day02:app --reload
"""

from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()


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


class DatasetRowChurn(FeatureVectorChurn):
    """Строка тренировочного датасета"""

    churn: int                 


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn):
    # временный эндпоинт: возвращает входные данные, чтобы проверить схему
    return features

