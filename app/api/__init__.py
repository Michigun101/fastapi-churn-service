from fastapi import APIRouter

from app.api import dataset, health, model, predict

api_router = APIRouter()
for module in (health, dataset, model, predict):
    api_router.include_router(module.router)
