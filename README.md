# ML FastAPI — сервис предсказания оттока (churn)

Учебный проект: пошаговая разработка ML-сервиса на FastAPI.
README обновляется по мере выполнения заданий.

## Прогресс

| День | Тема | Статус |
|------|------|--------|
| 1 | Базовое приложение FastAPI | ✅ выполнено |

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Сервис будет доступен на http://127.0.0.1:8000
Интерактивная документация (Swagger): http://127.0.0.1:8000/docs

## Эндпоинты

| Метод | Путь | Описание | Ответ |
|-------|------|----------|-------|
| GET | `/` | Проверка работы сервиса | `{"message": "ml churn service is running"}` |

Проверка:

```bash
curl http://127.0.0.1:8000/
```

## Структура проекта

```
.
├── main.py            # приложение FastAPI
├── requirements.txt   # зависимости
└── README.md
```

## Журнал по дням

### День 1 — Базовое приложение FastAPI
**Цель:** поднять минимальное приложение FastAPI и убедиться, что оно работает.

- [x] Установлены `fastapi` и `uvicorn`
- [x] Создан `main.py`
- [x] Создан объект `app = FastAPI()`
- [x] Добавлен `GET /` → `{"message": "ml churn service is running"}`
- [x] Приложение запущено через `uvicorn main:app --reload`, сервис отвечает
