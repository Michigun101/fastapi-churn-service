# Churn service — предсказание оттока клиентов

REST-сервис на FastAPI, который обучает модель машинного обучения на данных о клиентах подписочного сервиса и предсказывает, уйдёт ли клиент (**churn**). Проект учебный: разработан по шагам, от минимального приложения до контейнеризации (журнал по дням — [docs/development-log.md](docs/development-log.md)).

## Что умеет сервис

- загружать и показывать тренировочный датасет (`/dataset/*`);
- обучать модель (логистическая регрессия или случайный лес) с настраиваемыми гиперпараметрами (`POST /model/train`);
- сохранять модель и историю обучений на диск, поднимать их при перезапуске;
- выдавать предсказание и вероятности для одного клиента или списка (`POST /predict`);
- показывать состояние модели, метрики и сравнение обучений (`/model/status`, `/model/metrics`, `/model/schema`);
- отвечать на любые ошибки единым JSON-форматом, писать логи, отдавать `/health`.

Интерактивная документация (Swagger) доступна на `/docs`.

## Формат датасета `data/churn_dataset.csv`

CSV с заголовком, одна строка — один клиент. Нужны ровно эти столбцы (порядок не важен, лишние столбцы игнорируются):

| Столбец | Тип | Описание | Значения в датасете проекта |
|---------|-----|----------|-----------------------------|
| `monthly_fee` | float | ежемесячный платёж | 9.99 … 49.99 |
| `usage_hours` | float | время использования сервиса, часы | 0 … 52.07 |
| `support_requests` | int | обращений в поддержку | 0 … 4 |
| `account_age_months` | int | возраст аккаунта, месяцы | 1 … 39 |
| `failed_payments` | int | неудачных платежей | 0 … 2 |
| `region` | str | регион | africa, america, asia, europe |
| `device_type` | str | тип устройства | desktop, mobile, tablet |
| `payment_method` | str | способ оплаты | card, crypto, paypal |
| `autopay_enabled` | int | автоплатёж: 1 — включён, 0 — выключен | 0, 1 |
| `churn` | int | **целевая метка**: 1 — клиент ушёл, 0 — остался | 0, 1 |

Датасет проекта: 2000 строк, ушедших клиентов около 20 % (403 из 2000). Пропуски в `monthly_fee` и `usage_hours` заполняются медианой; пропуск в категориальных или целочисленных столбцах считается ошибкой данных (`DATASET_INVALID`).

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

## Структура проекта

```
.
├── app/
│   ├── main.py              # создание приложения, загрузка модели при старте
│   ├── api/                 # эндпоинты (роутеры)
│   │   ├── health.py        #   GET /, GET /health
│   │   ├── dataset.py       #   GET /dataset/preview, /info, /split-info
│   │   ├── model.py         #   POST /model/train, GET /model/status, /schema, /metrics
│   │   └── predict.py       #   POST /predict
│   ├── ml/                  # ML-пайплайн
│   │   ├── dataset.py       #   чтение и валидация CSV
│   │   ├── preprocessing.py #   X/y, пропуски, train/test, ColumnTransformer
│   │   ├── training.py      #   выбор модели, обучение, метрики
│   │   ├── inference.py     #   предсказание
│   │   └── persistence.py   #   сохранение модели и истории обучений
│   ├── schemas/             # Pydantic-модели запросов и ответов
│   └── core/                # настройки, логирование, ошибки, состояние в памяти
├── tests/                   # pytest (unit + интеграционные)
├── data/churn_dataset.csv   # датасет
├── models/                  # обученная модель и история (создаются сервисом)
├── docs/development-log.md  # журнал разработки по дням
├── fastapi_churn_day01…14.py# версии сервиса по дням (учебная история)
├── Dockerfile, .dockerignore
└── requirements.txt, requirements-dev.txt, pytest.ini
```

Зависимости направлены в одну сторону: `main → api → ml → schemas, core`.

## Запуск локально

Нужен Python 3.11 или новее (проверено на 3.14).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Сервис: http://127.0.0.1:8000, документация: http://127.0.0.1:8000/docs.
Сразу после запуска модели ещё нет — её нужно обучить (`POST /model/train`, см. ниже). Дальше модель и история сохраняются в `models/` и подхватываются при следующем запуске.

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
# с сохранением модели и истории между запусками контейнера:
docker run -p 8000:8000 -v churn-models:/app/models churn-service
```

Образ на базе `python:3.14-slim`, сервис работает не от root, `HEALTHCHECK` опрашивает `/health`. Датасет копируется в образ (чтобы использовать другой, подключите его через `-v путь/к/data:/app/data`). Логи: `docker logs <контейнер>`.

## Примеры запросов

Ниже `curl` (Linux/macOS/Git Bash). В Windows PowerShell используйте `curl.exe` или `Invoke-RestMethod`; проще всего — кнопка **Try it out** в `/docs`.

### Обучение: `POST /model/train`

Тело запроса необязательно: без него обучается логистическая регрессия с параметрами по умолчанию.

```bash
curl -X POST http://127.0.0.1:8000/model/train
```
```json
{"model_type": "logreg",
 "hyperparameters": {"C": 1.0, "class_weight": null, "max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
 "train_size": 1600, "test_size": 400, "accuracy": 0.79, "f1": 0.0455, "roc_auc": 0.6089}
```

Выбор модели и гиперпараметров (`model_type`: `logreg` или `random_forest`):

```bash
curl -X POST http://127.0.0.1:8000/model/train -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 6, "class_weight": "balanced"}}'
```

Допустимые гиперпараметры: `logreg` — `C`, `solver`, `max_iter`, `class_weight`, `random_state`; `random_forest` — `n_estimators` (1–1000), `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `random_state`. Метрики считаются на тестовой выборке (20 % данных, стратифицированное разбиение).

Другие эндпоинты модели: `GET /model/status` (обучена ли, когда, тип, метрики), `GET /model/metrics?limit=5&model_type=logreg` (история и лучшие обучения), `GET /model/schema` (признаки, типы, допустимые значения категорий).

### Предсказание: `POST /predict`

Один клиент:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "monthly_fee": 9.99, "usage_hours": 5.0, "support_requests": 6, "account_age_months": 1,
  "failed_payments": 3, "region": "america", "device_type": "desktop",
  "payment_method": "card", "autopay_enabled": 0}'
```
```json
{"predicted_churn": 1, "probability_no_churn": 0.1216, "probability_churn": 0.8784}
```

Список клиентов (ответ — список в том же порядке):

```json
[{"predicted_churn": 1, "probability_no_churn": 0.1216, "probability_churn": 0.8784},
 {"predicted_churn": 0, "probability_no_churn": 0.996,  "probability_churn": 0.004}]
```

Все 9 признаков обязательны, лишние поля не допускаются. Пример результата для рискованного клиента (новый аккаунт, мало использования, неудачные платежи, нет автоплатежа) и лояльного (3 года, активный, без проблем с платежами).

### Ошибки

Любая ошибка возвращается в одном формате:

```json
{"code": "MODEL_NOT_TRAINED", "message": "Модель ещё не обучена. Сначала вызовите POST /model/train", "details": null}
```

| HTTP | `code` | Когда |
|------|--------|-------|
| 422 | `VALIDATION_ERROR` | не хватает или лишний признак, неверный тип, некорректный JSON; `details` — список `{field, message, type}` |
| 422 | `EMPTY_REQUEST` | `/predict` получил пустой список |
| 503 | `MODEL_NOT_TRAINED` | `/predict` вызван до обучения |
| 404 | `DATASET_NOT_FOUND` | нет файла `data/churn_dataset.csv` |
| 400 | `DATASET_EMPTY` | датасет пуст |
| 422 | `DATASET_INVALID` | строки CSV не соответствуют схеме |
| 422 | `DATA_PREPARATION_FAILED`, `TRAINING_FAILED`, `PREDICTION_FAILED` | не удалось подготовить данные, обучить модель (например, неверные гиперпараметры) или получить предсказание |
| 500 | `DATASET_LOAD_FAILED`, `MODEL_SAVE_FAILED`, `INTERNAL_ERROR` | сбой чтения датасета, записи модели, неожиданная ошибка (трассировка только в логе) |
| 404 / 405 | `NOT_FOUND` / `METHOD_NOT_ALLOWED` | неверный путь или метод |

## Состояние сервиса и логи

`GET /health` всегда отвечает `200`, пока процесс жив:

```json
{"status": "degraded", "model_loaded": false, "model_type": null, "dataset_loaded": false, "dataset_file_found": true}
```
`status` — `ok`, когда модель загружена и файл датасета найден, иначе `degraded`. Датасет читается при первом обращении.

Логи (`время УРОВЕНЬ churn: сообщение`) покрывают загрузку модели и датасета, обучение (параметры, метрики, время), вызовы `/predict` (только число клиентов и итог, без значений признаков) и ошибки.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

68 тестов на синтетических данных во временных папках (настоящие `data/` и `models/` не затрагиваются): функции подготовки данных, обучения и сохранения, полный сценарий через `TestClient` (датасет → обучение → статус → предсказание → перезапуск), обработка ошибок, `/health` и логирование.

## Качество модели

На датасете проекта признаки слабо разделяют классы: `roc_auc` ≈ 0.58–0.61 у всех проверенных настроек. Классы несбалансированы (≈ 80 / 20), поэтому без `class_weight` модель почти всегда предсказывает «клиент останется»: `accuracy` ≈ 0.79, но `f1` по классу «ушёл» ≈ 0.05. С `class_weight=balanced` `f1` поднимается до ≈ 0.35 ценой `accuracy` ≈ 0.59. Сравнение настроек — в [журнале разработки](docs/development-log.md) (День 11); свои сравнения можно получить через `GET /model/metrics`.
