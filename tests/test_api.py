import pytest
from fastapi.testclient import TestClient

import fastapi_churn_day13 as churn


def train(client, **body):
    return client.post("/model/train", json=body or None)


class TestFullScenario:
    """Сценарий: датасет -> обучение -> статус -> предсказание."""

    def test_dataset_endpoints(self, client, df):
        info = client.get("/dataset/info").json()
        assert info["rows"] == len(df)
        assert info["columns"] == df.shape[1]
        assert "churn" not in info["features"]
        assert sum(info["churn_distribution"].values()) == len(df)

        preview = client.get("/dataset/preview", params={"n": 3}).json()
        assert len(preview) == 3
        assert set(preview[0]) == set(df.columns)

        split = client.get("/dataset/split-info").json()
        assert split["train_size"] + split["test_size"] == len(df)

    def test_train_status_predict(self, client, good_client):
        status = client.get("/model/status").json()
        assert status["trained"] is False

        resp = train(client)
        assert resp.status_code == 200
        result = resp.json()
        assert result["model_type"] == "logreg"
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["f1"] <= 1
        assert 0 <= result["roc_auc"] <= 1

        status = client.get("/model/status").json()
        assert status["trained"] is True
        assert status["model_type"] == "logreg"
        assert status["metrics"]["accuracy"] == result["accuracy"]

        single = client.post("/predict", json=good_client)
        assert single.status_code == 200
        body = single.json()
        assert body["predicted_churn"] in (0, 1)
        assert body["probability_no_churn"] + body["probability_churn"] == pytest.approx(1, abs=1e-3)

        many = client.post("/predict", json=[good_client, {**good_client, "failed_payments": 0}])
        assert many.status_code == 200
        assert len(many.json()) == 2

    def test_random_forest_config(self, client, good_client):
        resp = train(client, model_type="random_forest", hyperparameters={"n_estimators": 20})
        assert resp.status_code == 200
        assert resp.json()["hyperparameters"]["n_estimators"] == 20
        assert client.post("/predict", json=good_client).status_code == 200

    def test_schema_after_training(self, client):
        assert client.get("/model/schema").json()["features"][5]["known_values"] is None
        train(client)
        features = {f["name"]: f for f in client.get("/model/schema").json()["features"]}
        assert sorted(features["region"]["known_values"]) == ["america", "asia", "europe"]
        assert features["monthly_fee"]["type"] == "float"

    def test_history_and_metrics(self, client):
        train(client)
        train(client, model_type="random_forest", hyperparameters={"n_estimators": 10})
        data = client.get("/model/metrics").json()
        assert data["total"] == 2
        assert data["latest"]["model_type"] == "random_forest"
        assert [r["id"] for r in data["history"]] == [2, 1]
        only_logreg = client.get("/model/metrics", params={"model_type": "logreg"}).json()
        assert only_logreg["total"] == 1
        assert client.get("/model/metrics", params={"limit": 1}).json()["history"][0]["id"] == 2

    def test_model_survives_restart(self, client, app_env, good_client):
        train(client)
        before = client.post("/predict", json=good_client).json()
        # Имитируем перезапуск: сбрасываем память и запускаем приложение заново.
        churn.model = None
        churn.model_info = None
        churn.training_history = []
        with TestClient(churn.app) as restarted:
            assert restarted.get("/model/status").json()["trained"] is True
            assert restarted.post("/predict", json=good_client).json() == before
            assert restarted.get("/model/metrics").json()["total"] == 1


class TestErrors:
    def test_predict_without_model(self, client, good_client):
        resp = client.post("/predict", json=good_client)
        assert resp.status_code == 503
        assert resp.json()["code"] == "MODEL_NOT_TRAINED"

    def test_status_without_model(self, client):
        data = client.get("/model/status").json()
        assert data["trained"] is False and data["metrics"] is None

    def test_metrics_empty_history(self, client):
        data = client.get("/model/metrics").json()
        assert data == {
            "total": 0, "latest": None, "best_by_f1": None,
            "best_by_roc_auc": None, "history": [],
        }

    def test_predict_missing_feature(self, client, good_client):
        train(client)
        del good_client["usage_hours"]
        resp = client.post("/predict", json=good_client)
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "VALIDATION_ERROR"
        assert body["details"][0]["field"] == "usage_hours"

    def test_predict_extra_feature(self, client, good_client):
        train(client)
        resp = client.post("/predict", json={**good_client, "age": 30})
        assert resp.status_code == 422
        assert resp.json()["details"][0]["type"] == "extra_forbidden"

    def test_predict_wrong_type(self, client, good_client):
        train(client)
        resp = client.post("/predict", json={**good_client, "monthly_fee": "abc"})
        assert resp.status_code == 422
        assert resp.json()["details"][0]["field"] == "monthly_fee"

    def test_predict_error_points_to_list_item(self, client, good_client):
        train(client)
        resp = client.post("/predict", json=[good_client, {"monthly_fee": 1}])
        assert resp.status_code == 422
        assert resp.json()["details"][0]["field"].startswith("[1].")

    def test_predict_empty_list(self, client):
        train(client)
        resp = client.post("/predict", json=[])
        assert resp.status_code == 422
        assert resp.json()["code"] == "EMPTY_REQUEST"

    def test_predict_invalid_json(self, client):
        resp = client.post("/predict", content="{bad", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"

    def test_train_invalid_model_type(self, client):
        resp = train(client, model_type="svm")
        assert resp.status_code == 422
        assert resp.json()["code"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize(
        "config",
        [
            {"model_type": "logreg", "hyperparameters": {"C": -1}},
            {"model_type": "logreg", "hyperparameters": {"foo": 1}},
            {"model_type": "random_forest", "hyperparameters": {"n_estimators": 100000}},
        ],
    )
    def test_train_invalid_hyperparameters(self, client, config):
        resp = client.post("/model/train", json=config)
        assert resp.status_code == 422
        assert resp.json()["code"] == "TRAINING_FAILED"
        assert client.get("/model/status").json()["trained"] is False
        assert client.get("/model/metrics").json()["total"] == 0

    def test_failed_training_keeps_previous_model(self, client):
        train(client)
        before = client.get("/model/status").json()
        train(client, hyperparameters={"C": -1})
        assert client.get("/model/status").json() == before

    def test_train_without_dataset(self, client, app_env):
        (app_env / "churn_dataset.csv").unlink()
        resp = train(client)
        assert resp.status_code == 404
        assert resp.json()["code"] == "DATASET_NOT_FOUND"

    def test_train_with_header_only_dataset(self, client, app_env, df):
        df.head(0).to_csv(app_env / "churn_dataset.csv", index=False)
        resp = train(client)
        assert resp.status_code == 400
        assert resp.json()["code"] == "DATASET_EMPTY"

    def test_train_with_empty_file(self, client, app_env):
        (app_env / "churn_dataset.csv").write_text("")
        resp = train(client)
        assert resp.status_code == 400
        assert resp.json()["code"] == "DATASET_EMPTY"

    def test_dataset_with_bad_row(self, client, app_env, df):
        bad = df.copy().astype({"monthly_fee": object})
        bad.loc[0, "monthly_fee"] = "abc"
        bad.to_csv(app_env / "churn_dataset.csv", index=False)
        resp = client.get("/dataset/info")
        assert resp.status_code == 422
        assert resp.json()["code"] == "DATASET_INVALID"

    def test_dataset_with_missing_columns(self, client, app_env, df):
        df[["monthly_fee", "churn"]].to_csv(app_env / "churn_dataset.csv", index=False)
        assert client.get("/dataset/info").json()["code"] == "DATASET_INVALID"

    def test_dataset_with_one_class(self, client, app_env, df):
        df.assign(churn=0).to_csv(app_env / "churn_dataset.csv", index=False)
        resp = train(client)
        assert resp.status_code == 422
        assert resp.json()["code"] == "TRAINING_FAILED"

    def test_missing_float_values_are_filled(self, client, app_env, df):
        df = df.copy()
        df.loc[0, "monthly_fee"] = None
        df.loc[1, "usage_hours"] = None
        df.to_csv(app_env / "churn_dataset.csv", index=False)
        assert train(client).status_code == 200

    def test_missing_categorical_value_is_rejected(self, client, app_env, df):
        # Сейчас пропуск в str/int-столбце не проходит валидацию строки датасета.
        df = df.copy()
        df.loc[1, "region"] = None
        df.to_csv(app_env / "churn_dataset.csv", index=False)
        resp = train(client)
        assert resp.status_code == 422
        assert resp.json()["code"] == "DATASET_INVALID"

    def test_unknown_route_and_method(self, client):
        assert client.get("/nope").json()["code"] == "NOT_FOUND"
        resp = client.delete("/predict")
        assert resp.status_code == 405
        assert resp.json()["code"] == "METHOD_NOT_ALLOWED"

    def test_unexpected_error_hides_traceback(self, app_env, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(churn, "prepare_data", boom)
        # raise_server_exceptions=False: вместо проброса ошибки в тест получаем ответ сервера.
        with TestClient(churn.app, raise_server_exceptions=False) as client:
            resp = client.post("/model/train")
        assert resp.status_code == 500
        assert resp.json()["code"] == "INTERNAL_ERROR"
        assert "secret" not in resp.text
        assert "Traceback" not in resp.text


class TestHealth:
    def test_degraded_before_training(self, client):
        data = client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["dataset_loaded"] is False
        assert data["dataset_file_found"] is True

    def test_ok_after_training(self, client):
        client.post("/model/train")
        data = client.get("/health").json()
        assert data == {
            "status": "ok", "model_loaded": True, "model_type": "logreg",
            "dataset_loaded": True, "dataset_file_found": True,
        }

    def test_degraded_without_dataset_file(self, client, app_env):
        (app_env / "churn_dataset.csv").unlink()
        data = client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["dataset_file_found"] is False

    def test_docs_available(self, client):
        assert client.get("/docs").status_code == 200


class TestLogging:
    def test_key_events_are_logged(self, client, good_client, caplog):
        caplog.set_level("INFO", logger="churn")
        client.post("/model/train")
        client.post("/predict", json=good_client)
        text = caplog.text
        assert "Датасет загружен" in text
        assert "Обучение запущено" in text
        assert "Обучение завершено" in text
        assert "predict: клиентов=1" in text

    def test_predict_log_has_no_client_data(self, client, good_client, caplog):
        caplog.set_level("INFO", logger="churn")
        client.post("/model/train")
        client.post("/predict", json=good_client)
        assert "america" not in caplog.text
        assert "usage_hours" not in caplog.text

    def test_errors_are_logged(self, client, good_client, caplog):
        caplog.set_level("INFO", logger="churn")
        client.post("/predict", json=good_client)
        assert "MODEL_NOT_TRAINED" in caplog.text
        client.get("/nope")
        assert "NOT_FOUND" in caplog.text

    def test_unexpected_error_logged_with_traceback(self, app_env, monkeypatch, caplog):
        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(churn, "prepare_data", boom)
        with TestClient(churn.app, raise_server_exceptions=False) as client:
            client.post("/model/train")
        assert "INTERNAL_ERROR" in caplog.text
        assert "secret internal detail" in caplog.text
