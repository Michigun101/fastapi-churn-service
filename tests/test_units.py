import numpy as np
import pandas as pd
import pytest

import fastapi_churn_day13 as churn
from fastapi_churn_day13 import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    FeatureVectorChurn,
    TrainingConfigChurn,
)


@pytest.fixture
def prepared(df):
    X, y, _, _ = churn.prepare_data(df)
    return churn.split_data(X, y)


class TestPrepareData:
    def test_separates_features_and_target(self, df):
        X, y, numeric, categorical = churn.prepare_data(df)
        assert list(X.columns) == FEATURE_COLUMNS
        assert "churn" not in X.columns
        assert len(X) == len(y) == len(df)
        assert numeric == NUMERIC_FEATURES and categorical == CATEGORICAL_FEATURES

    def test_fills_missing_values(self, df):
        df = df.copy()
        df.loc[0, "monthly_fee"] = np.nan
        df.loc[1, "region"] = np.nan
        X, _, _, _ = churn.prepare_data(df)
        assert X.isna().sum().sum() == 0
        assert X.loc[0, "monthly_fee"] == df["monthly_fee"].median()
        assert X.loc[1, "region"] == df["region"].mode().iloc[0]

    def test_drops_rows_without_target(self, df):
        df = df.copy()
        df.loc[:4, "churn"] = np.nan
        X, y, _, _ = churn.prepare_data(df)
        assert len(X) == len(y) == len(df) - 5

    def test_feature_lists_match_request_model(self):
        assert set(FEATURE_COLUMNS) == set(FeatureVectorChurn.model_fields)


class TestSplit:
    def test_sizes(self, prepared, df):
        X_train, X_test, _, _ = prepared
        assert len(X_test) == int(len(df) * churn.TEST_SIZE)
        assert len(X_train) + len(X_test) == len(df)

    def test_stratified(self, prepared, df):
        _, _, y_train, y_test = prepared
        overall = df["churn"].mean()
        assert abs(y_train.mean() - overall) < 0.02
        assert abs(y_test.mean() - overall) < 0.02

    def test_reproducible(self, df):
        X, y, _, _ = churn.prepare_data(df)
        a = churn.split_data(X, y)[1].index
        b = churn.split_data(X, y)[1].index
        assert list(a) == list(b)

    def test_class_distribution(self):
        dist = churn.class_distribution(pd.Series([0, 0, 0, 1]))
        assert dist == {"0": {"count": 3, "share": 0.75}, "1": {"count": 1, "share": 0.25}}


class TestBuildClassifier:
    def test_logreg_with_hyperparameters(self):
        clf = churn.build_classifier(TrainingConfigChurn(hyperparameters={"C": 0.5}))
        assert clf.__class__.__name__ == "LogisticRegression"
        assert clf.C == 0.5

    def test_random_forest(self):
        cfg = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 10})
        clf = churn.build_classifier(cfg)
        assert clf.__class__.__name__ == "RandomForestClassifier"
        assert clf.n_estimators == 10

    def test_unknown_hyperparameter_rejected(self):
        with pytest.raises(ValueError, match="неизвестные гиперпараметры"):
            churn.build_classifier(TrainingConfigChurn(hyperparameters={"foo": 1}))

    @pytest.mark.parametrize("value", [0, 5000, 2.5])
    def test_n_estimators_bounds(self, value):
        cfg = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": value})
        with pytest.raises(ValueError, match="n_estimators"):
            churn.build_classifier(cfg)


class TestTraining:
    def test_pipeline_structure(self, prepared):
        X_train, _, y_train, _ = prepared
        model = churn.train_churn_model(X_train, y_train)
        assert list(model.named_steps) == ["preprocessor", "classifier"]
        assert list(model.feature_names_in_) == FEATURE_COLUMNS

    @pytest.mark.parametrize("model_type", ["logreg", "random_forest"])
    def test_predictions(self, prepared, model_type):
        X_train, X_test, y_train, _ = prepared
        cfg = TrainingConfigChurn(model_type=model_type, hyperparameters={})
        model = churn.train_churn_model(X_train, y_train, cfg)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1)
        assert set(model.predict(X_test)) <= {0, 1}

    def test_unseen_category_does_not_fail(self, prepared):
        X_train, X_test, y_train, _ = prepared
        model = churn.train_churn_model(X_train, y_train)
        row = X_test.iloc[[0]].copy()
        row["region"] = "mars"
        assert model.predict_proba(row).shape == (1, 2)

    def test_deterministic(self, prepared):
        X_train, X_test, y_train, _ = prepared
        a = churn.train_churn_model(X_train, y_train).predict_proba(X_test)
        b = churn.train_churn_model(X_train, y_train).predict_proba(X_test)
        assert np.allclose(a, b)

    def test_effective_hyperparameters(self, prepared):
        X_train, _, y_train, _ = prepared
        cfg = TrainingConfigChurn(hyperparameters={"C": 0.3})
        model = churn.train_churn_model(X_train, y_train, cfg)
        assert churn.effective_hyperparameters(model, "logreg")["C"] == 0.3


class TestPersistence:
    def test_model_roundtrip_is_single_object(self, prepared, tmp_path):
        X_train, X_test, y_train, _ = prepared
        model = churn.train_churn_model(X_train, y_train)
        path = tmp_path / "m.joblib"
        churn.save_churn_model(model, {"model_type": "logreg"}, path)
        loaded, info = churn.load_churn_model(path)
        assert info == {"model_type": "logreg"}
        assert np.allclose(model.predict_proba(X_test), loaded.predict_proba(X_test))

    def test_load_missing_file(self, tmp_path):
        assert churn.load_churn_model(tmp_path / "nope.joblib") is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.joblib"
        path.write_text("not a model")
        assert churn.load_churn_model(path) is None

    def test_history_roundtrip(self, tmp_path):
        path = tmp_path / "h.json"
        churn.save_history([{"id": 1}], path)
        assert churn.load_history(path) == [{"id": 1}]

    def test_history_missing_or_corrupt(self, tmp_path):
        assert churn.load_history(tmp_path / "none.json") == []
        bad = tmp_path / "bad.json"
        bad.write_text("{oops")
        assert churn.load_history(bad) == []


class TestFeaturesToFrame:
    def test_order_follows_model(self, prepared, good_client):
        X_train, _, y_train, _ = prepared
        model = churn.train_churn_model(X_train, y_train)
        shuffled = dict(reversed(list(good_client.items())))
        frame = churn.features_to_frame([FeatureVectorChurn(**shuffled)], model)
        assert list(frame.columns) == list(model.feature_names_in_)


class TestBestRecord:
    def test_picks_max_and_skips_none(self):
        records = [
            {"metrics": {"f1": 0.1, "roc_auc": None}},
            {"metrics": {"f1": 0.4, "roc_auc": 0.6}},
        ]
        assert churn.best_record(records, "f1") is records[1]
        assert churn.best_record(records, "roc_auc") is records[1]
        assert churn.best_record([], "f1") is None
