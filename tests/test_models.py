"""모델 인터페이스 계약 테스트 (합성 데이터 — 성능 검증 아님)."""

import numpy as np
import pandas as pd
import pytest

from apartment_ml.data.merge import attach_base_rate
from apartment_ml.data.preprocess import add_price_features, preprocess_transactions
from apartment_ml.models.base import (
    LabelsNotAvailableError,
    ModelNotReadyError,
    TrainingDisabledError,
)
from apartment_ml.models.classification import AnomalyClassificationModel, validate_anomaly_labels
from apartment_ml.models.ensemble import build_voting_classifier, build_voting_model
from apartment_ml.models.regression import PriceRegressionModel
from apartment_ml.scenarios.interest_rate import compare_interest_rate_scenario
from tests.conftest import make_rows
from tests.fakes import FakePriceModel


@pytest.fixture
def train_df():
    df = preprocess_transactions(make_rows(40)).data
    rates = pd.DataFrame({"effective_date": ["2022-01-01", "2023-01-20"], "base_rate": [3.0, 3.5]})
    return attach_base_rate(df, rates)


def test_regression_forbids_target_as_feature():
    with pytest.raises(ValueError, match="price_krw"):
        PriceRegressionModel(numeric_features=["area_m2", "price_krw"])


@pytest.mark.parametrize("name", ["linear_regression", "gradient_boosting"])
def test_regression_fit_predict_save_load(train_df, tmp_path, name):
    model = PriceRegressionModel(name)
    with pytest.raises(ModelNotReadyError):
        model.predict(train_df)
    with pytest.raises(ModelNotReadyError):
        model.save(tmp_path / "m.joblib")
    model.fit(train_df)
    pred = model.predict(train_df)
    assert pred.shape == (len(train_df),)
    metrics = model.evaluate(train_df)
    assert set(metrics) == {"mae", "rmse", "r2", "mape", "n"}
    assert "base_rate" in model.feature_ranges
    path = model.save(tmp_path / "m.joblib")
    loaded = PriceRegressionModel.load(path)
    np.testing.assert_allclose(loaded.predict(train_df), pred)


def test_regression_bias_correction(train_df):
    """보정계수는 예측에 곱해지고, 다시 학습하면 무효가 된다."""
    model = PriceRegressionModel("gradient_boosting").fit(train_df)
    raw = model.predict(train_df)
    assert model.log_bias_factor_ == 1.0

    factor = model.calibrate(train_df)
    assert factor > 1.0  # 로그 복원은 항상 낮게 나오므로 보정은 1보다 크다
    np.testing.assert_allclose(model.predict(train_df), raw * factor)
    assert model.metadata["log_bias_factor"] == factor

    model.fit(train_df)
    assert model.log_bias_factor_ == 1.0

    with pytest.raises(ValueError, match="보정계수"):
        model.set_log_bias_factor(0.0)


def test_regression_load_missing_file(tmp_path):
    with pytest.raises(ModelNotReadyError):
        PriceRegressionModel.load(tmp_path / "nope.joblib")


def test_classification_training_disabled_by_default(train_df):
    model = AnomalyClassificationModel("decision_tree")
    df = add_price_features(train_df)
    with pytest.raises(TrainingDisabledError):
        model.fit(df)
    with pytest.raises(LabelsNotAvailableError):  # 레이블 전부 결측
        model.fit(df, training_enabled=True)
    with pytest.raises(ModelNotReadyError):
        model.predict(df)


def test_validate_anomaly_labels():
    with pytest.raises(LabelsNotAvailableError):
        validate_anomaly_labels(pd.Series([None, None]))
    with pytest.raises(LabelsNotAvailableError):
        validate_anomaly_labels(pd.Series([0, 0, 0]))
    with pytest.raises(ValueError):
        validate_anomaly_labels(pd.Series([0, 1, 3]))
    assert validate_anomaly_labels(pd.Series([0, 1])).tolist() == [0, 1]


def test_voting_ensemble_is_built_unfitted():
    vc = build_voting_classifier()
    assert [n for n, _ in vc.estimators] == [
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    ]
    model = build_voting_model()
    assert not model.is_fitted
    with pytest.raises(ValueError):
        build_voting_classifier(["decision_tree"])


def test_scenario_requires_model():
    feats = {"district": "강남구", "contract_date": "2024-01-01", "area_m2": 84.9, "floor": 5}
    with pytest.raises(ModelNotReadyError):
        compare_interest_rate_scenario(None, feats, 3.5, 2.5)
    with pytest.raises(ModelNotReadyError):
        compare_interest_rate_scenario(PriceRegressionModel(), feats, 3.5, 2.5)


def test_scenario_only_rate_changes():
    feats = {"district": "강남구", "contract_date": "2024-01-01", "area_m2": 80, "floor": 5}
    r = compare_interest_rate_scenario(FakePriceModel(), feats, 3.0, 2.0)
    assert r.predicted_price_before_krw == 800_000_000 - 30_000_000
    assert r.predicted_price_after_krw == 800_000_000 - 20_000_000
    assert r.price_diff_krw == 10_000_000
    assert r.price_change_pct == pytest.approx(10_000_000 / 770_000_000 * 100)
    assert r.warnings == []
    assert "인과" in r.disclaimer
    out = compare_interest_rate_scenario(FakePriceModel(), feats, 3.0, 7.0)
    assert any("base_rate" in w for w in out.warnings)  # 학습 범위 밖 경고
    with pytest.raises(ValueError):
        compare_interest_rate_scenario(FakePriceModel(), feats, 3.0, 99.0)
