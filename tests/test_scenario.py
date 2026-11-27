"""테스트 전용 모델로 시나리오 계산 및 입력 격리를 검증한다."""

import numpy as np
import pytest

from apartment_ml.scenarios.interest_rate import compare_interest_rate_scenario
from tests.fakes import FakePriceModel
from tests.test_api import FEATURES


def test_only_rate_changes():
    class RecordingModel(FakePriceModel):
        def predict(self, X):
            assert X.loc[0].drop("base_rate").equals(X.loc[1].drop("base_rate"))
            assert "price_krw" not in X
            return super().predict(X)

    original = FEATURES.copy()
    result = compare_interest_rate_scenario(RecordingModel(), FEATURES, 3.5, 2.5)
    assert original == FEATURES
    assert result.price_diff_krw == pytest.approx(10_000_000)
    assert result.price_change_pct == pytest.approx(
        result.price_diff_krw / result.predicted_price_before_krw * 100
    )
    assert "인과효과" in result.disclaimer


@pytest.mark.parametrize("before,expected", [(0.0, None), (-100.0, -150.0)])
def test_zero_and_negative_baseline(before, expected):
    class FixedModel(FakePriceModel):
        def predict(self, X):
            return np.array([before, 50.0])

    result = compare_interest_rate_scenario(FixedModel(), FEATURES, 3, 2)
    assert result.price_change_pct == expected
    assert result.warnings


@pytest.mark.parametrize("values", [[float("nan"), 1], [1, float("inf")], [1]])
def test_invalid_model_output(values):
    class InvalidModel(FakePriceModel):
        def predict(self, X):
            return np.array(values)

    with pytest.raises(RuntimeError):
        compare_interest_rate_scenario(InvalidModel(), FEATURES, 3, 2)


def test_same_rates():
    result = compare_interest_rate_scenario(FakePriceModel(), FEATURES, 3, 3)
    assert result.price_diff_krw == 0
    assert result.price_change_pct == 0
