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


# 두 값만 비교하면 매끄러운 반응으로 오해한다. 실제 금리 반응은 분기 몇 개가 만든
# 계단이고, 학습 범위 밖은 경계와 똑같은 답이 나온다. 그래서 곡선을 같이 준다.


def test_curve_covers_observed_range():
    result = compare_interest_rate_scenario(FakePriceModel(), FEATURES, 1.0, 3.0)
    rates = [p["base_rate"] for p in result.curve]
    assert result.observed_range == [0.5, 3.5]
    assert result.extrapolated is False
    assert rates == sorted(rates)
    assert rates[0] == 0.5 and rates[-1] == 3.5
    assert len(rates) >= 13  # 0.25%p 간격


def test_curve_flags_extrapolation_and_stays_inside():
    result = compare_interest_rate_scenario(FakePriceModel(), FEATURES, 2.5, 5.0)
    assert result.extrapolated is True
    assert any("구분하지 못해서" in w for w in result.warnings)
    # 범위 밖 값은 곡선에 넣지 않는다 — 모델이 경계와 구분하지 못하므로
    assert max(p["base_rate"] for p in result.curve) == 3.5
