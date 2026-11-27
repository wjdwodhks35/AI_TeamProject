"""기준금리 변화량 결합 테스트."""

import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.data.merge import attach_base_rate, attach_base_rate_changes

RATES = pd.DataFrame(
    {
        "effective_date": ["2021-01-01", "2021-08-26", "2022-01-14", "2023-01-13"],
        "base_rate": [0.50, 0.75, 1.25, 3.50],
    }
)


def deals(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"district": ["강남구"] * len(dates), "contract_date": dates})


def test_change_12m_uses_rate_a_year_earlier():
    """2022-09 계약: 당시 1.25%, 1년 전(2021-09)은 0.75% → +0.50%p."""
    out = attach_base_rate_changes(deals(["2022-09-15"]), RATES, months=(12,))
    assert out[S.BASE_RATE].iloc[0] == pytest.approx(1.25)
    assert out[S.BASE_RATE_CHANGE_12M].iloc[0] == pytest.approx(0.50)


def test_change_is_negative_when_rate_fell():
    """금리가 내려간 구간에서는 변화량이 음수가 된다."""
    rates = pd.DataFrame(
        {"effective_date": ["2024-01-01", "2025-05-29"], "base_rate": [3.50, 2.50]}
    )
    out = attach_base_rate_changes(deals(["2025-07-01"]), rates, months=(12,))
    assert out[S.BASE_RATE_CHANGE_12M].iloc[0] == pytest.approx(-1.00)


def test_change_zero_during_freeze():
    """금리 동결 구간에서는 변화량이 0이다 (2023-01 ~ 2024-09 에 3.50% 로 묶여 있었다)."""
    out = attach_base_rate_changes(deals(["2024-06-01"]), RATES, months=(12,))
    assert out[S.BASE_RATE_CHANGE_12M].iloc[0] == pytest.approx(0.0)


def test_multiple_windows():
    out = attach_base_rate_changes(deals(["2022-09-15"]), RATES, months=(12, 3))
    assert S.BASE_RATE_CHANGE_12M in out.columns
    assert S.BASE_RATE_CHANGE_3M in out.columns
    assert out[S.BASE_RATE_CHANGE_3M].iloc[0] == pytest.approx(0.0)  # 6월에도 1.25%


def test_before_first_rate_is_nan():
    out = attach_base_rate_changes(deals(["2021-03-01"]), RATES, months=(12,))
    assert pd.isna(out[S.BASE_RATE_CHANGE_12M].iloc[0])


def test_row_order_and_count_preserved():
    dates = ["2023-05-01", "2021-09-01", "2022-09-15"]
    out = attach_base_rate_changes(deals(dates), RATES)
    assert out["contract_date"].tolist() == dates
    assert len(out) == len(dates)


def test_reuses_existing_base_rate():
    """base_rate 가 이미 붙어 있으면 다시 붙이지 않는다."""
    first = attach_base_rate(deals(["2022-09-15"]), RATES)
    out = attach_base_rate_changes(first, RATES, months=(12,))
    assert out[S.BASE_RATE].iloc[0] == pytest.approx(1.25)
    assert out[S.BASE_RATE_CHANGE_12M].iloc[0] == pytest.approx(0.50)


def test_missing_rate_column_raises():
    with pytest.raises(ValueError, match="기준금리 데이터에 컬럼이 없습니다"):
        attach_base_rate_changes(deals(["2022-09-15"]), RATES.drop(columns=["base_rate"]))


def test_change_columns_are_regression_features():
    """변화량은 회귀 입력으로 쓸 수 있고, 금지 목록에는 없어야 한다."""
    for column in S.BASE_RATE_CHANGE_COLUMNS:
        assert column in S.REGRESSION_NUMERIC_FEATURES
        assert column not in S.REGRESSION_FORBIDDEN_FEATURES
