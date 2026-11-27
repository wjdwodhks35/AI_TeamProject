"""시장지수 예측과 미래 가격 예측."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.models.forecast import (
    MAX_HORIZON,
    ForecastError,
    MarketIndexForecaster,
    forecast_prices,
)
from apartment_ml.models.market_index import MarketIndex


def history(months: int = 36, monthly_growth: float = 0.004) -> pd.DataFrame:
    """자치구 2곳 × months 개월, 매달 일정하게 오르는 시장."""
    rows = []
    start = pd.Timestamp("2022-01-15")
    for m in range(months):
        date = start + pd.DateOffset(months=m)
        for district, level in (("강남구", 20_000_000), ("노원구", 8_000_000)):
            for i in range(12):
                area = 60.0 + i
                ppm = level * np.exp(monthly_growth * m)
                rows.append(
                    {
                        S.CONTRACT_DATE: date,
                        S.DISTRICT: district,
                        S.AREA_M2: area,
                        S.REGRESSION_TARGET: ppm * area,
                    }
                )
    return pd.DataFrame(rows)


def test_extend_continues_the_series():
    """연장한 지수는 미래 달에도 값을 주고, 오르는 시장이면 계속 오른다."""
    f = MarketIndexForecaster().fit(history())
    extended = f.extend(6)

    future = [f.last_month + i for i in range(1, 7)]
    values = [extended.gu_[(m, "강남구")] for m in future]
    assert all(np.isfinite(values))
    assert values == sorted(values)  # 감쇠는 하지만 방향은 유지한다

    # 연장 전에는 미래 달이 마지막 값으로 고정되지만, 연장하면 달라진다
    frame = pd.DataFrame(
        {
            S.CONTRACT_DATE: [future[-1].to_timestamp()],
            S.DISTRICT: ["강남구"],
            S.AREA_M2: [84.0],
        }
    )
    assert extended.transform(frame)[0] > f.base_.transform(frame)[0]


def test_horizon_spread_grows():
    """멀리 볼수록 구간이 넓어져야 한다. 안 그러면 구간이 거짓말이 된다."""
    f = MarketIndexForecaster().fit(history())
    assert f.horizon_sd(1) <= f.horizon_sd(6) <= f.horizon_sd(12)
    assert f.horizon_sd(MAX_HORIZON + 99) == f.horizon_sd(MAX_HORIZON)


def test_forecast_prices_shape_and_band():
    class FakeModel:
        name = "fake"
        market_ = None
        log_residual_sd_ = 0.1

        def predict_with_market(self, frame, market):
            # 지수를 그대로 가격처럼 쓴다. 바꿔 끼운 지수가 쓰이는지 확인하는 용도.
            return np.exp(market.transform(frame)) * 1000.0

    f = MarketIndexForecaster().fit(history())
    row = pd.DataFrame(
        {
            S.CONTRACT_DATE: [pd.Timestamp("2025-01-15")],
            S.DISTRICT: ["강남구"],
            S.AREA_M2: [84.0],
        }
    )
    points = forecast_prices(FakeModel(), row, f, months=12)

    assert len(points) == 12
    assert [p.horizon for p in points] == list(range(1, 13))
    assert all(p.low_krw < p.price_krw < p.high_krw for p in points)
    # 지평이 길수록 구간이 넓어진다
    width = [(p.high_krw / p.price_krw) for p in points]
    assert width[-1] > width[0]

    with pytest.raises(ValueError, match="1행"):
        forecast_prices(FakeModel(), pd.concat([row, row]), f, months=3)
    with pytest.raises(ValueError, match="months"):
        forecast_prices(FakeModel(), row, f, months=0)


def test_rejects_short_history():
    with pytest.raises(ForecastError, match="개월 이상"):
        MarketIndexForecaster().fit(history(months=6))
    with pytest.raises(ForecastError, match="fit"):
        MarketIndexForecaster.from_index(MarketIndex())
