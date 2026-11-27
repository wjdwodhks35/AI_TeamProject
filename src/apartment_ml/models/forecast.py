"""향후 N개월 가격 예측 (기능 E).

예측 대상을 하나로 줄인다
-------------------------
회귀는 절대가격이 아니라 **시세 대비 상대가격**을 학습한다.

    log(가격 / 면적) = 모델출력 + 시장지수(자치구, 월)

모델출력은 "이 아파트가 그 시점 시세 대비 어느 수준인가" 라서 **시점과 무관**하다.
시점은 전부 시장지수가 들고 있다. 그래서 미래 가격 예측은 **시장지수 예측 하나**로
줄어든다. 25개 자치구의 월별 시계열만 내다보면, 자치구 지수도 특정 단지 가격도
같이 나온다.

어떻게 내다보는가
-----------------
추세를 연장하면 **안 된다.** rolling-origin 백테스트(기준시점 43개 × 자치구 25개,
오차는 로그 MAE 를 퍼센트로 환산):

    방법                    1개월   3개월   6개월  12개월
    직전값 그대로            4.7%    6.0%    7.4%    8.7%
    최근 1년 기울기 연장      5.0%    6.8%    8.9%   11.8%   <- 아무것도 안 하느니만 못하다
    자치구 수준 + 서울 감쇠추세  4.6%    5.6%    6.4%    8.3%   <- 채택

월간 변화의 자기상관이 +0.006 이라 지난달 움직임에 다음 달 정보가 없다. 월 평균
상승 +0.29% 에 표준편차 3.04% 로 잡음이 신호의 10배다. 그래서 "현재 수준을 유지하되
아주 약한 추세만 감쇠시키며 반영" 하는 쪽이 가장 낫다.

추세는 **자치구별로 재지 않고 서울 전체에서 잰다.** 거래가 적은 자치구(종로·중구는
월간 변화 표준편차가 노원구의 4배)의 기울기는 시세가 아니라 '그 달에 어떤 단지가
거래됐나' 를 재게 된다.

구간을 반드시 같이 준다
-----------------------
12개월 뒤 오차가 8.3% 다. 33억짜리면 ±2.7억이고, 여기에 현재 시점 모델 오차가 더
얹힌다. 숫자 하나만 내놓으면 거짓말이 된다. 구간은 백테스트에서 잰 지평별 산포와
회귀의 로그 잔차 산포를 합쳐서 만든다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.models.market_index import MarketIndex, MarketIndexError

if TYPE_CHECKING:  # pragma: no cover
    from apartment_ml.models.regression import PriceRegressionModel

MAX_HORIZON = 24
# 예측구간의 폭. 1.2816 은 정규분포에서 80% 구간에 해당한다. 좁게 잡아 과신하게
# 만드는 것보다 낫고, 95%(1.96)로 넓히면 쓸모가 없을 만큼 벌어진다.
BAND_Z = 1.2816


class ForecastError(RuntimeError):
    """시장지수를 내다볼 수 없을 때."""


@dataclass(frozen=True)
class ForecastPoint:
    """한 달치 예측. 가격은 원 단위."""

    month: str
    horizon: int
    price_krw: float
    low_krw: float
    high_krw: float


class MarketIndexForecaster:
    """시장지수를 미래로 연장한다.

    ``extend(n)`` 은 n개월치를 채운 :class:`MarketIndex` 를 돌려준다. 회귀 모델의
    지수를 그것으로 바꿔 끼우면(``predict_with_market``) 그대로 미래 가격이 된다.
    """

    def __init__(
        self,
        *,
        drift_window: int = 24,
        damping: float = 0.8,
        level_window: int = 3,
        min_history: int = 24,
        lag_months: int = 1,
        min_deals: int = 10,
    ) -> None:
        if not 0 < damping <= 1:
            raise ValueError(f"damping 은 0 초과 1 이하여야 합니다: {damping!r}")
        if drift_window < 2 or level_window < 1:
            raise ValueError("drift_window 는 2 이상, level_window 는 1 이상이어야 합니다.")
        self.drift_window = drift_window
        self.damping = damping
        self.level_window = level_window
        self.min_history = min_history
        self.lag_months = lag_months
        self.min_deals = min_deals

    # -- 학습 ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> MarketIndexForecaster:
        """거래 프레임에서 지수를 만들고 내다볼 준비를 한다."""
        return self._use(MarketIndex(lag_months=self.lag_months, min_deals=self.min_deals).fit(df))

    @classmethod
    def from_index(cls, market: MarketIndex, **kwargs) -> MarketIndexForecaster:
        """이미 학습된 시장지수에서 만든다.

        저장된 회귀 모델 안에 지수가 들어 있으므로, API 는 원본 CSV 없이도 이걸로
        예측기를 만들 수 있다. 모델과 지수가 어긋날 일이 없다는 장점도 있다.
        """
        if not hasattr(market, "seoul_"):
            raise ForecastError("fit 되지 않은 시장지수로는 만들 수 없습니다.")
        return cls(lag_months=market.lag_months, min_deals=market.min_deals, **kwargs)._use(market)

    def _use(self, market: MarketIndex) -> MarketIndexForecaster:
        self.base_ = market
        # 자치구 × 월 표. 지수와 같은 (늦춘) 달 기준이라 변환이 따로 필요 없다.
        self.table_ = market.gu_.unstack(level="district").sort_index()
        self.seoul_ = market.seoul_.sort_index()
        if len(self.seoul_) < self.min_history:
            raise ForecastError(
                f"시장지수를 내다보려면 {self.min_history}개월 이상이 필요합니다 "
                f"(현재 {len(self.seoul_)}개월)."
            )
        self.horizon_sd_ = self._backtest()
        return self

    def _slope(self, series: np.ndarray) -> float:
        window = min(self.drift_window, len(series) - 1)
        return float((series[-1] - series[-(window + 1)]) / window)

    def _path(self, level_series: np.ndarray, slope: float, horizon: int) -> np.ndarray:
        level = float(level_series[-min(self.level_window, len(level_series)) :].mean())
        steps = np.cumsum(self.damping ** np.arange(1, horizon + 1))
        return level + slope * steps

    def _backtest(self) -> dict[int, float]:
        """rolling-origin 으로 지평별 로그오차 산포를 잰다. 구간 폭의 근거가 된다."""
        errors: dict[int, list[float]] = {h: [] for h in range(1, MAX_HORIZON + 1)}
        months = self.table_.index
        for t in range(self.min_history, len(months) - 1):
            seoul = self.seoul_.iloc[: t + 1].dropna().to_numpy()
            if len(seoul) <= self.drift_window:
                continue
            slope = self._slope(seoul)
            for district in self.table_.columns:
                past = self.table_[district].iloc[: t + 1].dropna().to_numpy()
                if len(past) < self.level_window:
                    continue
                future = self.table_[district].iloc[t + 1 : t + 1 + MAX_HORIZON].to_numpy()
                path = self._path(past, slope, len(future))
                for h, (predicted, actual) in enumerate(zip(path, future, strict=True), start=1):
                    if np.isfinite(actual):
                        errors[h].append(predicted - actual)
        out: dict[int, float] = {}
        previous = 0.0
        for h in range(1, MAX_HORIZON + 1):
            values = errors[h]
            # 표본이 없으면 직전 지평 값을 쓴다 (지평이 길수록 표본이 줄어든다).
            previous = float(np.sqrt(np.mean(np.square(values)))) if values else previous
            out[h] = previous
        return out

    # -- 사용 ---------------------------------------------------------------
    @property
    def last_month(self) -> pd.Period:
        return self.base_.last_month_

    def horizon_sd(self, horizon: int) -> float:
        """그 지평의 로그오차 산포. 범위를 벗어나면 가장 가까운 값을 쓴다."""
        if not hasattr(self, "horizon_sd_"):
            raise ForecastError("MarketIndexForecaster 가 fit 되지 않았습니다.")
        return self.horizon_sd_[min(max(horizon, 1), MAX_HORIZON)]

    def extend(self, months: int) -> MarketIndex:
        """미래 ``months`` 개월을 채운 시장지수를 새로 만들어 돌려준다."""
        if not hasattr(self, "base_"):
            raise ForecastError("MarketIndexForecaster 가 fit 되지 않았습니다.")
        if months < 1:
            raise ValueError(f"months 는 1 이상이어야 합니다: {months!r}")

        slope = self._slope(self.seoul_.dropna().to_numpy())
        future_months = [self.last_month + i for i in range(1, months + 1)]

        seoul_path = self._path(self.seoul_.dropna().to_numpy(), slope, months)
        seoul_rows = pd.Series(seoul_path, index=pd.Index(future_months, name="ym"))

        keys: list[tuple[pd.Period, str]] = []
        values: list[float] = []
        for district in self.table_.columns:
            past = self.table_[district].dropna().to_numpy()
            if len(past) < self.level_window:
                continue
            path = self._path(past, slope, months)
            keys.extend((month, district) for month in future_months)
            values.extend(path.tolist())

        extended = copy.copy(self.base_)
        extended.seoul_ = pd.concat([self.base_.seoul_, seoul_rows])
        extended.gu_ = pd.concat(
            [
                self.base_.gu_,
                pd.Series(
                    values,
                    index=pd.MultiIndex.from_tuples(keys, names=["ym", "district"]),
                ),
            ]
        )
        extended.last_month_ = future_months[-1]
        extended.last_seoul_ = float(seoul_rows.iloc[-1])
        last = extended.gu_[extended.gu_.index.get_level_values("ym") == extended.last_month_]
        extended.last_gu_ = last.set_axis(last.index.get_level_values("district"))
        return extended


def forecast_prices(
    model: PriceRegressionModel,
    features: pd.DataFrame,
    forecaster: MarketIndexForecaster,
    months: int = 12,
    *,
    day: int = 15,
) -> list[ForecastPoint]:
    """한 건의 거래 조건을 미래 ``months`` 개월에 대해 예측한다.

    ``features`` 는 1행이어야 한다. 계약일은 각 미래 달의 ``day`` 일로 바꿔 넣는다.
    """
    if len(features) != 1:
        raise ValueError(f"거래 조건 1행만 받습니다 (받은 행 수 {len(features)}).")
    if months < 1 or months > MAX_HORIZON:
        raise ValueError(f"months 는 1 ~ {MAX_HORIZON} 사이여야 합니다: {months!r}")

    extended = forecaster.extend(months)
    target_months = [forecaster.last_month + i for i in range(1, months + 1)]

    frame = pd.concat([features] * months, ignore_index=True)
    frame[S.CONTRACT_DATE] = [
        month.to_timestamp() + pd.Timedelta(days=day - 1) for month in target_months
    ]
    frame = _refresh_date_parts(frame)

    try:
        prices = model.predict_with_market(frame, extended)
    except MarketIndexError as exc:  # pragma: no cover - 입력 검증은 상위에서 한다
        raise ForecastError(str(exc)) from exc

    base_sd = float(getattr(model, "log_residual_sd_", 0.0))
    points: list[ForecastPoint] = []
    for horizon, (month, price) in enumerate(zip(target_months, prices, strict=True), start=1):
        # 두 오차는 성격이 다르다 (하나는 단면, 하나는 시간) 이므로 제곱합으로 더한다.
        sd = float(np.hypot(base_sd, forecaster.horizon_sd(horizon)))
        spread = float(np.exp(BAND_Z * sd))
        value = float(max(price, 1.0))
        points.append(
            ForecastPoint(
                month=str(month),
                horizon=horizon,
                price_krw=value,
                low_krw=value / spread,
                high_krw=value * spread,
            )
        )
    return points


def _refresh_date_parts(frame: pd.DataFrame) -> pd.DataFrame:
    """계약일을 바꿨으면 거기서 파생된 칸도 같이 맞춘다."""
    dates = pd.to_datetime(frame[S.CONTRACT_DATE])
    if S.CONTRACT_YEAR in frame.columns:
        frame[S.CONTRACT_YEAR] = dates.dt.year
    if S.CONTRACT_MONTH in frame.columns:
        frame[S.CONTRACT_MONTH] = dates.dt.month
    if S.BUILDING_AGE in frame.columns and S.BUILT_YEAR in frame.columns:
        built = pd.to_numeric(frame[S.BUILT_YEAR], errors="coerce")
        frame[S.BUILDING_AGE] = (dates.dt.year - built).clip(lower=0)
    return frame
