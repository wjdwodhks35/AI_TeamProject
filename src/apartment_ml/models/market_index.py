"""월별 시장지수 — 직전월 자치구 평균 로그 평당가. 회귀에서 시점 흐름을 처리한다.

왜 필요한가: 트리 계열 모델은 학습 때 본 범위 밖으로 **외삽하지 못한다**. 시간순
분할에서 test 구간(2026년)은 학습 구간(~2025-07)보다 가격이 높은데, ``contract_year``
를 넣어도, 직전월 지수를 설명변수로 넣어도 트리는 마지막 분기점에서 멈춘다.

측정 결과 (학습 180,279건 / test 38,710건):

- 절대가격을 맞히면 test 편향 -1.69억, 그리고 그 편향이 2026-02 -1.26억에서
  2026-08 -2.01억으로 **시간이 갈수록 커졌다**.
- 직전월 지수를 설명변수로 추가해도 -1.66억 (거의 그대로).
- 지수 대비 상대가격을 맞히게 바꾸니 편향 -0.67억, 월별 추세도 사라졌다.
  test MAE 2.07억 -> 1.41억, R2 0.860 -> 0.934.

그래서 시간 흐름은 학습에 맡기지 않고 이 지수로 직접 처리하고, 모델은 '지수 대비
몇 배인가'만 맞힌다.

시점 규칙: 어떤 달의 지수는 **직전월** 거래로 만든다. 계약일 기준으로 이미 신고가
끝난 과거 거래만 쓰는 것이므로, 예측 시점에 알 수 없는 정보는 들어가지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apartment_ml import schemas as S


class MarketIndexError(ValueError):
    """지수를 만들 수 없을 때 (필요한 컬럼이 없거나 유효한 거래가 없을 때)."""


class MarketIndex:
    """자치구별 직전월 로그 평당가 평균. 없으면 서울 전체 → 마지막 알려진 값으로 대체.

    ``min_deals`` 보다 거래가 적은 자치구-월은 값이 흔들리므로 서울 전체 평균을 쓴다.
    """

    def __init__(self, lag_months: int = 1, min_deals: int = 10) -> None:
        if lag_months < 1:
            raise ValueError("lag_months 는 1 이상이어야 합니다 (같은 달 거래는 쓸 수 없습니다).")
        self.lag_months = lag_months
        self.min_deals = min_deals

    # -- 내부 ---------------------------------------------------------------
    @staticmethod
    def _prepare(df: pd.DataFrame, *, need_price: bool) -> pd.DataFrame:
        required = [S.CONTRACT_DATE, S.DISTRICT, S.AREA_M2]
        if need_price:
            required.append(S.REGRESSION_TARGET)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise MarketIndexError(f"시장지수에 필요한 컬럼이 없습니다: {missing}")
        out = pd.DataFrame(
            {
                "ym": pd.to_datetime(df[S.CONTRACT_DATE]).dt.to_period("M"),
                "district": df[S.DISTRICT].astype(object),
            },
            index=df.index,
        )
        if need_price:
            area = pd.to_numeric(df[S.AREA_M2], errors="coerce")
            price = pd.to_numeric(df[S.REGRESSION_TARGET], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                out["lppm"] = np.log(price / area)
            out = out[np.isfinite(out["lppm"]) & out["ym"].notna()]
        return out

    # -- 계약 ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> MarketIndex:
        """가격이 있는 거래 프레임에서 지수를 만든다 (보통 학습 구간)."""
        rows = self._prepare(df, need_price=True)
        if rows.empty:
            raise MarketIndexError("시장지수를 만들 유효한 거래가 없습니다.")
        seoul = rows.groupby("ym")["lppm"].mean()
        gu = rows.groupby(["ym", "district"])["lppm"].agg(["mean", "count"])
        gu = gu.loc[gu["count"] >= self.min_deals, "mean"]
        # 직전월 값을 그 달의 지수로 쓴다.
        self.seoul_ = seoul.set_axis(seoul.index + self.lag_months)
        self.gu_ = gu.set_axis(
            pd.MultiIndex.from_arrays(
                [
                    gu.index.get_level_values("ym") + self.lag_months,
                    gu.index.get_level_values("district"),
                ],
                names=["ym", "district"],
            )
        )
        self.last_month_ = self.seoul_.index.max()
        self.last_seoul_ = float(self.seoul_.loc[self.last_month_])
        last = self.gu_[self.gu_.index.get_level_values("ym") == self.last_month_]
        self.last_gu_ = last.set_axis(last.index.get_level_values("district"))
        self.global_ = float(rows["lppm"].mean())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """각 행의 지수값. 자치구 → 서울 전체 → 마지막 알려진 값 순으로 채운다."""
        if not hasattr(self, "seoul_"):
            raise MarketIndexError("MarketIndex 가 fit 되지 않았습니다.")
        rows = self._prepare(df, need_price=False)
        ym = rows["ym"]
        value = self.gu_.reindex(
            pd.MultiIndex.from_arrays([ym, rows["district"]], names=["ym", "district"])
        ).to_numpy("float64")
        seoul = self.seoul_.reindex(ym).to_numpy("float64")
        # 지수를 만든 마지막 달 이후(= 미래)는 마지막 값으로 고정한다.
        beyond = (ym > self.last_month_).to_numpy()
        last_gu = self.last_gu_.reindex(rows["district"]).to_numpy("float64")
        value = np.where(beyond & np.isfinite(last_gu), last_gu, value)
        seoul = np.where(beyond, self.last_seoul_, seoul)
        value = np.where(np.isfinite(value), value, seoul)
        return np.where(np.isfinite(value), value, self.global_)
