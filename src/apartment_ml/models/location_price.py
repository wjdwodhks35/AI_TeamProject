"""위치(자치구·동·단지)를 '그 위치의 평균 가격수준' 숫자로 요약하는 인코더 — 회귀 전용.

아파트 가격을 가장 크게 좌우하는 건 단지다. 그런데 단지는 서울에 7,600개가 넘어
원-핫으로 넣으면 변수가 폭발하고, HistGradientBoosting 의 범주형 직접 처리는
255개까지만 지원한다 (동도 332개로 이미 넘는다). 그래서 위치를 범주로 넣지 않고
'그 위치의 정답 평균'이라는 수치형 변수로 바꾼다.

측정 효과 (부스팅, 로그 타깃, test 38,710건):
자치구만 2.63억 -> +동 2.41억 -> +동+단지 2.17억 (R2 0.807 -> 0.866).

``y`` 는 모델이 맞히려는 값을 그대로 받는다. 상대가격(지수 대비 로그 평당가)으로
학습하면 인코딩도 자동으로 면적·시점이 제거된 '그 위치의 상대 가격수준'이 된다.

**분류에는 쓰지 않는다.** 분류 입력에는 ``price_krw`` 가 들어 있어서, 여기에 단지
수준 기준가가 함께 들어가면 ``price_ratio`` 를 역산해 이상거래 레이블 규칙을
그대로 복원할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from apartment_ml import schemas as S


class LocationPriceEncoder(BaseEstimator, TransformerMixin):
    """위치 수준별 정답 평균(+ 그 위치의 학습 거래 수)을 만든다.

    누출을 막는 장치 세 가지:

    - 값은 ``fit`` 에 들어온 데이터에서만 계산한다. 시간순 분할에서 fit 은 학습
      구간이므로, 예측 시점에 이미 알 수 있는 과거 정보만 쓰는 셈이다.
    - 학습 행 자신의 값은 ``n_folds`` 겹 out-of-fold 로 채운다 (``fit_transform``).
      자기 거래가 자기 설명변수에 섞이면 학습 성능만 올라가고 실제 성능은 떨어진다.
    - 거래가 적은 위치는 전체 평균 쪽으로 끌어당기고(``smoothing``), 학습에 없던
      위치는 전체 평균을 쓴다. 거래 수를 함께 내보내므로 모델이 '이 값을 믿을
      만한지' 스스로 판단할 수 있다 (test 단지의 2.7% 가 학습에 없는 단지다).
    """

    def __init__(
        self,
        levels: Sequence[str] = S.REGRESSION_LOCATION_FEATURES,
        smoothing: float = 10.0,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.levels = tuple(levels)
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state

    # -- 내부 ---------------------------------------------------------------
    def _table(self, keys: pd.Series, y: np.ndarray) -> pd.DataFrame:
        grouped = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k", dropna=True)["y"]
        out = grouped.agg(["sum", "count"])
        out["value"] = (out["sum"] + self.prior_ * self.smoothing) / (out["count"] + self.smoothing)
        return out[["value", "count"]]

    def _apply(self, table: pd.DataFrame, keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        hit = table.reindex(pd.Index(keys.to_numpy()))
        return (
            hit["value"].fillna(self.prior_).to_numpy("float64"),
            hit["count"].fillna(0.0).to_numpy("float64"),
        )

    @staticmethod
    def _check(y: object) -> np.ndarray:
        if y is None:
            raise ValueError("LocationPriceEncoder 는 정답값이 필요합니다.")
        arr = np.asarray(y, dtype="float64")
        if not np.isfinite(arr).all():
            raise ValueError("LocationPriceEncoder 의 정답값에 NaN/inf 가 있습니다.")
        return arr

    # -- 계약 ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: object = None) -> LocationPriceEncoder:
        target = self._check(y)
        self.prior_ = float(target.mean())
        self.tables_ = {level: self._table(X[level], target) for level in self.levels}
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for level in self.levels:
            value, count = self._apply(self.tables_[level], X[level])
            columns += [value, count]
        return np.column_stack(columns)

    def fit_transform(self, X: pd.DataFrame, y: object = None, **_: object) -> np.ndarray:
        target = self._check(y)
        self.fit(X, target)
        out = np.empty((len(X), 2 * len(self.levels)), dtype="float64")
        folds = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for fit_rows, fill_rows in folds.split(X):
            for j, level in enumerate(self.levels):
                table = self._table(X[level].iloc[fit_rows], target[fit_rows])
                value, count = self._apply(table, X[level].iloc[fill_rows])
                out[fill_rows, 2 * j] = value
                out[fill_rows, 2 * j + 1] = count
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = [n for level in self.levels for n in (f"loc_{level}_mean", f"loc_{level}_n")]
        return np.asarray(names, dtype=object)
