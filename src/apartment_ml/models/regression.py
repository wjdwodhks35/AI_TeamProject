"""가격 회귀 모델 공통 래퍼.

- 전처리(결측 대체·스케일링·인코딩)와 추정기를 하나의 scikit-learn Pipeline 으로 묶어
  ``fit`` 시 학습 데이터에만 전처리를 fit 한다 (데이터 누출 방지).
- 입력 변수에 ``price_krw`` 등 정답/정답 파생 컬럼이 포함되면 오류를 낸다.
- 성능 비교·튜닝·피처 엔지니어링은 feature/ml 담당자가 이 래퍼 위에서 수행한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import (
    assert_no_forbidden_features,
    build_feature_preprocessor,
    to_model_frame,
)
from apartment_ml.evaluation.metrics import regression_metrics
from apartment_ml.models.base import ModelNotReadyError, load_artifact, save_artifact, utc_now_iso

REGRESSOR_FACTORIES: dict[str, Callable[[int], RegressorMixin]] = {
    "linear_regression": lambda seed: LinearRegression(),
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=200, random_state=seed, n_jobs=-1
    ),
}


def create_regressor(name: str, random_state: int = 42) -> RegressorMixin:
    try:
        return REGRESSOR_FACTORIES[name](random_state)
    except KeyError as exc:
        raise ValueError(
            f"지원하지 않는 회귀 모델: {name} (가능: {list(REGRESSOR_FACTORIES)})"
        ) from exc


class PriceRegressionModel:
    """``PricePredictor`` 계약을 만족하는 회귀 모델."""

    def __init__(
        self,
        name: str = "linear_regression",
        estimator: RegressorMixin | None = None,
        numeric_features: Sequence[str] = S.REGRESSION_NUMERIC_FEATURES,
        categorical_features: Sequence[str] = S.REGRESSION_CATEGORICAL_FEATURES,
        random_state: int = 42,
    ) -> None:
        features = [*numeric_features, *categorical_features]
        assert_no_forbidden_features(features, S.REGRESSION_FORBIDDEN_FEATURES)
        self.name = name
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.estimator = (
            estimator if estimator is not None else create_regressor(name, random_state)
        )
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}

    # -- 계약 ---------------------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    @property
    def feature_ranges(self) -> dict[str, tuple[float, float]]:
        """학습 데이터의 수치형 변수 [최소, 최대] — 학습 범위 밖 입력 경고에 사용."""
        return dict(self.metadata.get("feature_ranges", {}))

    def fit(
        self, train: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> PriceRegressionModel:
        """train: 전처리된 거래 DataFrame. y 생략 시 ``train['price_krw']`` 사용."""
        X = to_model_frame(train, self.feature_columns)
        target = train[S.REGRESSION_TARGET] if y is None else y
        target = np.asarray(target, dtype="float64")
        if len(X) == 0:
            raise ValueError("학습 데이터가 비어 있습니다.")
        if len(target) != len(X) or not np.isfinite(target).all():
            raise ValueError("정답값(price_krw)의 길이가 맞지 않거나 결측이 있습니다.")
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_feature_preprocessor(self.numeric_features, self.categorical_features),
                ),
                ("model", self.estimator),
            ]
        )
        pipeline.fit(X, target)
        self.pipeline = pipeline
        ranges = {
            c: (float(X[c].min()), float(X[c].max()))
            for c in self.numeric_features
            if X[c].notna().any()
        }
        self.metadata = {
            "model_name": self.name,
            "trained_at": utc_now_iso(),
            "n_train": int(len(X)),
            "feature_columns": self.feature_columns,
            "feature_ranges": ranges,
            "train_date_range": _date_range(train),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ModelNotReadyError(f"회귀 모델 '{self.name}' 이 학습되지 않았습니다.")
        return self.pipeline.predict(to_model_frame(X, self.feature_columns))

    def evaluate(
        self, data: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> dict[str, float]:
        target = data[S.REGRESSION_TARGET] if y is None else y
        return regression_metrics(target, self.predict(data))

    # -- 저장 / 불러오기 -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if not self.is_fitted:
            raise ModelNotReadyError("학습되지 않은 모델은 저장할 수 없습니다.")
        return save_artifact(self, {**self.metadata, "kind": "price_regression"}, path)

    @classmethod
    def load(cls, path: str | Path) -> PriceRegressionModel:
        obj, meta = load_artifact(path)
        if not isinstance(obj, cls) or meta.get("kind") != "price_regression":
            raise ModelNotReadyError(f"가격 회귀 모델 파일이 아닙니다: {path}")
        return obj


def _date_range(df: pd.DataFrame) -> list[str] | None:
    if S.CONTRACT_DATE not in df.columns or df.empty:
        return None
    d = pd.to_datetime(df[S.CONTRACT_DATE])
    return [d.min().date().isoformat(), d.max().date().isoformat()]
