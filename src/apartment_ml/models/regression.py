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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import assert_no_forbidden_features, to_model_frame
from apartment_ml.evaluation.metrics import regression_metrics
from apartment_ml.models.base import ModelNotReadyError, load_artifact, save_artifact, utc_now_iso
from apartment_ml.models.location_price import LocationPriceEncoder
from apartment_ml.models.market_index import MarketIndex

# 저장된 모델을 **다른 버전의 코드로** 불러오면, 파이프라인은 그대로 복원되지만
# 예측값을 가격으로 되돌리는 방식(_to_price)이 달라 조용히 틀린 값이 나온다.
# 실제로 상대가격으로 학습한 파일을 절대가격 코드로 읽어 1원이 나온 적이 있다.
# 오류 없이 틀리는 것이 가장 위험하므로, 예측 경로가 바뀌면 이 숫자를 올리고
# 불일치하는 파일은 불러오기를 거부한다.
PREDICTION_CONTRACT = 2

REGRESSOR_FACTORIES: dict[str, Callable[[int], RegressorMixin]] = {
    "linear_regression": lambda seed: LinearRegression(),
    "decision_tree": lambda seed: DecisionTreeRegressor(
        max_depth=14, min_samples_leaf=5, random_state=seed
    ),
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1
    ),
    # 측정 결과 이 데이터에서 가장 잘 맞는 모델. 잔차를 순차로 줄이기 때문에
    # 자치구 x 면적 x 층 같은 깊은 상호작용을 잡는다 (랜덤포레스트는 독립 트리의
    # 평균이라 그게 뭉개진다).
    "gradient_boosting": lambda seed: HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.1, random_state=seed, early_stopping=False
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
        name: str = "gradient_boosting",
        estimator: RegressorMixin | None = None,
        numeric_features: Sequence[str] = S.REGRESSION_NUMERIC_FEATURES,
        categorical_features: Sequence[str] = S.REGRESSION_CATEGORICAL_FEATURES,
        location_features: Sequence[str] = S.REGRESSION_LOCATION_FEATURES,
        log_target: bool = True,
        relative_to_market: bool = True,
        market_index: MarketIndex | None = None,
        random_state: int = 42,
    ) -> None:
        features = [*numeric_features, *categorical_features, *location_features]
        assert_no_forbidden_features(features, S.REGRESSION_FORBIDDEN_FEATURES)
        if location_features and not log_target:
            raise ValueError(
                "위치 인코딩(location_features)은 log_target=True 에서만 쓸 수 있습니다."
            )
        self.name = name
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.location_features = list(location_features)
        self.log_target = log_target
        if relative_to_market and not log_target:
            raise ValueError("상대가격 학습(relative_to_market)은 log_target=True 에서만 됩니다.")
        self.relative_to_market = relative_to_market or market_index is not None
        self.market_: MarketIndex | None = market_index
        self.estimator = (
            estimator if estimator is not None else create_regressor(name, random_state)
        )
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}

    # -- 계약 ---------------------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        """파이프라인에 넣는 컬럼. 위치 컬럼은 인코더가 숫자로 바꿔 쓰므로 함께 받는다."""
        extra = [c for c in self.location_features if c not in self.categorical_features]
        return [*self.numeric_features, *self.categorical_features, *extra]

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    @property
    def feature_ranges(self) -> dict[str, tuple[float, float]]:
        """학습 데이터의 수치형 변수 [최소, 최대] — 학습 범위 밖 입력 경고에 사용."""
        return dict(self.metadata.get("feature_ranges", {}))

    def _build_preprocessor(self) -> ColumnTransformer:
        """결측 대체 + 스케일링 + 원-핫 + 위치 타깃 인코딩. 반드시 학습 데이터로만 fit 된다."""
        steps: list[tuple[str, object, list[str]]] = [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                self.numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                self.categorical_features,
            ),
        ]
        if self.location_features:
            steps.append(
                (
                    "loc",
                    LocationPriceEncoder(
                        levels=self.location_features, random_state=self.random_state
                    ),
                    self.location_features,
                )
            )
        return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)

    def _to_model_target(self, df: pd.DataFrame, price: np.ndarray) -> np.ndarray:
        """가격(원) → 모델이 맞히는 값. 역변환은 ``_to_price``."""
        if not self.log_target:
            return price
        if (price <= 0).any():
            raise ValueError("로그 타깃에는 0 이하의 가격을 쓸 수 없습니다.")
        if self.market_ is None:
            return np.log(price)
        area = pd.to_numeric(df[S.AREA_M2], errors="coerce").to_numpy("float64")
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("상대가격 학습에는 모든 행의 전용면적이 필요합니다.")
        return np.log(price / area) - self.market_.transform(df)

    def _to_price(self, df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        if not self.log_target:
            return pred
        if self.market_ is None:
            return np.exp(pred)
        area = pd.to_numeric(df[S.AREA_M2], errors="coerce").to_numpy("float64")
        return np.exp(pred + self.market_.transform(df)) * area

    def fit(
        self, train: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> PriceRegressionModel:
        """train: 전처리된 거래 DataFrame. y 생략 시 ``train['price_krw']`` 사용.

        ``relative_to_market=True`` 면 시장지수를 **이 학습 데이터에서** 만든다.
        운영처럼 최신 지수를 쓰려면 ``market_index=`` 로 직접 넘긴다.
        """
        X = to_model_frame(train, self.feature_columns)
        price = np.asarray(train[S.REGRESSION_TARGET] if y is None else y, dtype="float64")
        if len(X) == 0:
            raise ValueError("학습 데이터가 비어 있습니다.")
        if len(price) != len(X) or not np.isfinite(price).all():
            raise ValueError("정답값(price_krw)의 길이가 맞지 않거나 결측이 있습니다.")
        if self.relative_to_market and self.market_ is None:
            self.market_ = MarketIndex().fit(train.assign(**{S.REGRESSION_TARGET: price}))
        target = self._to_model_target(train, price)
        pipeline = Pipeline([("preprocess", self._build_preprocessor()), ("model", self.estimator)])
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
            "location_features": self.location_features,
            "log_target": self.log_target,
            "relative_to_market": self.market_ is not None,
            "feature_ranges": ranges,
            "train_date_range": _date_range(train),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ModelNotReadyError(f"회귀 모델 '{self.name}' 이 학습되지 않았습니다.")
        pred = self.pipeline.predict(to_model_frame(X, self.feature_columns))
        return self._to_price(X, pred)

    def evaluate(
        self, data: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> dict[str, float]:
        target = data[S.REGRESSION_TARGET] if y is None else y
        return regression_metrics(target, self.predict(data))

    # -- 저장 / 불러오기 -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if not self.is_fitted:
            raise ModelNotReadyError("학습되지 않은 모델은 저장할 수 없습니다.")
        meta = {
            **self.metadata,
            "kind": "price_regression",
            "prediction_contract": PREDICTION_CONTRACT,
        }
        return save_artifact(self, meta, path)

    @classmethod
    def load(cls, path: str | Path) -> PriceRegressionModel:
        obj, meta = load_artifact(path)
        if not isinstance(obj, cls) or meta.get("kind") != "price_regression":
            raise ModelNotReadyError(f"가격 회귀 모델 파일이 아닙니다: {path}")
        saved = meta.get("prediction_contract")
        if saved != PREDICTION_CONTRACT:
            raise ModelNotReadyError(
                f"이 모델 파일은 예측 방식이 다른 버전의 코드로 저장되었습니다 "
                f"(파일 {saved!r}, 현재 {PREDICTION_CONTRACT}). "
                f"scripts/train_price.py 로 다시 학습하세요: {path}"
            )
        return obj


def _date_range(df: pd.DataFrame) -> list[str] | None:
    if S.CONTRACT_DATE not in df.columns or df.empty:
        return None
    d = pd.to_datetime(df[S.CONTRACT_DATE])
    return [d.min().date().isoformat(), d.max().date().isoformat()]
