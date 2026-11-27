"""이상거래 의심 분류 모델 공통 래퍼.

⚠️ 현재 상태: **학습 비활성화**.
이상거래 정답 레이블(``anomaly_label``)의 정의·생성 기준이 확정되지 않았으므로
- ``fit`` 은 ``training_enabled=True`` 로 명시하지 않으면 ``TrainingDisabledError`` 를 낸다.
- 레이블이 비어 있거나 한 클래스뿐이면 ``LabelsNotAvailableError`` 를 낸다.
- 임의(가짜) 레이블을 생성하는 기능은 제공하지 않는다.
레이블 기준이 확정되면 docs/MODEL_PLAN.md 를 갱신하고 settings.yaml 의
``classification.training_enabled`` 를 true 로 바꾼 뒤 사용한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import (
    assert_no_forbidden_features,
    build_feature_preprocessor,
    to_model_frame,
)
from apartment_ml.evaluation.metrics import classification_metrics
from apartment_ml.models.base import (
    LabelsNotAvailableError,
    ModelNotReadyError,
    TrainingDisabledError,
    load_artifact,
    save_artifact,
    utc_now_iso,
)

CLASSIFIER_FACTORIES: dict[str, Callable[[int], ClassifierMixin]] = {
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1
    ),
    # Boosting 계열 기본 후보. XGBoost/LightGBM 등은 feature/ml 에서 의존성 합의 후 추가.
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
}


def create_classifier(name: str, random_state: int = 42) -> ClassifierMixin:
    try:
        return CLASSIFIER_FACTORIES[name](random_state)
    except KeyError as exc:
        raise ValueError(
            f"지원하지 않는 분류 모델: {name} (가능: {list(CLASSIFIER_FACTORIES)})"
        ) from exc


def validate_anomaly_labels(y: pd.Series | np.ndarray) -> np.ndarray:
    """학습 가능한 레이블인지 확인한다. 결측/단일 클래스/허용 외 값이면 오류."""
    s = pd.Series(y)
    if len(s) == 0 or s.isna().all():
        raise LabelsNotAvailableError("anomaly_label 이 확보되지 않았습니다 (전부 결측).")
    if s.isna().any():
        raise LabelsNotAvailableError(
            f"anomaly_label 결측 {int(s.isna().sum())}건 — 레이블이 있는 행만 학습에 사용하세요."
        )
    bad = set(s.unique().tolist()) - set(S.ANOMALY_LABEL_VALUES)
    if bad:
        raise ValueError(f"지원하지 않는 레이블 값: {sorted(bad, key=str)}")
    arr = s.astype(int).to_numpy()
    if len(np.unique(arr)) < 2:
        raise LabelsNotAvailableError("레이블이 한 종류뿐이라 분류 모델을 학습할 수 없습니다.")
    return arr


class AnomalyClassificationModel:
    """``AnomalyDetector`` 계약을 만족하는 분류 모델 래퍼 (개별 모델 / Voting 공용)."""

    def __init__(
        self,
        name: str = "random_forest",
        estimator: ClassifierMixin | None = None,
        numeric_features: Sequence[str] = S.CLASSIFICATION_NUMERIC_FEATURES,
        categorical_features: Sequence[str] = S.CLASSIFICATION_CATEGORICAL_FEATURES,
        random_state: int = 42,
        threshold: float = 0.5,
    ) -> None:
        features = [*numeric_features, *categorical_features]
        assert_no_forbidden_features(features, S.CLASSIFICATION_FORBIDDEN_FEATURES)
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold 는 0 과 1 사이여야 합니다.")
        self.name = name
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.estimator = (
            estimator if estimator is not None else create_classifier(name, random_state)
        )
        self.threshold = threshold
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def fit(
        self,
        train: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        *,
        training_enabled: bool = False,
    ) -> AnomalyClassificationModel:
        if not training_enabled:
            raise TrainingDisabledError(
                "이상거래 분류 학습이 비활성화되어 있습니다. 레이블 정의 확정 후 "
                "settings.yaml 의 classification.training_enabled 를 true 로 설정하세요."
            )
        target = train[S.CLASSIFICATION_TARGET] if y is None else y
        labels = validate_anomaly_labels(target)
        X = to_model_frame(train, self.feature_columns)
        if len(labels) != len(X):
            raise ValueError("입력과 레이블의 길이가 다릅니다.")
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    # HistGradientBoostingClassifier requires dense input, including in Voting.
                    build_feature_preprocessor(
                        self.numeric_features, self.categorical_features, sparse_threshold=0.0
                    ),
                ),
                ("model", self.estimator),
            ]
        )
        pipeline.fit(X, labels)
        self.pipeline = pipeline
        self.metadata = {
            "model_name": self.name,
            "trained_at": utc_now_iso(),
            "n_train": int(len(X)),
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
        }
        return self

    def _require_fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise ModelNotReadyError(f"이상거래 분류 모델 '{self.name}' 이 학습되지 않았습니다.")
        return self.pipeline

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """이상(1) 클래스 확률, shape (n,)."""
        pipe = self._require_fitted()
        proba = pipe.predict_proba(to_model_frame(X, self.feature_columns))
        classes = list(pipe.classes_)
        return proba[:, classes.index(1)]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, data: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> dict:
        target = data[S.CLASSIFICATION_TARGET] if y is None else y
        return classification_metrics(validate_anomaly_labels(target), self.predict(data))

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted:
            raise ModelNotReadyError("학습되지 않은 모델은 저장할 수 없습니다.")
        return save_artifact(self, {**self.metadata, "kind": "anomaly_classification"}, path)

    @classmethod
    def load(cls, path: str | Path) -> AnomalyClassificationModel:
        obj, meta = load_artifact(path)
        if not isinstance(obj, cls) or meta.get("kind") != "anomaly_classification":
            raise ModelNotReadyError(f"이상거래 분류 모델 파일이 아닙니다: {path}")
        return obj
