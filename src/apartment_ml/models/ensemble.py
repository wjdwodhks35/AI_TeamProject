"""Voting 앙상블 구성.

여기서는 **학습되지 않은** VotingClassifier 를 구성하는 기능만 제공한다.
학습은 ``AnomalyClassificationModel(estimator=...)`` 로 감싸서 수행하며,
레이블 미확보 상태에서는 classification.py 의 학습 차단 규칙이 그대로 적용된다.

개별 모델 vs 앙상블 비교는 동일한 시간순 분할(train/valid/test)과 동일한 지표
(evaluation.metrics.classification_metrics)로 수행한다 — docs/MODEL_PLAN.md 참고.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from sklearn.ensemble import VotingClassifier

from apartment_ml import schemas as S
from apartment_ml.models.classification import AnomalyClassificationModel, create_classifier

DEFAULT_MEMBERS: tuple[str, ...] = ("decision_tree", "random_forest", "hist_gradient_boosting")


def build_voting_classifier(
    members: Sequence[str] = DEFAULT_MEMBERS,
    voting: Literal["hard", "soft"] = "soft",
    weights: Sequence[float] | None = None,
    random_state: int = 42,
) -> VotingClassifier:
    if len(members) < 2:
        raise ValueError("Voting 앙상블에는 최소 2개 모델이 필요합니다.")
    if len(set(members)) != len(members):
        raise ValueError("앙상블 구성 모델 이름이 중복되었습니다.")
    if weights is not None and len(weights) != len(members):
        raise ValueError("weights 길이는 members 길이와 같아야 합니다.")
    estimators = [(m, create_classifier(m, random_state)) for m in members]
    return VotingClassifier(estimators=estimators, voting=voting, weights=weights)


def build_voting_model(
    members: Sequence[str] = DEFAULT_MEMBERS,
    voting: Literal["hard", "soft"] = "soft",
    weights: Sequence[float] | None = None,
    random_state: int = 42,
    *,
    numeric_features: Sequence[str] = S.CLASSIFICATION_NUMERIC_FEATURES,
) -> AnomalyClassificationModel:
    """공통 래퍼로 감싼 (미학습) Voting 앙상블. predict_proba 를 쓰려면 soft voting 필요."""
    if voting != "soft":
        raise ValueError("AnomalyDetector 계약(predict_proba)을 위해 soft voting 만 지원합니다.")
    return AnomalyClassificationModel(
        name="voting_" + "+".join(members),
        estimator=build_voting_classifier(members, voting, weights, random_state),
        numeric_features=numeric_features,
        random_state=random_state,
    )
