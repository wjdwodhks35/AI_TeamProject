"""모델 공통 계약: 예외, Protocol, 저장/불러오기 형식."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd

ARTIFACT_FORMAT_VERSION = 1


class ModelNotReadyError(RuntimeError):
    """학습되지 않았거나 불러오지 못한 모델을 사용하려 할 때."""


class LabelsNotAvailableError(RuntimeError):
    """이상거래 정답 레이블이 확보되지 않아 학습할 수 없을 때."""


class TrainingDisabledError(RuntimeError):
    """설정상 학습이 비활성화되어 있을 때 (예: classification.training_enabled=false)."""


@runtime_checkable
class PricePredictor(Protocol):
    """가격 예측기 계약. API·금리 시나리오는 이 계약에만 의존한다."""

    name: str

    @property
    def is_fitted(self) -> bool: ...

    @property
    def feature_columns(self) -> list[str]: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class AnomalyDetector(Protocol):
    """이상거래 판정기 계약. ``predict_proba`` 는 (n,) 형태의 '이상(1)' 확률을 반환한다."""

    name: str

    @property
    def is_fitted(self) -> bool: ...

    @property
    def feature_columns(self) -> list[str]: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_artifact(obj: Any, metadata: dict[str, Any], path: str | Path) -> Path:
    """모델 객체와 메타데이터를 joblib 파일 하나로 저장한다.

    파일 형식: ``{"format_version": int, "metadata": dict, "model": obj}``
    ※ joblib/pickle 파일은 신뢰할 수 있는 출처의 것만 불러올 것.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"format_version": ARTIFACT_FORMAT_VERSION, "metadata": metadata, "model": obj}, path
    )
    return path


def load_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ModelNotReadyError(f"모델 파일이 없습니다: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ModelNotReadyError(f"지원하지 않는 모델 파일 형식입니다: {path}")
    return payload["model"], payload["metadata"]
