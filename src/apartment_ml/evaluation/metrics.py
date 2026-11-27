"""평가 지표 (회귀 / 분류). 모델 구현과 독립적으로 사용한다."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from apartment_ml.schemas import ANOMALY_LABEL_VALUES


class MetricInputError(ValueError):
    """평가 입력값이 잘못되었을 때 (길이 불일치, 빈 데이터, 결측, 지원하지 않는 레이블)."""


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1) if arr.ndim == 2 and 1 in arr.shape else arr
    if arr.ndim != 1:
        raise MetricInputError(f"{name} 는 1차원이어야 합니다 (shape={arr.shape}).")
    return arr


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t, p = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if t.size == 0 or p.size == 0:
        raise MetricInputError("평가 데이터가 비어 있습니다.")
    if t.shape != p.shape:
        raise MetricInputError(f"y_true({t.size})와 y_pred({p.size})의 길이가 다릅니다.")
    return t, p


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE (단위: 입력과 동일, 가격이면 원)."""
    t, p = _validate_pair(y_true, y_pred)
    try:
        t, p = t.astype("float64"), p.astype("float64")
    except (TypeError, ValueError) as exc:
        raise MetricInputError("회귀 평가 값은 숫자여야 합니다.") from exc
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise MetricInputError("회귀 평가 값에 NaN/inf 가 포함되어 있습니다.")
    err = p - t
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "n": int(t.size),
    }


def classification_metrics(
    y_true, y_pred, labels: Sequence[int] = ANOMALY_LABEL_VALUES, positive_label: int = 1
) -> dict[str, object]:
    """Precision / Recall / F1 (positive_label 기준) + Confusion Matrix.

    confusion_matrix 는 ``labels`` 순서의 행(정답) × 열(예측) 2차원 리스트.
    양성 예측이 없을 때 precision 은 0 으로 처리한다 (zero_division=0).
    """
    t, p = _validate_pair(y_true, y_pred)
    allowed = set(labels)
    if positive_label not in allowed:
        raise MetricInputError(f"positive_label {positive_label} 이 labels 에 없습니다.")
    for name, arr in (("y_true", t), ("y_pred", p)):
        try:
            values = set(arr.tolist())
        except TypeError as exc:  # pragma: no cover
            raise MetricInputError(f"{name} 값을 해석할 수 없습니다.") from exc
        bad = values - allowed
        if bad:
            raise MetricInputError(
                f"{name} 에 지원하지 않는 레이블이 있습니다: {sorted(bad, key=str)}"
            )
    t, p = t.astype(int), p.astype(int)
    kw = {"pos_label": positive_label, "zero_division": 0, "labels": list(labels)}
    if len(labels) == 2:
        kw["average"] = "binary"
    else:
        kw["average"] = "macro"
    return {
        "precision": float(precision_score(t, p, **kw)),
        "recall": float(recall_score(t, p, **kw)),
        "f1": float(f1_score(t, p, **kw)),
        "confusion_matrix": confusion_matrix(t, p, labels=list(labels)).tolist(),
        "labels": list(labels),
        "n": int(t.size),
    }
