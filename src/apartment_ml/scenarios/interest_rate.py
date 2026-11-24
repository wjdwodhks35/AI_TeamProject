"""금리 시나리오 비교.

동일한 거래 조건에서 ``base_rate`` 만 바꿔 회귀 모델의 예측가격을 비교한다.

⚠️ 해석 주의: 결과는 '학습된 모델이 입력 조건 변화에 따라 산출한 예측값의 차이' 일 뿐이며,
금리 변화가 실제 시장 가격에 미치는 **인과효과가 아니다**.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import add_date_features, ensure_optional_columns
from apartment_ml.models.base import ModelNotReadyError, PricePredictor

DISCLAIMER = "모델 예측값의 조건별 비교 결과이며, 금리 변동이 실제 시장 가격에 미치는 인과효과를 의미하지 않습니다."


class ScenarioNotSupportedError(ValueError):
    """모델이 금리를 입력 변수로 사용하지 않을 때."""


@dataclass
class InterestRateScenarioResult:
    base_rate_before: float
    base_rate_after: float
    predicted_price_before_krw: float
    predicted_price_after_krw: float
    price_diff_krw: float
    price_change_pct: float | None
    model_name: str
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _check_rate(value: float, label: str) -> float:
    v = float(value)
    if not math.isfinite(v) or not S.MIN_BASE_RATE <= v <= S.MAX_BASE_RATE:
        raise ValueError(
            f"{label} 기준금리가 허용 범위({S.MIN_BASE_RATE}~{S.MAX_BASE_RATE}%) 밖입니다: {v}"
        )
    return v


def build_feature_frame(features: S.ApartmentFeatures | dict) -> pd.DataFrame:
    """단건 거래 조건 → 모델 입력용 1행 DataFrame (파생변수 포함)."""
    if not isinstance(features, S.ApartmentFeatures):
        features = S.ApartmentFeatures.model_validate(features)
    frame = pd.DataFrame([features.to_record()])
    frame[S.CONTRACT_DATE] = pd.to_datetime(frame[S.CONTRACT_DATE])
    for col in (S.BUILT_YEAR, *S.REGULATION_COLUMNS):
        frame[col] = frame[col].astype("Int64")
    frame[S.BASE_RATE] = frame[S.BASE_RATE].astype("float64")
    return add_date_features(ensure_optional_columns(frame))


def out_of_range_warnings(model: PricePredictor, frame: pd.DataFrame) -> list[str]:
    ranges = getattr(model, "feature_ranges", None) or {}
    warnings: list[str] = []
    for col, (lo, hi) in ranges.items():
        if col not in frame.columns:
            continue
        for v in frame[col].dropna().unique():
            if v < lo or v > hi:
                warnings.append(f"{col}={v} 은(는) 학습 데이터 범위 [{lo}, {hi}] 밖입니다 (외삽).")
    return warnings


def compare_interest_rate_scenario(
    model: PricePredictor | None,
    features: S.ApartmentFeatures | dict,
    base_rate_before: float,
    base_rate_after: float,
) -> InterestRateScenarioResult:
    """금리 이외 조건을 고정하고 두 기준금리에서의 예측가격을 비교한다.

    ``features.base_rate`` 값은 무시되고 ``base_rate_before``/``base_rate_after`` 로 대체된다.
    학습된 모델이 없으면 ``ModelNotReadyError`` (임의 가격을 반환하지 않음).
    """
    if model is None or not getattr(model, "is_fitted", False):
        raise ModelNotReadyError("학습된 가격 회귀 모델이 없어 금리 시나리오를 계산할 수 없습니다.")
    if S.BASE_RATE not in model.feature_columns:
        raise ScenarioNotSupportedError(
            "이 모델은 base_rate 를 입력 변수로 사용하지 않아 금리 시나리오를 비교할 수 없습니다."
        )
    before = _check_rate(base_rate_before, "기존")
    after = _check_rate(base_rate_after, "변경")

    base = build_feature_frame(features)
    frame = pd.concat([base, base], ignore_index=True)  # 금리 외 모든 조건 동일
    frame[S.BASE_RATE] = [before, after]

    preds = model.predict(frame)
    if getattr(preds, "shape", None) != (2,):
        raise RuntimeError("Scenario predictions must have shape (2,)")
    p_before, p_after = float(preds[0]), float(preds[1])
    if not all(math.isfinite(p) for p in (p_before, p_after)):
        raise RuntimeError("Scenario predictions must be finite")
    warnings = out_of_range_warnings(model, frame)
    diff = p_after - p_before
    pct: float | None = None
    if p_before != 0:
        pct = diff / p_before * 100.0
    else:
        warnings.append("기존 조건의 예측가격이 0이어서 변화율을 계산하지 않았습니다.")
    if p_before < 0 or p_after < 0:
        warnings.append("음수 예측가격이 포함되어 있습니다. 모델과 입력 조건을 확인하세요.")
    if not math.isfinite(diff) or (pct is not None and not math.isfinite(pct)):
        raise RuntimeError("Scenario calculations must be finite")
    return InterestRateScenarioResult(
        base_rate_before=before,
        base_rate_after=after,
        predicted_price_before_krw=p_before,
        predicted_price_after_krw=p_after,
        price_diff_krw=diff,
        price_change_pct=pct,
        model_name=getattr(model, "name", type(model).__name__),
        warnings=warnings,
    )
