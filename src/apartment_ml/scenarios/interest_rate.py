"""금리 시나리오 비교.

동일한 거래 조건에서 ``base_rate`` 만 바꿔 회귀 모델의 예측가격을 비교한다.

두 값만 비교하면 안 되는 이유
-----------------------------
모델의 금리 반응은 **매끄러운 관계가 아니라 계단**이다. 대치동 은마로 재면:

    0.5 ~ 2.5%   37.7억 (거의 평평)
    2.5 -> 3.0%  -10.4%  <- 분기 하나가 만든 절벽
    3.5% 초과    3.5% 와 완전히 동일

학습 데이터의 기준금리가 0.5~3.5% 열두 값뿐이라, **그 밖은 트리가 구분하지 못한다.**
5% 를 넣어도 3.5% 와 소수점까지 같은 답이 나온다. 숫자 하나만 돌려주면 사람은 그것을
매끄러운 반응으로 읽는다. 그래서 관측 범위 전체의 **곡선**을 같이 돌려준다.

⚠️ 해석 주의: 결과는 '학습된 모델이 입력 조건 변화에 따라 산출한 예측값의 차이' 일 뿐이며,
금리 변화가 실제 시장 가격에 미치는 **인과효과가 아니다**. 금리 변수는 전체 정확도에
기여하지도 않는다 (값을 섞어도 MAE 변화 0.006억 미만, docs/MODEL_PLAN.md).
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


# 한국은행 기준금리는 0.25%p 단위로 움직인다. 곡선을 그 간격으로 뽑는다.
CURVE_STEP = 0.25


@dataclass
class InterestRateScenarioResult:
    base_rate_before: float
    base_rate_after: float
    predicted_price_before_krw: float
    predicted_price_after_krw: float
    price_diff_krw: float
    price_change_pct: float | None
    model_name: str
    # 관측 범위 전체의 반응. 계단이라는 사실을 그래프로 드러내기 위한 것이다.
    curve: list[dict[str, float]] = field(default_factory=list)
    observed_range: list[float] | None = None
    extrapolated: bool = False
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
    curve, observed, outside = _rate_curve(model, base, (before, after))
    warnings = out_of_range_warnings(model, frame)
    if outside:
        lo, hi = observed
        warnings.append(
            f"학습 데이터의 기준금리는 {lo:g}~{hi:g}% 뿐입니다. 그 밖의 값은 모델이 "
            f"구분하지 못해서 가장 가까운 경계({lo:g}% 또는 {hi:g}%)와 똑같은 답이 "
            f"나옵니다. 숫자가 달라 보여도 새로 계산된 것이 아닙니다."
        )
    step = _largest_step(curve)
    if step is not None:
        lo_rate, hi_rate, change = step
        warnings.append(
            f"금리 반응은 매끄럽지 않습니다. {lo_rate:g}% 에서 {hi_rate:g}% 로 "
            f"넘어가는 구간 하나에서만 {abs(change):.1f}% {'내려갑니다' if change < 0 else '올라갑니다'}. "
            f"분기 하나가 만든 계단이므로 두 값의 차이만 보지 말고 곡선 전체를 보세요."
        )
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
        curve=curve,
        observed_range=list(observed) if observed else None,
        extrapolated=outside,
        warnings=warnings,
    )


# 인접 구간에서 이만큼 넘게 움직이면 '계단' 으로 보고 알린다.
STEP_WARN_PCT = 3.0


def _largest_step(curve: list[dict[str, float]]) -> tuple[float, float, float] | None:
    """곡선에서 가장 큰 한 칸 변화를 찾는다. 작으면 None."""
    biggest: tuple[float, float, float] | None = None
    for left, right in zip(curve, curve[1:], strict=False):
        before, after = left["predicted_price_krw"], right["predicted_price_krw"]
        if before <= 0:
            continue
        change = (after - before) / before * 100.0
        if abs(change) >= STEP_WARN_PCT and (biggest is None or abs(change) > abs(biggest[2])):
            biggest = (left["base_rate"], right["base_rate"], change)
    return biggest


def _rate_curve(
    model: PricePredictor, base: pd.DataFrame, chosen: tuple[float, float]
) -> tuple[list[dict[str, float]], tuple[float, float] | None, bool]:
    """관측된 기준금리 범위 전체에서 예측가격을 뽑는다.

    돌려주는 것: (곡선, 관측 범위, 요청값이 범위 밖인지).
    범위를 모르면 곡선 없이 빈 목록을 준다.
    """
    ranges = getattr(model, "feature_ranges", None) or {}
    bounds = ranges.get(S.BASE_RATE)
    if not bounds:
        return [], None, False
    lo, hi = float(bounds[0]), float(bounds[1])

    rates: set[float] = {lo, hi}
    step = CURVE_STEP
    value = lo
    while value < hi:
        value = round(value + step, 4)
        if value <= hi:
            rates.add(value)
    rates.update(r for r in chosen if lo <= r <= hi)

    ordered = sorted(rates)
    frame = pd.concat([base] * len(ordered), ignore_index=True)
    frame[S.BASE_RATE] = ordered
    outside = any(not (lo <= r <= hi) for r in chosen)

    prices = model.predict(frame)
    # 곡선은 부가 정보다. 모델이 요청한 수만큼 돌려주지 않으면 곡선만 빼고 넘어간다
    # (본체 비교값은 위에서 이미 검증했다).
    if len(prices) != len(ordered):
        return [], (lo, hi), outside
    curve = [
        {"base_rate": float(r), "predicted_price_krw": float(p)}
        for r, p in zip(ordered, prices, strict=True)
        if math.isfinite(float(p))
    ]
    return curve, (lo, hi), outside
