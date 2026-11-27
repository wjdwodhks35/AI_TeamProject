"""API 요청·응답 스키마 (docs/API_CONTRACT.md 와 동기화할 것)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apartment_ml.models.forecast import MAX_HORIZON as MAX_FORECAST_MONTHS
from apartment_ml.schemas import (
    MAX_BASE_RATE,
    MAX_PRICE_KRW,
    MIN_BASE_RATE,
    MIN_PRICE_KRW,
    ApartmentFeatures,
)

EXAMPLE_FEATURES = {
    "district": "강남구",
    # 동·단지는 선택 입력이지만 넣어야 정확도가 제대로 나온다 (없으면 MAE 1.41억 -> 4.18억).
    # 고를 수 있는 값은 GET /locations/강남구 가 내려준다.
    "dong": "대치동",
    "complex_id": "11680-2234",
    "contract_date": "2024-05-20",
    "area_m2": 84.9,
    "floor": 12,
    "built_year": 2008,
    "base_rate": 3.5,
    "regulation": {"speculation_overheated": True, "adjustment_target": True, "land_permit": False},
}


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models: dict[str, bool] = Field(..., description="모델별 준비(로드) 여부")


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["MODEL_NOT_READY"])
    message: str
    issues: list[dict[str, object]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class PricePredictionRequest(ApartmentFeatures):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_FEATURES]})


class ComplexItem(BaseModel):
    """단지 하나. `complex_id` 를 그대로 예측 요청의 `complex_id` 로 보내면 된다."""

    complex_id: str = Field(..., examples=["11680-2234"])
    apartment_name: str = Field(..., examples=["은마"])
    dong: str = Field(..., examples=["대치동"])
    n_deals: int = Field(..., description="학습 데이터에서의 거래 수 (많을수록 추정이 안정적)")
    built_year: int | None = Field(
        None, description="준공연도. 단지마다 고정이라 화면에서 자동으로 채우면 됩니다"
    )


class LocationsResponse(BaseModel):
    """자치구 하나의 동·단지 목록. 프론트는 자치구를 고를 때 한 번만 호출하면 된다."""

    district: str
    dongs: list[str] = Field(..., description="법정동 목록 (가나다순)")
    complexes: list[ComplexItem] = Field(..., description="단지 목록 (동 -> 단지명 순)")


class RegulationStateResponse(BaseModel):
    """그 날 그 자치구에 걸려 있던 규제. 고시로 정해진 사실이라 사용자가 입력할 값이 아니다."""

    district: str
    on: str = Field(..., description="기준일 YYYY-MM-DD")
    speculation_overheated: bool = Field(..., description="투기과열지구")
    adjustment_target: bool = Field(..., description="조정대상지역")
    land_permit: bool = Field(..., description="토지거래허가구역")
    note: str = Field(
        "자치구 단위 판정입니다. 동·단지 단위로 지정된 구역은 그 자치구 전체로 적용됩니다.",
        description="해석 주의",
    )


class PricePredictionResponse(BaseModel):
    predicted_price_krw: int = Field(..., description="예상 거래가격(원)")
    model_name: str
    warnings: list[str] = Field(default_factory=list)


class ForecastRequest(ApartmentFeatures):
    """향후 N개월 예측 요청. 계약일은 '지금 기준'을 뜻하며 예측 달은 서버가 정한다."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{**EXAMPLE_FEATURES, "months": 12}]}
    )
    months: int = Field(12, ge=1, le=MAX_FORECAST_MONTHS, description="몇 개월 앞까지")


class ForecastPointOut(BaseModel):
    """한 달치 예측. 구간은 80% 범위이고 지평이 길수록 넓어진다."""

    month: str = Field(..., examples=["2026-10"], description="YYYY-MM")
    horizon: int = Field(..., description="기준월로부터 몇 번째 달인지")
    predicted_price_krw: int
    low_krw: int = Field(..., description="80% 구간 하한")
    high_krw: int = Field(..., description="80% 구간 상한")


class ForecastResponse(BaseModel):
    base_month: str = Field(..., description="예측의 출발이 되는 마지막 실거래 집계월")
    model_name: str
    points: list[ForecastPointOut]
    warnings: list[str] = Field(default_factory=list)


class AnomalyPredictionRequest(ApartmentFeatures):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{**EXAMPLE_FEATURES, "price_krw": 2_500_000_000}]},
    )
    price_krw: int = Field(..., ge=MIN_PRICE_KRW, le=MAX_PRICE_KRW, description="실제 거래가격(원)")


class AnomalyPredictionResponse(BaseModel):
    is_suspicious: bool = Field(..., description="이상거래 '의심' 여부 (확정 판정 아님)")
    suspicion_score: float = Field(..., ge=0, le=1, description="이상(1) 클래스 확률")
    threshold: float
    reference_price_krw: int | None = Field(
        None, description="기준가격: 가격 회귀 모델의 예상가격 (실거래가 정답값과 다른 개념)"
    )
    price_gap_pct: float | None = Field(None, description="(거래가 - 기준가격) / 기준가격 × 100")
    model_name: str
    # 규칙 판정. 모델보다 성능이 좋아서 같이 돌려준다 (models/anomaly_rules.py).
    rule_level: str = Field(
        ..., description="strong(강한 의심) / suspect(의심) / normal(의심 아님)"
    )
    rule_label: str = Field(..., description="rule_level 을 사람이 읽는 말로")
    rule_reasons: list[str] = Field(
        default_factory=list, description="그 판정이 나온 이유. 화면에 그대로 보여주세요"
    )
    warnings: list[str] = Field(default_factory=list)


class InterestRateScenarioRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"features": EXAMPLE_FEATURES, "base_rate_before": 3.5, "base_rate_after": 2.5}
            ]
        },
    )
    features: ApartmentFeatures = Field(..., description="거래 조건 (features.base_rate 는 무시됨)")
    base_rate_before: float = Field(..., ge=MIN_BASE_RATE, le=MAX_BASE_RATE)
    base_rate_after: float = Field(..., ge=MIN_BASE_RATE, le=MAX_BASE_RATE)


class RateCurvePoint(BaseModel):
    base_rate: float
    predicted_price_krw: float


class InterestRateScenarioResponse(BaseModel):
    base_rate_before: float
    base_rate_after: float
    predicted_price_before_krw: float
    predicted_price_after_krw: float
    price_diff_krw: float
    price_change_pct: float | None
    model_name: str
    # 두 값만 비교하면 매끄러운 반응으로 오해한다. 실제로는 분기 몇 개가 만든 계단이다.
    curve: list[RateCurvePoint] = Field(
        default_factory=list, description="관측된 금리 범위 전체의 예측가격. 화면에 그려 주세요"
    )
    observed_range: list[float] | None = Field(
        None, description="학습 데이터의 기준금리 [최소, 최대]. 이 밖은 모델이 구분하지 못합니다"
    )
    extrapolated: bool = Field(
        False, description="요청한 금리가 관측 범위 밖인지. true 면 경계와 같은 답입니다"
    )
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
