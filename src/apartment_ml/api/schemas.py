"""API 요청·응답 스키마 (docs/API_CONTRACT.md 와 동기화할 것)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

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


class LocationsResponse(BaseModel):
    """자치구 하나의 동·단지 목록. 프론트는 자치구를 고를 때 한 번만 호출하면 된다."""

    district: str
    dongs: list[str] = Field(..., description="법정동 목록 (가나다순)")
    complexes: list[ComplexItem] = Field(..., description="단지 목록 (동 -> 단지명 순)")


class PricePredictionResponse(BaseModel):
    predicted_price_krw: int = Field(..., description="예상 거래가격(원)")
    model_name: str
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


class InterestRateScenarioResponse(BaseModel):
    base_rate_before: float
    base_rate_after: float
    predicted_price_before_krw: float
    predicted_price_after_krw: float
    price_diff_krw: float
    price_change_pct: float | None
    model_name: str
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
