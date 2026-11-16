"""공통 데이터 계약(Data Contract).

모든 모듈(수집·전처리·모델·시나리오·API)은 이 파일에 정의된 컬럼명과 규칙을 사용한다.
필드를 추가·변경할 때는 반드시 ``docs/DATA_SCHEMA.md`` 도 함께 수정한다.

용어 구분 (중요)
----------------
- ``price_krw``          : 실제 거래가격(원). 가격 회귀 모델의 **정답값(target)**.
                           회귀 모델의 입력 변수로 사용 금지. 이상거래 분류에서는 입력 변수로 사용 가능.
- ``anomaly_label``      : 이상거래 분류의 **정답값(target)**. 입력 변수 아님. 현재 미확보(전부 null 허용).
- ``reference_price_krw``: '기준가격'. 회귀 모델이 산출한 예상가격 등, 거래가격을 비교하기 위한 기준값.
                           실제 거래가격(``price_krw``)과 다른 개념이며 데이터 원천 컬럼이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# 도메인 상수
# ---------------------------------------------------------------------------

SEOUL_DISTRICTS: tuple[str, ...] = (
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
    "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
    "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
)  # fmt: skip

# 유효성 검사 범위 (데이터 품질 기준 — 변경 시 DATA_SCHEMA.md 갱신)
MIN_AREA_M2 = 10.0
MAX_AREA_M2 = 400.0
MIN_FLOOR = -5  # 국토교통부 자료에는 지하층이 음수로 기록될 수 있음
MAX_FLOOR = 100
MIN_BUILT_YEAR = 1950
MAX_BUILT_YEAR = 2030
MIN_PRICE_KRW = 1  # 0 이하 금지
MAX_PRICE_KRW = 1_000_000_000_000  # 1조원: 입력 오류 방지용 상한
MIN_BASE_RATE = -1.0
MAX_BASE_RATE = 20.0

# 금액 단위: 국토교통부 원본은 '만원' 단위 → 내부는 '원(KRW)' 정수
MANWON_TO_KRW = 10_000

# ---------------------------------------------------------------------------
# 내부 표준 컬럼명
# ---------------------------------------------------------------------------

DISTRICT = "district"
CONTRACT_DATE = "contract_date"
AREA_M2 = "area_m2"
FLOOR = "floor"
BUILT_YEAR = "built_year"
PRICE_KRW = "price_krw"
BASE_RATE = "base_rate"
ANOMALY_LABEL = "anomaly_label"

# 선택 메타 컬럼
APARTMENT_NAME = "apartment_name"
DONG = "dong"  # 법정동
SOURCE = "source"  # 데이터 출처 식별자 (예: "molit_api", "synthetic_sample")

# 규제 정보(평탄화 컬럼, 0/1/null). 원천 규제 이력 테이블은 merge.attach_regulation 참고.
REG_SPECULATION_OVERHEATED = "reg_speculation_overheated"  # 투기과열지구
REG_ADJUSTMENT_TARGET = "reg_adjustment_target"  # 조정대상지역
REG_LAND_PERMIT = "reg_land_permit"  # 토지거래허가구역
REGULATION_COLUMNS: tuple[str, ...] = (
    REG_SPECULATION_OVERHEATED,
    REG_ADJUSTMENT_TARGET,
    REG_LAND_PERMIT,
)

# 파생 컬럼 (preprocess 에서 행 단위로 생성 — 데이터 누출 없음)
CONTRACT_YEAR = "contract_year"
CONTRACT_MONTH = "contract_month"
BUILDING_AGE = "building_age"
PRICE_PER_M2 = "price_per_m2"  # price_krw 에서 파생 → 회귀 입력 금지

# 기준가격 (UI/이상거래 판단용, 데이터 원천 컬럼 아님)
REFERENCE_PRICE_KRW = "reference_price_krw"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool
    unit: str
    description: str


TRANSACTION_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec(DISTRICT, "str", True, "-", "서울 자치구명 (SEOUL_DISTRICTS 중 하나)"),
    ColumnSpec(CONTRACT_DATE, "datetime64 (날짜)", True, "YYYY-MM-DD", "계약일"),
    ColumnSpec(AREA_M2, "float", True, "㎡", "전용면적"),
    ColumnSpec(FLOOR, "int", True, "층", "층수 (지하는 음수)"),
    ColumnSpec(PRICE_KRW, "int", True, "원", "실거래가격 (만원 × 10,000)"),
    ColumnSpec(BUILT_YEAR, "Int64 (nullable)", False, "년", "건축연도"),
    ColumnSpec(BASE_RATE, "float (nullable)", False, "%", "계약일 시점 유효 기준금리"),
    ColumnSpec(REG_SPECULATION_OVERHEATED, "Int64 (nullable)", False, "0/1", "투기과열지구 여부"),
    ColumnSpec(REG_ADJUSTMENT_TARGET, "Int64 (nullable)", False, "0/1", "조정대상지역 여부"),
    ColumnSpec(REG_LAND_PERMIT, "Int64 (nullable)", False, "0/1", "토지거래허가구역 여부"),
    ColumnSpec(ANOMALY_LABEL, "Int64 (nullable)", False, "0/1", "이상거래 레이블 (미확보 허용)"),
    ColumnSpec(APARTMENT_NAME, "str (nullable)", False, "-", "단지명"),
    ColumnSpec(DONG, "str (nullable)", False, "-", "법정동"),
    ColumnSpec(SOURCE, "str (nullable)", False, "-", "데이터 출처"),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(c.name for c in TRANSACTION_SCHEMA if c.required)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(c.name for c in TRANSACTION_SCHEMA if not c.required)

# 중복 거래 판단 키: 동일 단지·동일 날짜·동일 면적·동일 층·동일 가격이면 중복으로 본다.
# (apartment_name/dong 이 존재하면 키에 포함 — preprocess.drop_duplicate_transactions 참고)
DUPLICATE_KEY: tuple[str, ...] = (DISTRICT, CONTRACT_DATE, AREA_M2, FLOOR, PRICE_KRW)
DUPLICATE_KEY_OPTIONAL: tuple[str, ...] = (DONG, APARTMENT_NAME)

# ---------------------------------------------------------------------------
# 모델 입력 변수 / 정답값 구분
# ---------------------------------------------------------------------------

REGRESSION_TARGET = PRICE_KRW
CLASSIFICATION_TARGET = ANOMALY_LABEL

REGRESSION_NUMERIC_FEATURES: tuple[str, ...] = (
    AREA_M2,
    FLOOR,
    BUILT_YEAR,
    BUILDING_AGE,
    CONTRACT_YEAR,
    CONTRACT_MONTH,
    BASE_RATE,
    *REGULATION_COLUMNS,
)
REGRESSION_CATEGORICAL_FEATURES: tuple[str, ...] = (DISTRICT,)
REGRESSION_FEATURES: tuple[str, ...] = (
    *REGRESSION_NUMERIC_FEATURES,
    *REGRESSION_CATEGORICAL_FEATURES,
)

# 회귀 입력으로 절대 사용하면 안 되는 컬럼 (정답값 및 정답에서 파생된 값)
REGRESSION_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {PRICE_KRW, PRICE_PER_M2, ANOMALY_LABEL, REFERENCE_PRICE_KRW}
)

# 이상거래 분류: 실제 거래가격 사용 가능
CLASSIFICATION_NUMERIC_FEATURES: tuple[str, ...] = (
    *REGRESSION_NUMERIC_FEATURES,
    PRICE_KRW,
    PRICE_PER_M2,
)
CLASSIFICATION_CATEGORICAL_FEATURES: tuple[str, ...] = REGRESSION_CATEGORICAL_FEATURES
CLASSIFICATION_FEATURES: tuple[str, ...] = (
    *CLASSIFICATION_NUMERIC_FEATURES,
    *CLASSIFICATION_CATEGORICAL_FEATURES,
)
CLASSIFICATION_FORBIDDEN_FEATURES: frozenset[str] = frozenset({ANOMALY_LABEL})

ANOMALY_LABEL_VALUES: tuple[int, ...] = (0, 1)  # 0 = 정상, 1 = 이상거래 의심

# ---------------------------------------------------------------------------
# 국토교통부 원본 컬럼 → 내부 컬럼 매핑
# ---------------------------------------------------------------------------
# 1) 공공데이터포털 OpenAPI (아파트 매매 실거래가 상세 자료) 응답 필드명
# 2) 국토교통부 실거래가 공개시스템(rt.molit.go.kr) CSV 다운로드 컬럼명
# ※ 실제 수집 시 원본 필드명을 재확인하고 변경되면 이 매핑만 수정한다.
#   계약일은 년/월/일이 분리되어 있으므로 collect.normalize_molit_frame 에서 조합한다.

MOLIT_API_COLUMN_MAP: dict[str, str] = {
    "sggCd": "sgg_code",
    "umdNm": DONG,
    "aptNm": APARTMENT_NAME,
    "excluUseAr": AREA_M2,
    "dealYear": "deal_year",
    "dealMonth": "deal_month",
    "dealDay": "deal_day",
    "dealAmount": "deal_amount_manwon",
    "floor": FLOOR,
    "buildYear": BUILT_YEAR,
    "cdealType": "cancel_type",  # 해제여부
    "cdealDay": "cancel_date",  # 해제사유발생일
    "dealingGbn": "deal_type",  # 거래유형 (중개/직거래)
}

MOLIT_CSV_COLUMN_MAP: dict[str, str] = {
    "시군구": "sigungu",  # 예: "서울특별시 강남구 개포동" → district, dong 분리
    "단지명": APARTMENT_NAME,
    "전용면적(㎡)": AREA_M2,
    "계약년월": "deal_year_month",  # 예: 202301
    "계약일": "deal_day",
    "거래금액(만원)": "deal_amount_manwon",  # 예: "125,000"
    "층": FLOOR,
    "건축년도": BUILT_YEAR,
    "해제사유발생일": "cancel_date",
    "거래유형": "deal_type",
}


# ---------------------------------------------------------------------------
# 단건 레코드 검증용 Pydantic 모델 (API 입력과 공유)
# ---------------------------------------------------------------------------


class RegulationInfo(BaseModel):
    """계약일 시점의 규제 정보. 알 수 없으면 None."""

    model_config = ConfigDict(extra="forbid")

    speculation_overheated: bool | None = Field(None, description="투기과열지구 여부")
    adjustment_target: bool | None = Field(None, description="조정대상지역 여부")
    land_permit: bool | None = Field(None, description="토지거래허가구역 여부")

    def to_columns(self) -> dict[str, int | None]:
        def _b(v: bool | None) -> int | None:
            return None if v is None else int(v)

        return {
            REG_SPECULATION_OVERHEATED: _b(self.speculation_overheated),
            REG_ADJUSTMENT_TARGET: _b(self.adjustment_target),
            REG_LAND_PERMIT: _b(self.land_permit),
        }


class ApartmentFeatures(BaseModel):
    """거래 조건(가격 제외). 가격 예측·금리 시나리오·이상거래 판정의 공통 입력."""

    model_config = ConfigDict(extra="forbid")

    district: str = Field(..., description="서울 자치구명", examples=["강남구"])
    contract_date: date = Field(..., description="계약일 (YYYY-MM-DD)")
    area_m2: float = Field(..., ge=MIN_AREA_M2, le=MAX_AREA_M2, description="전용면적(㎡)")
    floor: int = Field(..., ge=MIN_FLOOR, le=MAX_FLOOR, description="층수")
    built_year: int | None = Field(None, ge=MIN_BUILT_YEAR, le=MAX_BUILT_YEAR)
    base_rate: float | None = Field(
        None, ge=MIN_BASE_RATE, le=MAX_BASE_RATE, description="기준금리(%)"
    )
    regulation: RegulationInfo | None = None

    @field_validator("district")
    @classmethod
    def _check_district(cls, v: str) -> str:
        v = v.strip()
        if v not in SEOUL_DISTRICTS:
            raise ValueError(f"서울 25개 자치구 중 하나여야 합니다: {v!r}")
        return v

    def to_record(self) -> dict[str, object]:
        """DataFrame 1행으로 변환할 수 있는 dict (내부 표준 컬럼명)."""
        record: dict[str, object] = {
            DISTRICT: self.district,
            CONTRACT_DATE: self.contract_date,
            AREA_M2: self.area_m2,
            FLOOR: self.floor,
            BUILT_YEAR: self.built_year,
            BASE_RATE: self.base_rate,
        }
        record.update((self.regulation or RegulationInfo()).to_columns())
        return record


class TransactionRecord(ApartmentFeatures):
    """실거래 1건 (가격 포함)."""

    price_krw: int = Field(..., ge=MIN_PRICE_KRW, le=MAX_PRICE_KRW, description="거래가격(원)")
    anomaly_label: Literal[0, 1] | None = Field(
        None, description="이상거래 레이블 (미확보 시 null)"
    )

    def to_record(self) -> dict[str, object]:
        record = super().to_record()
        record[PRICE_KRW] = self.price_krw
        record[ANOMALY_LABEL] = self.anomaly_label
        return record
