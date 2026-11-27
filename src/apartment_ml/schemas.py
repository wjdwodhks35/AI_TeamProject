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
MAX_NAME_LENGTH = 60  # 동명·단지 일련번호 등 짧은 문자열 입력 상한

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

# 전국 공통 규제. 자치구별로 갈리지 않고 시간축으로만 변하므로 구간 테이블이 아니라
# 월별 시계열로 다룬다 (collect.to_national_regulation_state).
# 0/1 이 아닌 값이 있다: DSR 단계는 0~3, 주담대 한도는 억 단위 금액이다.
REG_DSR_STAGE = "reg_dsr_stage"  # 차주단위 DSR 단계 (0~3)
REG_STRESS_DSR = "reg_stress_dsr"  # 스트레스 DSR 단계 (0~2)
REG_MORTGAGE_CAP = "reg_mortgage_cap"  # 주담대 한도(억). 한도 없음 = MORTGAGE_CAP_NONE
REG_TRANSFER_TAX_HEAVY = "reg_transfer_tax_heavy"  # 양도소득세 중과
REG_POLICY_LOAN = "reg_policy_loan"  # 정책대출 (완화 방향)
REG_MATURITY_LIMIT = "reg_maturity_limit"  # 다주택자 만기연장 제한
MORTGAGE_CAP_NONE = 99  # 한도 규제가 없는 상태를 나타내는 값

NATIONAL_REGULATION_COLUMNS: tuple[str, ...] = (
    REG_DSR_STAGE,
    REG_STRESS_DSR,
    REG_MORTGAGE_CAP,
    REG_TRANSFER_TAX_HEAVY,
    REG_POLICY_LOAN,
    REG_MATURITY_LIMIT,
)
NATIONAL_REGULATION_DEFAULTS: dict[str, int] = {
    REG_DSR_STAGE: 0,
    REG_STRESS_DSR: 0,
    REG_MORTGAGE_CAP: MORTGAGE_CAP_NONE,
    REG_TRANSFER_TAX_HEAVY: 0,
    REG_POLICY_LOAN: 0,
    REG_MATURITY_LIMIT: 0,
}

# 규제 점수: 그 시점에 걸려 있는 규제들의 가중합.
# 누적 합계로 만들면 값이 한 방향으로만 움직여서 경과 시간을 그대로 학습한다.
# 그래서 '지금 걸려 있는 것' 만 더하는 상태 기준으로 계산한다.
# 부호는 집값에 주는 방향이고 크기는 우리가 정한 값이다 (팀 합의 필요 — docs/DATA_SCHEMA.md).
REG_SCORE = "reg_score"
REG_SCORE_CHANGE = "reg_score_change"  # 전월 대비 변화 (자치구별)
REGULATION_SCORE_COLUMNS: tuple[str, ...] = (REG_SCORE, REG_SCORE_CHANGE)

REGULATION_WEIGHTS: dict[str, float] = {
    # 투기과열지구는 조정대상지역과 겹쳐 지정되므로 둘이 합쳐 -2.0 이 된다.
    REG_ADJUSTMENT_TARGET: -1.0,
    REG_SPECULATION_OVERHEATED: -1.0,
    REG_LAND_PERMIT: -2.0,  # 거래 자체를 막으니 가장 세게
    REG_DSR_STAGE: -0.5,  # 단계당
    REG_STRESS_DSR: -0.5,  # 단계당
    REG_TRANSFER_TAX_HEAVY: -1.0,
    REG_MATURITY_LIMIT: -0.5,
    REG_POLICY_LOAN: +1.0,  # 완화라서 양수
}
# 주담대 한도는 억 단위 금액이라 단계로 더할 수 없다. 낮은 한도부터 먼저 본다.
MORTGAGE_CAP_WEIGHTS: tuple[tuple[int, float], ...] = ((4, -2.0), (6, -1.0))

# 파생 컬럼 (preprocess 에서 행 단위로 생성 — 데이터 누출 없음)
CONTRACT_YEAR = "contract_year"
CONTRACT_MONTH = "contract_month"
BUILDING_AGE = "building_age"
PRICE_PER_M2 = "price_per_m2"  # price_krw 에서 파생 → 회귀 입력 금지

# 기준가격 (UI/이상거래 판단용, 데이터 원천 컬럼 아님)
REFERENCE_PRICE_KRW = "reference_price_krw"

# ---------------------------------------------------------------------------
# 거래 특성 (국토교통부 상세 자료에서 받을 수 있는 항목)
# ---------------------------------------------------------------------------
# 상세 자료에는 단지 일련번호·거래유형·중개사 소재지·토지임대부 여부가 들어 있다.
# 실제 수집 데이터(서울 28만건)에서 이상거래와 뚜렷하게 연관된 항목들이다.

COMPLEX_ID = "complex_id"  # 단지 일련번호(aptSeq). 같은 이름의 다른 단지를 구분한다
IS_DIRECT_DEAL = "is_direct_deal"  # 직거래 1 / 중개거래 0
AGENT_LOCATION = "agent_location"  # 0 소재지 없음 / 1 같은 자치구 / 2 다른 자치구
LAND_LEASEHOLD = "land_leasehold"  # 토지임대부 1. 땅값이 빠져 같은 평형이라도 훨씬 싸다
DEAL_FEATURE_COLUMNS: tuple[str, ...] = (IS_DIRECT_DEAL, AGENT_LOCATION, LAND_LEASEHOLD)

# ---------------------------------------------------------------------------
# 기준가 비교 (이상거래 레이블의 재료)
# ---------------------------------------------------------------------------
# 같은 단지·평형의 '직전 기간 시세'를 기준으로 삼아 실제 거래가가 얼마나 벗어났는지 본다.
# REFERENCE_PRICE_KRW(회귀 모델의 예상가)와는 다른 개념이다.
# peer_tier 가 클수록 넓은 그룹에서 만든 기준이라 그 자체가 흔들린다.

PEER_MEDIAN_PRICE_KRW = "peer_median_price_krw"  # 직전 기간 같은 조건의 시세 중앙값
PEER_TIER = "peer_tier"  # 1 단지+평형 / 2 단지 / 3 법정동+평형 / 4 자치구+평형 / 0 못 만듦
PRICE_RATIO = "price_ratio"  # price_krw / peer_median_price_krw

# ---------------------------------------------------------------------------
# 금리 변화량
# ---------------------------------------------------------------------------
# 금리는 '수준' 으로 넣으면 안 된다. 금리가 오르는 시기는 경기가 좋은 시기이기도 해서
# 수준끼리 비교하면 "금리가 높을 때 집값도 높다" 는 거꾸로 된 관계가 나온다.
# 월별 평균 평당가 기준 으로 기준금리 수준은 +0.209, 전년 대비 변화는 -0.752 였다 (68개월).

BASE_RATE_CHANGE_12M = "base_rate_change_12m"  # 12개월 전 대비 변화(%포인트)
BASE_RATE_CHANGE_3M = "base_rate_change_3m"  # 3개월 전 대비 변화(%포인트)
BASE_RATE_CHANGE_COLUMNS: tuple[str, ...] = (BASE_RATE_CHANGE_12M, BASE_RATE_CHANGE_3M)

# 기준금리 하나로는 부족하다. 가계가 실제로 부담하는 금리는 기준금리와 다르게 움직이고,
# 특히 대출금리는 '6개월 전의 변화' 가 지금 가격과 더 강하게 맞물린다.
# 월별 평균 평당가 기준 전년 대비 변화와의 상관 (동시점 -> 6개월 선행):
#   기준금리 -0.752 -> -0.632,  주담대 -0.435 -> -0.737,  국고채3년 -0.557 -> -0.718
# 선행 변수를 쓸지는 모델 담당자가 정한다. 여기서는 동시점과 3개월 변화만 붙인다.
MORTGAGE_RATE = "mortgage_rate"  # 주택담보대출
MORTGAGE_RATE_FIXED = "mortgage_rate_fixed"  # 주택담보대출 고정형
MORTGAGE_RATE_VARIABLE = "mortgage_rate_variable"  # 주택담보대출 변동형
GROUP_LOAN_RATE = "group_loan_rate"  # 집단대출
JEONSE_LOAN_RATE = "jeonse_loan_rate"  # 전세자금대출
HOUSEHOLD_LOAN_RATE = "household_loan_rate"  # 가계대출 전체
TREASURY_3Y = "treasury_3y"  # 국고채 3년
TREASURY_10Y = "treasury_10y"  # 국고채 10년
CD_91D = "cd_91d"  # CD 91일

# (내부 컬럼명, 수집 JSON 의 name 값). 수집 스크립트가 쓰는 한글 이름과 짝지어 둔다.
RATE_SERIES: tuple[tuple[str, str], ...] = (
    (BASE_RATE, "기준금리"),
    (MORTGAGE_RATE, "주담대"),
    (MORTGAGE_RATE_FIXED, "주담대_고정"),
    (MORTGAGE_RATE_VARIABLE, "주담대_변동"),
    (GROUP_LOAN_RATE, "집단대출"),
    (JEONSE_LOAN_RATE, "전세자금대출"),
    (HOUSEHOLD_LOAN_RATE, "가계대출"),
    (TREASURY_3Y, "국고채3년"),
    (TREASURY_10Y, "국고채10년"),
    (CD_91D, "CD91일"),
)
RATE_COLUMNS: tuple[str, ...] = tuple(col for col, _ in RATE_SERIES)
RATE_NAME_TO_COLUMN: dict[str, str] = {name: col for col, name in RATE_SERIES}
RATE_CHANGE_MONTHS: tuple[int, ...] = (12, 3)


def rate_change_column(rate_column: str, months: int) -> str:
    """금리 변화량 컬럼명. (base_rate, 12) -> base_rate_change_12m"""
    return f"{rate_column}_change_{months}m"


RATE_CHANGE_COLUMNS: tuple[str, ...] = tuple(
    rate_change_column(col, months) for col in RATE_COLUMNS for months in RATE_CHANGE_MONTHS
)


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
    ColumnSpec(COMPLEX_ID, "str (nullable)", False, "-", "단지 일련번호(aptSeq)"),
    ColumnSpec(IS_DIRECT_DEAL, "Int64 (nullable)", False, "0/1", "직거래 여부"),
    ColumnSpec(
        AGENT_LOCATION, "Int64 (nullable)", False, "0/1/2", "중개사 소재지 (없음/관내/관외)"
    ),
    ColumnSpec(LAND_LEASEHOLD, "Int64 (nullable)", False, "0/1", "토지임대부 여부"),
    ColumnSpec(BASE_RATE_CHANGE_12M, "float (nullable)", False, "%p", "기준금리 전년 대비 변화"),
    ColumnSpec(BASE_RATE_CHANGE_3M, "float (nullable)", False, "%p", "기준금리 3개월 대비 변화"),
    # 기준금리를 제외한 9종의 수준
    *(
        ColumnSpec(col, "float (nullable)", False, "%", f"{name} 금리 수준")
        for col, name in RATE_SERIES
        if col != BASE_RATE
    ),
    # 기준금리를 제외한 9종의 변화량
    *(
        ColumnSpec(
            rate_change_column(col, months),
            "float (nullable)",
            False,
            "%p",
            f"{name} {months}개월 대비 변화",
        )
        for col, name in RATE_SERIES
        for months in RATE_CHANGE_MONTHS
        if col != BASE_RATE
    ),
    ColumnSpec(REG_DSR_STAGE, "Int64 (nullable)", False, "단계", "차주단위 DSR 단계 (0~3)"),
    ColumnSpec(REG_STRESS_DSR, "Int64 (nullable)", False, "단계", "스트레스 DSR 단계 (0~2)"),
    ColumnSpec(
        REG_MORTGAGE_CAP,
        "Int64 (nullable)",
        False,
        "억원",
        f"주담대 한도 ({MORTGAGE_CAP_NONE} = 한도 규제 없음)",
    ),
    ColumnSpec(REG_TRANSFER_TAX_HEAVY, "Int64 (nullable)", False, "0/1", "양도소득세 중과"),
    ColumnSpec(REG_POLICY_LOAN, "Int64 (nullable)", False, "0/1", "정책대출 시행 (완화)"),
    ColumnSpec(REG_MATURITY_LIMIT, "Int64 (nullable)", False, "0/1", "만기연장 제한"),
    ColumnSpec(REG_SCORE, "float (nullable)", False, "점", "그 시점 규제 가중합"),
    ColumnSpec(REG_SCORE_CHANGE, "float (nullable)", False, "점", "규제 점수 전월 대비 변화"),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(c.name for c in TRANSACTION_SCHEMA if c.required)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(c.name for c in TRANSACTION_SCHEMA if not c.required)

# 중복 거래 판단 키: 동일 단지·동일 날짜·동일 면적·동일 층·동일 가격이면 중복으로 본다.
# (apartment_name/dong 이 존재하면 키에 포함 — preprocess.drop_duplicate_transactions 참고)
DUPLICATE_KEY: tuple[str, ...] = (DISTRICT, CONTRACT_DATE, AREA_M2, FLOOR, PRICE_KRW)
# complex_id 가 있으면 단지를 정확히 구분한다 (같은 이름의 다른 단지가 511개 있었다).
DUPLICATE_KEY_OPTIONAL: tuple[str, ...] = (DONG, APARTMENT_NAME, COMPLEX_ID)

# ---------------------------------------------------------------------------
# 학습 데이터에서 빼야 하는 원본 컬럼
# ---------------------------------------------------------------------------
# 국토교통부 상세 자료에는 계약 시점에 존재하지 않는 칸이 섞여 있다. 자치구별 CSV
# (data/processed/transactions_*.csv) 에는 원본 확인용으로 남겨 두고, 학습용 한 장의
# 표에서는 뺀다. scripts/build_dataset.py 가 이 목록을 사용한다.

# 값이 비어 있는지만 봐도 해제 여부가 드러나는 칸. 해제된 거래는 등기를 하지 않고
# 동 정보도 붙지 않는다. 실제 수집 데이터에서 rgstDate 는 해제 거래의 0.0%,
# 정상 거래의 79.1% 에 있었고 aptDong 은 0.0% 대 72.5% 였다.
LABEL_LEAKING_RAW_COLUMNS: tuple[str, ...] = ("rgstDate", "aptDong")

# 2024년부터만 제공되는 칸. 값이 있다는 것 자체가 '최근 거래' 라는 뜻이 되어
# 계약일을 빼도 시점 정보가 새어 들어간다.
TIME_LEAKING_RAW_COLUMNS: tuple[str, ...] = ("slerGbn", "buyerGbn")

# 정답값·계약일을 그대로 다시 담고 있는 칸. price_krw, contract_date 를 빼고
# 나머지 전부로 학습하면 이 칸들을 통해 정답이 들어간다.
DUPLICATE_TARGET_RAW_COLUMNS: tuple[str, ...] = (
    "deal_amount_manwon",
    "deal_year",
    "deal_month",
    "deal_day",
)

# 주소 계열 원본. 자치구·법정동·단지 일련번호로 이미 표현되어 있다.
UNUSED_RAW_COLUMNS: tuple[str, ...] = (
    "roadNm",
    "roadNmBonbun",
    "roadNmBubun",
    "roadNmCd",
    "roadNmSeq",
    "roadNmSggCd",
    "roadNmbCd",
    "bonbun",
    "bubun",
    "jibun",
    "landCd",
    "umdCd",
)

# 해제 정보. 해제 거래를 지운 기본 설정에서는 전부 비어 있어 의미가 없고, 남겨 둔
# 경우(--keep-cancelled) 에는 레이블 검증에 쓰므로 그때만 유지한다.
CANCELLATION_RAW_COLUMNS: tuple[str, ...] = ("cancel_type", "cancel_date")

# 해제 여부 (레이블 검증용). 계약 이후에 생기는 사실이므로 입력 변수로 쓰면 안 된다.
# 해제 쌍은 중복 제거에서 한 줄로 묶이므로, 묶이기 전에 양쪽에 표시해 둬야 수가 맞는다.
WAS_CANCELLED = "was_cancelled"

TRAINING_DROP_COLUMNS: tuple[str, ...] = (
    *LABEL_LEAKING_RAW_COLUMNS,
    *TIME_LEAKING_RAW_COLUMNS,
    *DUPLICATE_TARGET_RAW_COLUMNS,
    *UNUSED_RAW_COLUMNS,
)


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
    # 금리는 변화량을 쓴다. 수준은 데이터에는 남겨 두지만 기본 입력에서 뺀다 —
    # 금리가 오르는 시기는 경기가 좋은 시기이기도 해서 수준끼리 비교하면 부호가 뒤집힌다.
    # base_rate 수준만 예외로 남긴다 (scenarios.interest_rate 가 이 컬럼을 요구한다).
    BASE_RATE,
    *RATE_CHANGE_COLUMNS,
    *REGULATION_COLUMNS,
    *NATIONAL_REGULATION_COLUMNS,
    # 규제 점수는 플래그의 가중합이라 정보가 겹치지만, 68개월에 규제 이벤트가 20건뿐이어서
    # 각 플래그의 부호를 데이터에서 학습하기엔 표본이 적다. 방향을 미리 넣어 주는 값이다.
    *REGULATION_SCORE_COLUMNS,
    *DEAL_FEATURE_COLUMNS,
)
REGRESSION_CATEGORICAL_FEATURES: tuple[str, ...] = (DISTRICT,)
# 위치 변수 — **회귀 전용**. 범주로 넣지 않고 '그 위치의 과거 로그 평당가'로 요약해
# 쓴다 (models/location_price.py). 동이 332개, 단지가 7,600개가 넘어 원-핫으로는
# 변수가 폭발하고, HistGradientBoosting 의 범주형 직접 처리는 255개까지만 된다.
# 분류에는 넣지 않는다: 분류 입력에는 price_krw 가 있어서 단지 수준 기준가가 함께
# 들어가면 price_ratio 를 역산해 이상거래 레이블을 그대로 복원할 수 있다.
REGRESSION_LOCATION_FEATURES: tuple[str, ...] = (DISTRICT, DONG, COMPLEX_ID)
REGRESSION_FEATURES: tuple[str, ...] = (
    *REGRESSION_NUMERIC_FEATURES,
    *REGRESSION_CATEGORICAL_FEATURES,
)

# 회귀 입력으로 절대 사용하면 안 되는 컬럼 (정답값 및 정답에서 파생된 값)
REGRESSION_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        PRICE_KRW,
        PRICE_PER_M2,
        ANOMALY_LABEL,
        WAS_CANCELLED,
        REFERENCE_PRICE_KRW,
        PEER_MEDIAN_PRICE_KRW,
        PRICE_RATIO,
        PEER_TIER,
    }
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
# 이상거래 레이블은 price_ratio = price_krw / peer_median_price_krw 로 만든다.
# 분류에서는 price_krw 를 입력으로 쓸 수 있으므로, peer_median_price_krw 가 함께 들어가면
# 레이블을 그대로 역산할 수 있다. peer_tier 도 넓은 단계일수록 이상 판정이 많아 부분 누출이다.
CLASSIFICATION_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {ANOMALY_LABEL, PEER_MEDIAN_PRICE_KRW, PRICE_RATIO, PEER_TIER, WAS_CANCELLED}
)

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
    "aptSeq": COMPLEX_ID,  # 단지 일련번호 (예: "11110-2234")
    "estateAgentSggNm": "agent_district",  # 중개사 소재지 (예: "서울 종로구")
    "landLeaseholdGbn": "land_leasehold_raw",  # 토지임대부 여부 (Y/N)
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


def _blank_to_none(value: str | None) -> str | None:
    """빈 문자열은 '값 없음' 과 같게 다룬다.

    프론트의 select 는 미선택을 "" 로 보내는 일이 흔하고, 그대로 두면 위치 인코더가
    ''를 별도 범주로 보게 된다.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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
    # 동·단지는 회귀의 위치 인코딩 입력이다. 없으면 자치구 수준 추정으로 떨어지는데,
    # 같은 모델로 test 를 재면 MAE 1.41억 -> 4.18억, R2 0.935 -> 0.499 로 기준선(0.603)
    # 보다도 낮아진다. 그래서 선택 입력이지만 **되도록 받아야 하는** 값이다.
    # `complex_id` 는 국토부 단지 일련번호(aptSeq)이고, 고를 수 있는 목록은
    # GET /locations/{자치구} 가 내려준다 (models/complex_directory.py).
    dong: str | None = Field(
        None, max_length=MAX_NAME_LENGTH, description="법정동 (예: 대치동)", examples=["대치동"]
    )
    complex_id: str | None = Field(
        None,
        max_length=MAX_NAME_LENGTH,
        description="단지 일련번호(aptSeq). GET /locations/{자치구} 참고",
        examples=["11680-2234"],
    )
    # 이상거래 규칙이 쓰는 값. 이상으로 확인된 건에 직거래가 몰려 있다
    # (전체 2.74% vs 1단계 싸게 67%).
    is_direct_deal: bool | None = Field(
        None, description="중개 없이 직접 거래했는지. 모르면 비워 둡니다."
    )

    @field_validator("district")
    @classmethod
    def _check_district(cls, v: str) -> str:
        v = v.strip()
        if v not in SEOUL_DISTRICTS:
            raise ValueError(f"서울 25개 자치구 중 하나여야 합니다: {v!r}")
        return v

    def to_record(self) -> dict[str, object]:
        """DataFrame 1행으로 변환할 수 있는 dict (내부 표준 컬럼명)."""
        dong = _blank_to_none(self.dong)
        complex_id = _blank_to_none(self.complex_id)
        record: dict[str, object] = {
            DISTRICT: self.district,
            CONTRACT_DATE: self.contract_date,
            AREA_M2: self.area_m2,
            FLOOR: self.floor,
            BUILT_YEAR: self.built_year,
            BASE_RATE: self.base_rate,
            # 단지가 없으면 동도 쓰지 않는다. 측정값: 동·단지 모두 없음 MAE 4.18억,
            # 동만 있고 단지 없음 4.85억 — 학습 때 본 적 없는 조합이라 오히려 더 나쁘다.
            DONG: dong if complex_id else None,
            COMPLEX_ID: complex_id,
            IS_DIRECT_DEAL: None if self.is_direct_deal is None else int(self.is_direct_deal),
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
