"""데이터 수집 인터페이스.

Phase 0 범위: 수집기 **계약(인터페이스)** 과 원본 → 내부 컬럼 정규화 함수만 제공한다.
실제 외부 API 호출(국토교통부, ECOS, 정책 이력)은 ``feature/data`` 담당자가 구현한다.

규칙
----
- 수집 결과는 ``data/raw/`` 에 원본 그대로 저장하고, 정규화 결과는 ``data/processed/`` 에 저장한다.
- 테스트에서는 외부 네트워크를 호출하지 않는다. (수집기는 네트워크 없이 생성 가능해야 함)
- API 키는 ``config.get_api_key`` 로만 읽는다.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import manwon_to_krw

# 국토교통부 API 의 지역코드(LAWD_CD, 시군구 5자리) — 서울 25개 자치구
SEOUL_LAWD_CODES: dict[str, str] = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}  # fmt: skip


@runtime_checkable
class TransactionCollector(Protocol):
    """아파트 실거래 수집기 계약. 반환값은 원본 컬럼 그대로의 DataFrame."""

    def fetch(self, lawd_code: str, year_month: str) -> pd.DataFrame:
        """lawd_code: 시군구 5자리, year_month: 'YYYYMM'."""
        ...


@runtime_checkable
class BaseRateCollector(Protocol):
    """기준금리 수집기 계약. 반환 컬럼: ``effective_date``(datetime), ``base_rate``(float, %)."""

    def fetch(self, start: date, end: date) -> pd.DataFrame: ...


@runtime_checkable
class RegulationCollector(Protocol):
    """규제 이력 수집기 계약. 반환 컬럼은 ``merge.REGULATION_HISTORY_COLUMNS`` 참고."""

    def fetch(self) -> pd.DataFrame: ...


class MolitApiCollector:
    """국토교통부 아파트 매매 실거래가 API 수집기 (미구현 — feature/data 담당)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch(self, lawd_code: str, year_month: str) -> pd.DataFrame:
        raise NotImplementedError("MOLIT API 수집은 feature/data 브랜치에서 구현 예정입니다.")


class EcosBaseRateCollector:
    """한국은행 ECOS 기준금리 수집기 (미구현 — feature/data 담당)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("ECOS 수집은 feature/data 브랜치에서 구현 예정입니다.")


def normalize_molit_api_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """MOLIT OpenAPI 응답(DataFrame) → 내부 표준 컬럼.

    - 년/월/일을 ``contract_date`` 로 조합
    - 거래금액(만원, 쉼표 포함) → ``price_krw``(원)
    - 시군구코드 → 자치구명
    - 해제(취소)된 거래는 제거 (cancel_type 값이 있는 행)
    타입 변환·유효성 검사는 ``preprocess.preprocess_transactions`` 에서 수행한다.
    """
    df = raw.rename(columns=S.MOLIT_API_COLUMN_MAP).copy()
    needed = ["deal_year", "deal_month", "deal_day", "deal_amount_manwon", "sgg_code"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"MOLIT 원본 컬럼이 없습니다: {missing}")

    if "cancel_type" in df.columns:
        cancelled = df["cancel_type"].astype("string").str.strip().fillna("").ne("")
        df = df.loc[~cancelled].copy()

    y = df["deal_year"].astype("string").str.strip()
    m = df["deal_month"].astype("string").str.strip().str.zfill(2)
    d = df["deal_day"].astype("string").str.strip().str.zfill(2)
    df[S.CONTRACT_DATE] = y + "-" + m + "-" + d
    df[S.PRICE_KRW] = manwon_to_krw(df["deal_amount_manwon"])
    df[S.DISTRICT] = df["sgg_code"].astype("string").str.strip().map(SEOUL_LAWD_CODES)
    df[S.SOURCE] = "molit_api"
    return df
