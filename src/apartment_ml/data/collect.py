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

import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

import pandas as pd
import requests

from apartment_ml import schemas as S
from apartment_ml.data.merge import REGULATION_HISTORY_COLUMNS
from apartment_ml.data.preprocess import manwon_to_krw


class CollectError(RuntimeError):
    """외부 API 응답이 비정상일 때."""


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


def month_range(start: str, end: str) -> list[str]:
    """'202101', '202608' -> ['202101', '202102', ...]. 양쪽 끝을 포함한다."""
    for label, value in (("start", start), ("end", end)):
        if not (len(value) == 6 and value.isdigit()):
            raise ValueError(f"{label} 는 'YYYYMM' 형식이어야 합니다: {value!r}")
    if start > end:
        raise ValueError(f"start 가 end 보다 늦습니다: {start} > {end}")

    year, month = int(start[:4]), int(start[4:])
    end_year, end_month = int(end[:4]), int(end[4:])
    months: list[str] = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year}{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def raw_month_path(raw_dir: str | Path, lawd_code: str, year_month: str) -> Path:
    """수집 결과를 저장할 경로. 자치구별 폴더 아래 한 달에 한 파일."""
    if lawd_code not in SEOUL_LAWD_CODES:
        raise ValueError(f"서울 자치구 코드가 아닙니다: {lawd_code!r}")
    folder = f"{lawd_code}_{SEOUL_LAWD_CODES[lawd_code]}"
    return Path(raw_dir) / folder / f"{year_month}.json"


def parse_molit_xml(xml_text: str) -> tuple[list[dict[str, str]], int]:
    """응답 XML 에서 거래 목록과 전체 건수를 뽑는다.

    필드 이름을 하드코딩하지 않고 ``item`` 안의 태그를 모두 가져온다.
    국토교통부가 항목을 추가해도 수집 코드를 고치지 않아도 되고,
    이름을 잘못 적어서 특정 칸이 조용히 비는 일도 없다.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise CollectError(f"XML 이 아닌 응답: {xml_text[:200]}") from exc

    code = root.findtext(".//resultCode")
    if code is not None and code.strip() not in ("00", "000"):
        message = (root.findtext(".//resultMsg") or "").strip()
        raise CollectError(f"국토교통부 API 오류 resultCode={code.strip()} {message}")

    items: list[dict[str, str]] = []
    for item in root.iter("item"):
        row = {child.tag: (child.text or "").strip() for child in item}
        if row:
            items.append(row)

    # 공공데이터포털은 장애 시 XML 대신 HTML 오류 페이지를 돌려줄 때가 있다.
    # 그대로 0건으로 넘기면 그 달이 조용히 비어버리므로 여기서 걸러낸다.
    if code is None and not items:
        raise CollectError(f"응답 형식이 다릅니다: {xml_text[:200]}")

    total = (root.findtext(".//totalCount") or "").strip()
    return items, int(total) if total.isdigit() else len(items)


class MolitApiCollector:
    """국토교통부 아파트 매매 실거래가 상세 자료 수집기.

    ``api_key`` 는 공공데이터포털의 **디코딩 키**를 넣는다. requests 가 쿼리
    문자열을 인코딩하므로 인코딩 키를 넣으면 두 번 인코딩되어 인증에 실패한다.

    한 번 호출로 한 자치구의 한 달을 받는다. 전체 기간 수집과 파일 저장은
    ``scripts/collect_molit.py`` 가 담당한다.
    """

    BASE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

    def __init__(
        self,
        api_key: str,
        *,
        rows_per_page: int = 1000,
        timeout: float = 20.0,
        max_retries: int = 3,
        sleep_seconds: float = 0.15,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("국토교통부 API 키가 비어 있습니다.")
        self.api_key = api_key.strip()
        self.rows_per_page = rows_per_page
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def fetch(self, lawd_code: str, year_month: str) -> pd.DataFrame:
        """lawd_code: 시군구 5자리, year_month: 'YYYYMM'. 원본 컬럼 그대로 반환."""
        if lawd_code not in SEOUL_LAWD_CODES:
            raise ValueError(f"서울 자치구 코드가 아닙니다: {lawd_code!r}")
        if not (len(year_month) == 6 and year_month.isdigit()):
            raise ValueError(f"year_month 는 'YYYYMM' 형식이어야 합니다: {year_month!r}")

        items, total = self._request(lawd_code, year_month, page=1)
        page = 1
        while len(items) < total:
            page += 1
            time.sleep(self.sleep_seconds)
            more, _ = self._request(lawd_code, year_month, page=page)
            if not more:
                break
            items.extend(more)
        return pd.DataFrame(items)

    def _request(
        self, lawd_code: str, year_month: str, page: int
    ) -> tuple[list[dict[str, str]], int]:
        params = {
            "serviceKey": self.api_key,
            "LAWD_CD": lawd_code,
            "DEAL_YMD": year_month,
            "pageNo": page,
            "numOfRows": self.rows_per_page,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                response.encoding = "utf-8"
                return parse_molit_xml(response.text)
            except CollectError:
                raise  # 키·한도 문제는 다시 요청해도 같은 결과다
            except Exception as exc:  # noqa: BLE001 - 네트워크 오류는 재시도 대상
                last_error = exc
                time.sleep(self.sleep_seconds * (attempt + 1) * 10)
        raise CollectError(f"{lawd_code} {year_month} 수집 실패: {last_error}")


class EcosBaseRateCollector:
    """한국은행 ECOS 기준금리 수집기.

    기준금리는 금융통화위원회 결정일에 바뀌므로 **일별로 받아 값이 바뀐 날만** 남긴다.
    월별로 받으면 실제 적용일을 알 수 없어, 계약일 시점의 금리를 붙이는
    ``merge.attach_base_rate`` 의 시점 정합이 한 달 단위로 거칠어진다.
    """

    BASE_URL = "https://ecos.bok.or.kr/api"
    STAT_CODE = "722Y001"  # 한국은행 기준금리 및 여수신금리
    ITEM_CODE = "0101000"  # 한국은행 기준금리

    def __init__(self, api_key: str, *, timeout: float = 30.0, max_retries: int = 3) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("ECOS API 키가 비어 있습니다.")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.max_retries = max_retries

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        """``effective_date``(적용일), ``base_rate``(%) 두 컬럼. 값이 바뀐 날만 남긴다."""
        if start > end:
            raise ValueError(f"start 가 end 보다 늦습니다: {start} > {end}")
        rows = self._request(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        if not rows:
            raise CollectError(f"ECOS 에서 받은 기준금리가 없습니다: {start} ~ {end}")
        return to_base_rate_changes(rows)

    def _request(self, start: str, end: str) -> list[dict[str, Any]]:
        url = "/".join(
            [
                self.BASE_URL,
                "StatisticSearch",
                quote(self.api_key),
                "json",
                "kr",
                "1",
                "100000",
                self.STAT_CODE,
                "D",
                start,
                end,
                self.ITEM_CODE,
            ]
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001 - 네트워크 오류는 재시도 대상
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise CollectError(f"ECOS 요청 실패: {last_error}")

        if "RESULT" in payload:  # 인증키 오류 등
            result = payload["RESULT"]
            raise CollectError(f"ECOS 오류 {result.get('CODE')}: {result.get('MESSAGE')}")
        if "StatisticSearch" not in payload:
            raise CollectError(f"예상과 다른 ECOS 응답: {str(payload)[:200]}")
        return payload["StatisticSearch"].get("row", [])


def to_base_rate_changes(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """ECOS 일별 응답 → 금리가 바뀐 날만 담은 (effective_date, base_rate) 표."""
    records = []
    for row in rows:
        value = row.get("DATA_VALUE")
        when = row.get("TIME")
        if value in (None, "") or when in (None, ""):
            continue
        records.append({"effective_date": str(when), "base_rate": float(value)})
    if not records:
        raise CollectError("ECOS 응답에 값이 있는 날이 없습니다.")

    df = pd.DataFrame(records)
    df["effective_date"] = pd.to_datetime(df["effective_date"], format="%Y%m%d")
    df = df.sort_values("effective_date").reset_index(drop=True)
    changed = df["base_rate"].ne(df["base_rate"].shift())  # 첫 행은 NaN 과 비교되어 True
    return df.loc[changed].reset_index(drop=True)


def _cancelled_mask(df: pd.DataFrame) -> pd.Series:
    """``cancel_type`` 에 값이 있으면 해제된 거래."""
    return df["cancel_type"].astype("string").str.strip().fillna("").ne("")


def _drop_cancelled_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """해제된 거래를 원본 행까지 함께 제거한다.

    국토교통부 자료는 계약이 해제되면 원본 행을 그대로 남기고 해제 행을 한 줄 더 넣는다.
    그래서 해제 행만 지우면 원본 행이 남아 깨진 계약이 정상 거래로 보인다.
    같은 거래(자치구·계약일·면적·층·가격, 있으면 법정동·단지명까지)를 모두 지운다.
    """
    cancelled = _cancelled_mask(df)
    if not cancelled.any():
        return df
    key_columns = [c for c in (*S.DUPLICATE_KEY, *S.DUPLICATE_KEY_OPTIONAL) if c in df.columns]
    key = pd.MultiIndex.from_frame(df[key_columns].astype("string").fillna(""))
    return df.loc[~key.isin(key[cancelled.to_numpy()])].copy()


def normalize_molit_api_frame(raw: pd.DataFrame, *, drop_cancelled: bool = True) -> pd.DataFrame:
    """MOLIT OpenAPI 응답(DataFrame) → 내부 표준 컬럼.

    - 년/월/일을 ``contract_date`` 로 조합
    - 거래금액(만원, 쉼표 포함) → ``price_krw``(원)
    - 시군구코드 → 자치구명

    해제(취소)된 거래
    ----------------
    ``drop_cancelled=True`` (기본) 는 해제된 거래를 **원본 행까지 함께** 제거한다.
    ``False`` 는 모두 남기므로 ``cancel_type`` 으로 해제 여부를 구분할 수 있다.
    이상거래 레이블이 실제 해제와 얼마나 겹치는지 확인할 때 쓴다.

    타입 변환·유효성 검사는 ``preprocess.preprocess_transactions`` 에서 수행한다.
    """
    df = raw.rename(columns=S.MOLIT_API_COLUMN_MAP).copy()
    needed = ["deal_year", "deal_month", "deal_day", "deal_amount_manwon", "sgg_code"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"MOLIT 원본 컬럼이 없습니다: {missing}")

    y = df["deal_year"].astype("string").str.strip()
    m = df["deal_month"].astype("string").str.strip().str.zfill(2)
    d = df["deal_day"].astype("string").str.strip().str.zfill(2)
    df[S.CONTRACT_DATE] = y + "-" + m + "-" + d
    df[S.PRICE_KRW] = manwon_to_krw(df["deal_amount_manwon"])
    df[S.DISTRICT] = df["sgg_code"].astype("string").str.strip().map(SEOUL_LAWD_CODES)
    df[S.SOURCE] = "molit_api"

    # 키에 계약일·가격이 들어가므로 파생 컬럼을 만든 뒤에 지운다.
    if drop_cancelled and "cancel_type" in df.columns:
        df = _drop_cancelled_pairs(df)
    return df


# ---------------------------------------------------------------------------
# 규제 이력
# ---------------------------------------------------------------------------
# 규제는 공개 API 가 없어서 고시를 직접 찾아 ``configs/regulation_events.csv`` 에 적는다.
# 그 파일의 한 줄은 "이 날부터 이 지역의 이 규제가 이 값이 된다" 는 뜻이고,
# 여기서 ``merge.attach_regulation`` 이 받는 기간 테이블로 바꾼다.

REGULATION_EVENTS_PATH = Path("configs/regulation_events.csv")
REGULATION_EVENT_COLUMNS = (
    "effective_date",
    "districts",
    "regulation_type",
    "value",
    "description",
    "source",
)
DISTRICTS_ALL = "ALL"
DISTRICTS_EXCEPT = "EXCEPT:"


def parse_districts(text: str) -> list[str]:
    """``districts`` 칸을 자치구 목록으로 바꾼다.

    - ``ALL``                      서울 25개 구 전체
    - ``강남구|송파구``             열거한 구만
    - ``EXCEPT:강남구|서초구``      열거한 구만 빼고 전체
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("districts 칸이 비어 있습니다.")
    if text == DISTRICTS_ALL:
        return list(S.SEOUL_DISTRICTS)

    inverted = text.startswith(DISTRICTS_EXCEPT)
    if inverted:
        text = text[len(DISTRICTS_EXCEPT) :]
    names = [n.strip() for n in text.split("|") if n.strip()]
    unknown = [n for n in names if n not in S.SEOUL_DISTRICTS]
    if unknown:
        raise ValueError(f"서울 자치구가 아닙니다: {unknown}")
    if inverted:
        return [d for d in S.SEOUL_DISTRICTS if d not in names]
    return names


def load_regulation_events(path: str | Path = REGULATION_EVENTS_PATH) -> pd.DataFrame:
    """규제 이벤트 CSV 를 읽는다.

    ``regulation_type`` 이 빈 줄은 근거 기록용이다. 전국 공통 규제(DSR, 대출 한도 등)는
    자치구별로 값이 다르지 않아 현재 스키마의 규제 컬럼으로 표현할 수 없으므로,
    파일에는 남겨 두고 계산에서는 건너뛴다.
    """
    events = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = [c for c in REGULATION_EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"규제 이벤트 파일에 컬럼이 없습니다: {missing}")

    used = events.loc[events["regulation_type"].str.strip() != ""].copy()
    unknown = set(used["regulation_type"].str.strip()) - set(S.REGULATION_COLUMNS)
    if unknown:
        raise ValueError(f"알 수 없는 regulation_type: {sorted(unknown)}")

    used["effective_date"] = pd.to_datetime(used["effective_date"], format="%Y-%m-%d")
    used["value"] = used["value"].astype(int)
    if not used["value"].isin((0, 1)).all():
        raise ValueError("value 는 0 또는 1 이어야 합니다.")
    return used.sort_values("effective_date").reset_index(drop=True)


def to_regulation_history(events: pd.DataFrame) -> pd.DataFrame:
    """규제 이벤트 → ``merge.REGULATION_HISTORY_COLUMNS`` 기간 테이블.

    자치구·규제별로 값이 1이 되는 날부터 0이 되는 날까지를 한 구간으로 만든다.
    끝까지 1이면 ``effective_to`` 를 비워 현재까지 유효한 것으로 둔다.
    """
    rows: list[dict[str, object]] = []
    for regulation in sorted(set(events["regulation_type"])):
        same = events.loc[events["regulation_type"] == regulation]
        for district in S.SEOUL_DISTRICTS:
            applies = [
                (e["effective_date"], e["value"])
                for _, e in same.iterrows()
                if district in parse_districts(e["districts"])
            ]
            active = 0
            began: pd.Timestamp | None = None
            for when, value in applies:
                if value == 1 and active == 0:
                    began = when
                elif value == 0 and active == 1:
                    rows.append(
                        {
                            S.DISTRICT: district,
                            "regulation_type": regulation,
                            "announced_date": pd.NaT,
                            "effective_from": began,
                            "effective_to": when,
                        }
                    )
                    began = None
                active = value
            if active == 1 and began is not None:
                rows.append(
                    {
                        S.DISTRICT: district,
                        "regulation_type": regulation,
                        "announced_date": pd.NaT,
                        "effective_from": began,
                        "effective_to": pd.NaT,
                    }
                )
    return pd.DataFrame(rows, columns=list(REGULATION_HISTORY_COLUMNS))


class RegulationCsvCollector:
    """``RegulationCollector`` 계약 구현. 이벤트 CSV 를 기간 테이블로 바꿔 돌려준다."""

    def __init__(self, path: str | Path = REGULATION_EVENTS_PATH) -> None:
        self.path = Path(path)

    def fetch(self) -> pd.DataFrame:
        return to_regulation_history(load_regulation_events(self.path))
