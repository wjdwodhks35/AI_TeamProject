"""수집·정규화 테스트."""

from datetime import date

import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.data import collect as C
from apartment_ml.data.collect import (
    SEOUL_LAWD_CODES,
    CollectError,
    EcosBaseRateCollector,
    MolitApiCollector,
    normalize_molit_api_frame,
    parse_molit_xml,
    to_base_rate_changes,
)


class FakeResponse:
    """requests.get 대체용. 테스트는 네트워크를 쓰지 않는다 (conftest 가 차단)."""

    def __init__(self, text: str = "", payload: object = None) -> None:
        self.text = text
        self.encoding = "utf-8"
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def molit_xml(count: int, total: int, *, start: int = 0) -> str:
    items = "".join(
        f"<item><aptNm>단지{i}</aptNm><dealAmount>125,000</dealAmount>"
        f"<excluUseAr>84.97</excluUseAr><floor>{i % 20 + 1}</floor>"
        f"<dealYear>2026</dealYear><dealMonth>4</dealMonth><dealDay>5</dealDay>"
        f"<sggCd>11350</sggCd></item>"
        for i in range(start, start + count)
    )
    return (
        "<response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header>"
        f"<body><items>{items}</items><numOfRows>1000</numOfRows>"
        f"<pageNo>1</pageNo><totalCount>{total}</totalCount></body></response>"
    )


def molit_rows(rows: list[dict]) -> pd.DataFrame:
    """MOLIT OpenAPI 응답 형태의 가짜 DataFrame."""
    base = {
        "sggCd": "11680",
        "umdNm": "대치동",
        "aptNm": "은마",
        "dealYear": "2024",
        "dealMonth": "3",
        "dealDay": "5",
        "dealAmount": "125,000",
        "excluUseAr": "84.97",
        "floor": "10",
        "buildYear": "1979",
        "cdealType": "",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_normalize_happy_path():
    out = normalize_molit_api_frame(molit_rows([{}]))
    assert len(out) == 1
    assert out.iloc[0]["district"] == "강남구"
    assert out.iloc[0]["contract_date"] == "2024-03-05"
    assert out.iloc[0]["price_krw"] == 1_250_000_000
    assert out.iloc[0]["source"] == "molit_api"


def test_normalize_missing_column_raises():
    raw = molit_rows([{}]).drop(columns=["dealAmount"])
    with pytest.raises(ValueError, match="MOLIT 원본 컬럼이 없습니다"):
        normalize_molit_api_frame(raw)


def test_cancelled_pair_removes_original_too():
    """해제된 거래는 원본 행까지 함께 사라진다.

    국토교통부 자료는 계약 해제 시 원본 행을 남기고 해제 행을 한 줄 더 넣는다.
    해제 행만 지우면 원본이 남아 깨진 계약이 정상 거래로 보인다.
    """
    raw = molit_rows(
        [
            {},  # 원본
            {"cdealType": "O", "cdealDay": "24.05.01"},  # 같은 거래의 해제 행
            {"dealDay": "20", "dealAmount": "130,000"},  # 무관한 정상 거래
        ]
    )
    out = normalize_molit_api_frame(raw)
    assert len(out) == 1
    assert out.iloc[0]["contract_date"] == "2024-03-20"


def test_cancelled_without_original_is_removed():
    raw = molit_rows([{"cdealType": "O"}, {"dealDay": "20"}])
    out = normalize_molit_api_frame(raw)
    assert len(out) == 1
    assert out.iloc[0]["contract_date"] == "2024-03-20"


def test_keep_cancelled_for_validation():
    """drop_cancelled=False 면 해제 여부를 cancel_type 으로 구분할 수 있다."""
    raw = molit_rows([{}, {"cdealType": "O"}])
    out = normalize_molit_api_frame(raw, drop_cancelled=False)
    assert len(out) == 2
    assert out["cancel_type"].tolist() == ["", "O"]


def test_same_complex_different_price_is_not_a_pair():
    """가격이 다르면 다른 거래이므로 해제에 휘말리지 않는다."""
    raw = molit_rows([{"dealAmount": "120,000"}, {"cdealType": "O"}])
    out = normalize_molit_api_frame(raw)
    assert len(out) == 1
    assert out.iloc[0]["price_krw"] == 1_200_000_000


def test_seoul_lawd_codes_cover_all_districts():
    assert len(SEOUL_LAWD_CODES) == len(S.SEOUL_DISTRICTS)
    assert set(SEOUL_LAWD_CODES.values()) == set(S.SEOUL_DISTRICTS)


# ---------------------------------------------------------------------------
# 응답 파싱
# ---------------------------------------------------------------------------


def test_parse_xml_reads_all_tags():
    """필드 이름을 하드코딩하지 않으므로 item 안의 태그가 모두 들어온다."""
    items, total = parse_molit_xml(molit_xml(2, 2))
    assert total == 2
    assert items[0]["aptNm"] == "단지0"
    assert items[0]["excluUseAr"] == "84.97"


def test_parse_xml_api_error_raises():
    xml = (
        "<response><header><resultCode>30</resultCode>"
        "<resultMsg>SERVICE KEY IS NOT REGISTERED ERROR</resultMsg></header></response>"
    )
    with pytest.raises(CollectError, match="resultCode=30"):
        parse_molit_xml(xml)


def test_parse_html_error_page_raises():
    """장애 시 HTML 이 오는데, 0건으로 넘기면 그 달이 조용히 비어버린다."""
    with pytest.raises(CollectError, match="응답 형식이 다릅니다"):
        parse_molit_xml("<html><body>502 Bad Gateway</body></html>")


def test_parse_non_xml_raises():
    with pytest.raises(CollectError, match="XML 이 아닌 응답"):
        parse_molit_xml("not xml at all {")


# ---------------------------------------------------------------------------
# 국토교통부 수집기
# ---------------------------------------------------------------------------


def test_molit_requires_key():
    with pytest.raises(ValueError, match="API 키가 비어 있습니다"):
        MolitApiCollector("   ")


@pytest.mark.parametrize(
    ("lawd_code", "year_month", "match"),
    [
        ("99999", "202604", "서울 자치구 코드가 아닙니다"),
        ("11350", "2026-04", "YYYYMM"),
        ("11350", "20264", "YYYYMM"),
    ],
)
def test_molit_rejects_bad_arguments(lawd_code, year_month, match):
    with pytest.raises(ValueError, match=match):
        MolitApiCollector("KEY").fetch(lawd_code, year_month)


def test_molit_paginates(monkeypatch):
    """1000건을 넘으면 다음 페이지까지 받는다 (노원구 2026-04 가 1042건이었다)."""
    pages: list[int] = []

    def fake_get(url, params=None, timeout=None):
        page = params["pageNo"]
        pages.append(page)
        count = 1000 if page == 1 else 42
        return FakeResponse(molit_xml(count, 1042, start=(page - 1) * 1000))

    monkeypatch.setattr(C.requests, "get", fake_get)
    df = MolitApiCollector("KEY", sleep_seconds=0).fetch("11350", "202604")
    assert len(df) == 1042
    assert pages == [1, 2]


def test_molit_single_page_does_not_request_more(monkeypatch):
    pages: list[int] = []

    def fake_get(url, params=None, timeout=None):
        pages.append(params["pageNo"])
        return FakeResponse(molit_xml(55, 55))

    monkeypatch.setattr(C.requests, "get", fake_get)
    df = MolitApiCollector("KEY", sleep_seconds=0).fetch("11350", "202604")
    assert len(df) == 55
    assert pages == [1]


def test_molit_api_error_is_not_retried(monkeypatch):
    """키 오류는 다시 요청해도 같은 결과이므로 바로 올린다."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return FakeResponse(
            "<response><header><resultCode>30</resultCode>"
            "<resultMsg>KEY ERROR</resultMsg></header></response>"
        )

    monkeypatch.setattr(C.requests, "get", fake_get)
    with pytest.raises(CollectError, match="resultCode=30"):
        MolitApiCollector("KEY", sleep_seconds=0).fetch("11350", "202604")
    assert len(calls) == 1


def test_molit_network_error_is_retried(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        raise OSError("connection reset")

    monkeypatch.setattr(C.requests, "get", fake_get)
    with pytest.raises(CollectError, match="수집 실패"):
        MolitApiCollector("KEY", max_retries=2, sleep_seconds=0).fetch("11350", "202604")
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 기준금리 수집기
# ---------------------------------------------------------------------------


def test_base_rate_keeps_only_change_days():
    """일별로 받아 값이 바뀐 날만 남긴다. 그래야 실제 적용일을 쓸 수 있다."""
    rows = [
        {"TIME": "20210801", "DATA_VALUE": "0.5"},
        {"TIME": "20210825", "DATA_VALUE": "0.5"},
        {"TIME": "20210826", "DATA_VALUE": "0.75"},  # 금통위 인상
        {"TIME": "20210901", "DATA_VALUE": "0.75"},
        {"TIME": "20211125", "DATA_VALUE": "1.0"},
    ]
    out = to_base_rate_changes(rows)
    assert out["base_rate"].tolist() == [0.5, 0.75, 1.0]
    assert out["effective_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2021-08-01",
        "2021-08-26",
        "2021-11-25",
    ]


def test_base_rate_skips_blank_values():
    rows = [
        {"TIME": "20210801", "DATA_VALUE": "0.5"},
        {"TIME": "20210802", "DATA_VALUE": ""},
        {"TIME": "20210803", "DATA_VALUE": None},
        {"TIME": "20210826", "DATA_VALUE": "0.75"},
    ]
    out = to_base_rate_changes(rows)
    assert len(out) == 2


def test_base_rate_all_blank_raises():
    with pytest.raises(CollectError, match="값이 있는 날이 없습니다"):
        to_base_rate_changes([{"TIME": "20210801", "DATA_VALUE": ""}])


def test_ecos_requires_key():
    with pytest.raises(ValueError, match="ECOS API 키가 비어 있습니다"):
        EcosBaseRateCollector("")


def test_ecos_rejects_reversed_period():
    with pytest.raises(ValueError, match="start 가 end 보다 늦습니다"):
        EcosBaseRateCollector("KEY").fetch(date(2026, 1, 1), date(2025, 1, 1))


def test_ecos_fetch(monkeypatch):
    payload = {
        "StatisticSearch": {
            "list_total_count": 3,
            "row": [
                {"TIME": "20210101", "DATA_VALUE": "0.5", "UNIT_NAME": "연%"},
                {"TIME": "20210826", "DATA_VALUE": "0.75", "UNIT_NAME": "연%"},
            ],
        }
    }
    monkeypatch.setattr(C.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
    out = EcosBaseRateCollector("KEY").fetch(date(2021, 1, 1), date(2021, 12, 31))
    assert list(out.columns) == ["effective_date", "base_rate"]
    assert out["base_rate"].tolist() == [0.5, 0.75]


def test_ecos_auth_error_raises(monkeypatch):
    payload = {"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}}
    monkeypatch.setattr(C.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
    with pytest.raises(CollectError, match="INFO-100"):
        EcosBaseRateCollector("KEY").fetch(date(2021, 1, 1), date(2021, 12, 31))


def test_ecos_empty_result_raises(monkeypatch):
    payload = {"StatisticSearch": {"row": []}}
    monkeypatch.setattr(C.requests, "get", lambda *a, **k: FakeResponse(payload=payload))
    with pytest.raises(CollectError, match="받은 기준금리가 없습니다"):
        EcosBaseRateCollector("KEY").fetch(date(2021, 1, 1), date(2021, 12, 31))


# ---------------------------------------------------------------------------
# 규제 이력
# ---------------------------------------------------------------------------

EVENTS_HEADER = "effective_date,districts,regulation_type,value,description,source\n"


def write_events(tmp_path, body: str):
    path = tmp_path / "regulation_events.csv"
    path.write_text(EVENTS_HEADER + body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ALL", 25),
        ("강남구|송파구", 2),
        ("EXCEPT:강남구|서초구|송파구|용산구", 21),
    ],
)
def test_parse_districts(text, expected):
    assert len(C.parse_districts(text)) == expected


def test_parse_districts_except_excludes_named():
    out = C.parse_districts("EXCEPT:강남구|서초구|송파구|용산구")
    assert "강남구" not in out
    assert "노원구" in out


@pytest.mark.parametrize("text", ["", "   ", "강남군", "ALL|강남구"])
def test_parse_districts_invalid(text):
    with pytest.raises(ValueError):
        C.parse_districts(text)


def test_load_events_skips_record_only_rows(tmp_path):
    """regulation_type 이 빈 줄은 근거 기록용이므로 계산에서 빠진다."""
    path = write_events(
        tmp_path,
        "2021-01-01,ALL,reg_adjustment_target,1,조정대상지역,국토부\n"
        "2022-07-01,ALL,,,DSR 3단계 — 전국 공통,금융위\n",
    )
    events = C.load_regulation_events(path)
    assert len(events) == 1
    assert events.iloc[0]["regulation_type"] == "reg_adjustment_target"


def test_load_events_unknown_type_raises(tmp_path):
    path = write_events(tmp_path, "2021-01-01,ALL,reg_made_up,1,없는 규제,출처\n")
    with pytest.raises(ValueError, match="알 수 없는 regulation_type"):
        C.load_regulation_events(path)


def test_load_events_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("effective_date,districts\n2021-01-01,ALL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="컬럼이 없습니다"):
        C.load_regulation_events(path)


def test_history_builds_open_interval(tmp_path):
    """해제 이벤트가 없으면 현재까지 유효한 구간이 된다."""
    path = write_events(tmp_path, "2021-01-01,강남구,reg_land_permit,1,지정,서울시\n")
    history = C.RegulationCsvCollector(path).fetch()
    row = history[history["district"] == "강남구"].iloc[0]
    assert row["effective_from"] == pd.Timestamp("2021-01-01")
    assert pd.isna(row["effective_to"])
    assert len(history) == 1  # 지정되지 않은 구는 행이 생기지 않는다


def test_history_splits_on_release_and_redesignation(tmp_path):
    """해제 뒤 재지정되면 두 구간으로 나뉜다 (강남구 토지거래허가구역)."""
    path = write_events(
        tmp_path,
        "2021-01-01,강남구,reg_land_permit,1,지정,서울시\n"
        "2025-02-13,강남구,reg_land_permit,0,해제,서울시\n"
        "2025-03-19,강남구,reg_land_permit,1,재지정,서울시\n",
    )
    history = C.RegulationCsvCollector(path).fetch()
    assert len(history) == 2
    first, second = history.iloc[0], history.iloc[1]
    assert first["effective_to"] == pd.Timestamp("2025-02-13")
    assert second["effective_from"] == pd.Timestamp("2025-03-19")
    assert pd.isna(second["effective_to"])


def test_history_except_leaves_named_districts_untouched(tmp_path):
    """EXCEPT 해제는 열거된 구의 구간을 끊지 않는다."""
    path = write_events(
        tmp_path,
        "2021-01-01,ALL,reg_adjustment_target,1,전역 지정,국토부\n"
        "2023-01-05,EXCEPT:강남구,reg_adjustment_target,0,강남 제외 해제,국토부\n",
    )
    history = C.RegulationCsvCollector(path).fetch()
    강남 = history[history["district"] == "강남구"]
    노원 = history[history["district"] == "노원구"]
    assert len(강남) == 1 and pd.isna(강남.iloc[0]["effective_to"])
    assert len(노원) == 1 and 노원.iloc[0]["effective_to"] == pd.Timestamp("2023-01-05")


def test_history_feeds_attach_regulation(tmp_path):
    """만든 이력이 merge.attach_regulation 에 그대로 들어간다."""
    from apartment_ml.data.merge import attach_regulation

    path = write_events(
        tmp_path,
        "2021-01-01,강남구,reg_land_permit,1,지정,서울시\n"
        "2025-02-13,강남구,reg_land_permit,0,해제,서울시\n",
    )
    history = C.RegulationCsvCollector(path).fetch()
    transactions = pd.DataFrame(
        {
            "district": ["강남구", "강남구", "노원구"],
            "contract_date": ["2024-06-01", "2025-06-01", "2024-06-01"],
        }
    )
    out = attach_regulation(transactions, history)
    assert out["reg_land_permit"].tolist() == [1, 0, 0]


def test_real_events_file_is_loadable():
    """저장소에 든 실제 규제 파일이 깨지지 않았는지 확인한다."""
    history = C.RegulationCsvCollector(C.REGULATION_EVENTS_PATH).fetch()
    assert len(history) > 0
    assert set(history["regulation_type"]) <= set(S.REGULATION_COLUMNS)
    assert (history["effective_from"].notna()).all()


# ---------------------------------------------------------------------------
# 수집 범위·경로
# ---------------------------------------------------------------------------


def test_month_range_inclusive():
    assert C.month_range("202101", "202103") == ["202101", "202102", "202103"]
    assert C.month_range("202612", "202612") == ["202612"]


def test_month_range_crosses_year():
    out = C.month_range("202211", "202302")
    assert out == ["202211", "202212", "202301", "202302"]


def test_month_range_full_period():
    """2021-01 ~ 2026-08 은 68개월이고 25개 구를 곱하면 1,700회다."""
    assert len(C.month_range("202101", "202608")) == 68


@pytest.mark.parametrize(
    ("start", "end"),
    [("2021-01", "202608"), ("20211", "202608"), ("202608", "202101")],
)
def test_month_range_invalid(start, end):
    with pytest.raises(ValueError):
        C.month_range(start, end)


def test_raw_month_path():
    path = C.raw_month_path("data/raw", "11350", "202604")
    assert path.parts[-2:] == ("11350_노원구", "202604.json")


def test_raw_month_path_unknown_code():
    with pytest.raises(ValueError, match="서울 자치구 코드가 아닙니다"):
        C.raw_month_path("data/raw", "41135", "202604")
