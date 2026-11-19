"""수집·정규화 테스트."""

import pandas as pd
import pytest

from apartment_ml.data.collect import SEOUL_LAWD_CODES, normalize_molit_api_frame


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
    from apartment_ml import schemas as S

    assert len(SEOUL_LAWD_CODES) == len(S.SEOUL_DISTRICTS)
    assert set(SEOUL_LAWD_CODES.values()) == set(S.SEOUL_DISTRICTS)
