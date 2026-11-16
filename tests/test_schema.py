from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from apartment_ml import schemas as S
from apartment_ml.data.collect import SEOUL_LAWD_CODES, normalize_molit_api_frame
from apartment_ml.data.preprocess import SchemaValidationError, validate_required_columns


def test_25_districts_and_lawd_codes_consistent():
    assert len(S.SEOUL_DISTRICTS) == 25
    assert set(SEOUL_LAWD_CODES.values()) == set(S.SEOUL_DISTRICTS)


def test_required_and_optional_columns_disjoint():
    assert set(S.REQUIRED_COLUMNS) == {"district", "contract_date", "area_m2", "floor", "price_krw"}
    assert not set(S.REQUIRED_COLUMNS) & set(S.OPTIONAL_COLUMNS)
    assert "anomaly_label" in S.OPTIONAL_COLUMNS


def test_regression_features_exclude_targets():
    assert not set(S.REGRESSION_FEATURES) & S.REGRESSION_FORBIDDEN_FEATURES
    assert "price_krw" not in S.REGRESSION_FEATURES
    assert "anomaly_label" not in S.REGRESSION_FEATURES
    assert "anomaly_label" not in S.CLASSIFICATION_FEATURES
    assert "price_krw" in S.CLASSIFICATION_FEATURES  # 분류에서는 사용 가능


def test_missing_required_column_raises(raw_df):
    with pytest.raises(SchemaValidationError, match="price_krw"):
        validate_required_columns(raw_df.drop(columns=["price_krw"]))


def test_transaction_record_valid():
    rec = S.TransactionRecord(
        district="강남구", contract_date=date(2024, 1, 2), area_m2=84.9, floor=10,
        price_krw=1_500_000_000,
    )  # fmt: skip
    row = rec.to_record()
    assert row["anomaly_label"] is None
    assert row["reg_land_permit"] is None


@pytest.mark.parametrize(
    "override",
    [
        {"district": "부산진구"},
        {"price_krw": -1},
        {"area_m2": 0},
        {"floor": 500},
        {"contract_date": "2024-13-40"},
        {"anomaly_label": 2},
    ],
)
def test_transaction_record_invalid(override):
    base = dict(district="강남구", contract_date="2024-01-02", area_m2=84.9, floor=10, price_krw=1)
    with pytest.raises(ValidationError):
        S.TransactionRecord(**{**base, **override})


def test_normalize_molit_api_frame():
    raw = pd.DataFrame(
        {
            "sggCd": ["11680", "11440"],
            "dealYear": ["2024", "2024"],
            "dealMonth": ["3", "11"],
            "dealDay": ["5", "30"],
            "dealAmount": ["125,000", "98,500"],
            "excluUseAr": ["84.97", "59.9"],
            "floor": ["10", "3"],
            "buildYear": ["2008", "2015"],
            "cdealType": ["", "O"],  # 두 번째는 해제 거래 → 제거
        }
    )
    out = normalize_molit_api_frame(raw)
    assert len(out) == 1
    assert out.iloc[0]["district"] == "강남구"
    assert out.iloc[0]["contract_date"] == "2024-03-05"
    assert out.iloc[0]["price_krw"] == 1_250_000_000
