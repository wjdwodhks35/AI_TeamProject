import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from apartment_ml import schemas as S
from apartment_ml.data.merge import attach_base_rate, attach_regulation
from apartment_ml.data.preprocess import (
    SchemaValidationError,
    build_feature_preprocessor,
    coerce_types,
    drop_duplicate_transactions,
    manwon_to_krw,
    normalize_dates,
    preprocess_transactions,
    to_model_frame,
)


def test_preprocess_happy_path(raw_df):
    res = preprocess_transactions(raw_df)
    df = res.data
    assert len(df) == len(raw_df)
    assert str(df["contract_date"].dtype) == "datetime64[ns]"
    assert {"contract_year", "contract_month", "building_age"} <= set(df.columns)
    assert df["contract_date"].is_monotonic_increasing
    assert set(S.OPTIONAL_COLUMNS) - {"apartment_name", "dong", "source"} <= set(df.columns)
    assert res.report.missing_counts["anomaly_label"] == len(df)  # 레이블 미확보 허용


def test_invalid_date_raises(raw_df):
    raw_df.loc[3, "contract_date"] = "2024-02-30"
    with pytest.raises(SchemaValidationError):
        preprocess_transactions(raw_df)


def test_invalid_date_dropped(raw_df):
    raw_df.loc[3, "contract_date"] = "not-a-date"
    res = preprocess_transactions(raw_df, invalid="drop")
    assert len(res.data) == len(raw_df) - 1


def test_date_formats_normalized():
    out = normalize_dates(pd.Series(["2024-01-05", "20240105", "2024.01.05", "2024/1/5"]))
    assert out.nunique() == 1 and out.iloc[0] == pd.Timestamp("2024-01-05")


@pytest.mark.parametrize("price", [-1, 0])
def test_non_positive_price(raw_df, price):
    raw_df.loc[2, "price_krw"] = price
    with pytest.raises(SchemaValidationError, match="non_positive_price"):
        preprocess_transactions(raw_df)
    res = preprocess_transactions(raw_df, invalid="drop")
    assert res.report.invalid_rows["non_positive_price"] == 1
    assert len(res.data) == len(raw_df) - 1


@pytest.mark.parametrize("area", [0, -5, 5000, "abc"])
def test_invalid_area(raw_df, area):
    raw_df["area_m2"] = raw_df["area_m2"].astype(object)
    raw_df.loc[1, "area_m2"] = area
    with pytest.raises(SchemaValidationError):
        preprocess_transactions(raw_df)


def test_unknown_district(raw_df):
    raw_df.loc[0, "district"] = "해운대구"
    with pytest.raises(SchemaValidationError, match="unknown_district"):
        preprocess_transactions(raw_df)


def test_duplicates_removed(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[[0, 1]]], ignore_index=True)
    res = preprocess_transactions(dup)
    assert res.report.duplicate_rows == 2
    assert len(res.data) == len(raw_df)


def test_same_key_different_apartment_is_not_duplicate(raw_df):
    df = coerce_types(pd.concat([raw_df.iloc[[0]], raw_df.iloc[[0]]], ignore_index=True))
    df["apartment_name"] = ["A단지", "B단지"]
    out, n = drop_duplicate_transactions(df)
    assert n == 0 and len(out) == 2


def test_manwon_to_krw():
    out = manwon_to_krw(pd.Series(["125,000", " 9,800 ", None]))
    assert out.iloc[0] == 1_250_000_000 and out.iloc[1] == 98_000_000 and pd.isna(out.iloc[2])


def test_attach_base_rate_point_in_time(raw_df):
    df = preprocess_transactions(raw_df).data
    rates = pd.DataFrame({"effective_date": ["2023-01-03", "2023-01-06"], "base_rate": [3.25, 3.5]})
    out = attach_base_rate(df, rates)
    by_date = dict(zip(out["contract_date"].dt.strftime("%Y-%m-%d"), out["base_rate"], strict=True))
    assert np.isnan(by_date["2023-01-02"])  # 첫 적용일 이전
    assert by_date["2023-01-03"] == 3.25  # 적용일 당일 포함
    assert by_date["2023-01-05"] == 3.25
    assert by_date["2023-01-06"] == 3.5
    assert len(out) == len(df)


def test_attach_regulation_effective_window(raw_df):
    df = preprocess_transactions(raw_df).data
    regs = pd.DataFrame(
        {
            "district": ["강남구"],
            "regulation_type": ["reg_land_permit"],
            "effective_from": ["2023-01-03"],
            "effective_to": ["2023-01-07"],  # 종료일 미포함
        }
    )
    out = attach_regulation(df, regs)
    gangnam = out[out["district"] == "강남구"]
    gangnam = gangnam.set_index(gangnam["contract_date"].dt.day)
    assert gangnam.loc[1, "reg_land_permit"] == 0
    assert gangnam.loc[3, "reg_land_permit"] == 1
    assert gangnam.loc[5, "reg_land_permit"] == 1
    assert gangnam.loc[7, "reg_land_permit"] == 0
    assert (out.loc[out["district"] == "마포구", "reg_land_permit"] == 0).all()


def test_feature_preprocessor_fits_on_train_only():
    """누출 방지: 대체값/스케일 통계는 train 에서만 학습되어 test 에 영향받지 않는다."""
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan], "c": ["강남구", "마포구", "강남구"]})
    test = pd.DataFrame({"x": [1000.0, np.nan], "c": ["노원구", "강남구"]})
    pre = build_feature_preprocessor(["x"], ["c"], scale_numeric=False)
    Pipeline([("p", pre)]).fit(train)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == 1.5  # train 중앙값
    out = pre.transform(test)
    assert out[1, 0] == 1.5  # test 결측도 train 통계로 대체
    assert out.shape[1] == 1 + 2  # 범주는 train 에 있던 2개만 (미지 범주 무시)


def test_to_model_frame_converts_nullable():
    df = pd.DataFrame({"a": pd.array([1, None], dtype="Int64"), "d": ["강남구", None]})
    out = to_model_frame(df, ["a", "d"])
    assert out["a"].dtype == "float64" and np.isnan(out["a"].iloc[1])
    with pytest.raises(SchemaValidationError):
        to_model_frame(df, ["missing_col"])
