import pandas as pd
import pytest

from apartment_ml.data.preprocess import preprocess_transactions
from apartment_ml.evaluation.split import (
    InsufficientDataError,
    expanding_window_splits,
    split_by_dates,
    time_based_split,
)
from tests.conftest import make_rows


def test_ratio_split_is_chronological():
    df = preprocess_transactions(make_rows(100)).data.sample(frac=1, random_state=0)  # 섞기
    s = time_based_split(df, 0.7, 0.15, 0.15)
    assert len(s.train) + len(s.valid) + len(s.test) == 100
    assert s.train["contract_date"].max() < s.valid["contract_date"].min()
    assert s.valid["contract_date"].max() < s.test["contract_date"].min()
    assert (len(s.train), len(s.valid), len(s.test)) == (70, 15, 15)


def test_no_leakage_between_splits():
    df = preprocess_transactions(make_rows(60)).data
    s = time_based_split(df)
    idx = [set(p.index) for p in (s.train, s.valid, s.test)]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert s.train["contract_date"].max() < s.test["contract_date"].min()


def test_same_date_not_split_across_boundary():
    # 하루에 4건씩 10일 → 비율 경계가 날짜 중간에 걸려도 같은 날짜는 같은 구간
    df = pd.concat([make_rows(10)] * 4, ignore_index=True)
    df["floor"] = range(len(df))  # 중복 아님
    df = preprocess_transactions(df).data
    s = time_based_split(df, 0.65, 0.2, 0.15)
    dates = [set(p["contract_date"]) for p in (s.train, s.valid, s.test)]
    assert not (dates[0] & dates[1] or dates[1] & dates[2] or dates[0] & dates[2])
    assert all(len(p) % 4 == 0 for p in (s.train, s.valid, s.test))


def test_split_by_dates_boundaries():
    df = preprocess_transactions(make_rows(10)).data  # 2023-01-01 ~ 01-10
    s = split_by_dates(df, "2023-01-06", "2023-01-09")
    assert s.train["contract_date"].max() == pd.Timestamp("2023-01-05")
    assert s.valid["contract_date"].min() == pd.Timestamp("2023-01-06")
    assert s.test["contract_date"].min() == pd.Timestamp("2023-01-09")
    summary = s.summary()
    assert summary.loc["train", "rows"] == 5 and summary.loc["test", "rows"] == 2


def test_insufficient_data_errors():
    with pytest.raises(InsufficientDataError):
        time_based_split(make_rows(2))
    with pytest.raises(InsufficientDataError):
        time_based_split(make_rows(0))
    one_day = pd.concat([make_rows(1)] * 10, ignore_index=True)
    with pytest.raises(InsufficientDataError):  # 날짜가 하나뿐이면 분할 불가
        time_based_split(one_day)
    with pytest.raises(InsufficientDataError):
        split_by_dates(make_rows(10), "2030-01-01", "2031-01-01")


def test_invalid_args():
    with pytest.raises(ValueError):
        time_based_split(make_rows(10), 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        split_by_dates(make_rows(10), "2023-01-09", "2023-01-06")
    bad = make_rows(10)
    bad.loc[0, "contract_date"] = None
    with pytest.raises(ValueError):
        time_based_split(bad)


def test_expanding_window_cv():
    df = preprocess_transactions(make_rows(30)).data
    folds = list(expanding_window_splits(df, n_splits=3))
    assert len(folds) == 3
    prev = 0
    for tr, va in folds:
        assert df["contract_date"].iloc[tr].max() < df["contract_date"].iloc[va].min()
        assert len(tr) > prev
        prev = len(tr)
    with pytest.raises(InsufficientDataError):
        list(expanding_window_splits(make_rows(3), n_splits=5))
