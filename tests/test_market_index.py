"""시장지수 — 시차 규칙과 대체 규칙 (회귀가 시점 흐름을 처리하는 방식)."""

import numpy as np
import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.models.market_index import MarketIndex, MarketIndexError


def deals(spec: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """(계약일, 자치구, 면적, 가격) 목록 → 거래 프레임."""
    return pd.DataFrame(
        {
            S.CONTRACT_DATE: pd.to_datetime([s[0] for s in spec]),
            S.DISTRICT: [s[1] for s in spec],
            S.AREA_M2: [s[2] for s in spec],
            S.REGRESSION_TARGET: [s[3] for s in spec],
        }
    )


def test_uses_previous_month():
    """3월 지수는 2월 거래로 만든다 — 계약 시점에 이미 신고가 끝난 거래만 쓴다."""
    df = deals(
        [
            ("2024-02-10", "강남구", 100.0, 1_000_000_000),
            ("2024-02-20", "강남구", 100.0, 1_000_000_000),
            ("2024-03-05", "강남구", 100.0, 9_999_999_999),
        ]
    )
    index = MarketIndex(min_deals=1).fit(df)
    march = index.transform(df[df[S.CONTRACT_DATE] >= "2024-03-01"])
    assert march[0] == pytest.approx(np.log(1_000_000_000 / 100.0))


def test_thin_district_month_falls_back_to_seoul():
    """거래가 적은 자치구-월은 값이 흔들리므로 서울 전체 평균을 쓴다."""
    df = deals(
        [("2024-01-05", "강남구", 100.0, 2_000_000_000)]
        + [(f"2024-01-{d:02d}", "마포구", 100.0, 1_000_000_000) for d in range(6, 16)]
        + [("2024-02-10", "강남구", 100.0, 1_234_567_890)]
    )
    january = df[df[S.CONTRACT_DATE] < "2024-02-01"]
    february = df[df[S.CONTRACT_DATE] >= "2024-02-01"]
    seoul_mean = float(np.log(january[S.REGRESSION_TARGET] / january[S.AREA_M2]).mean())
    # 강남구 1월 거래는 1건뿐이라 min_deals=5 에 걸려 서울 전체 평균으로 대체된다.
    index = MarketIndex(min_deals=5).fit(df)
    assert index.transform(february)[0] == pytest.approx(seoul_mean)


def test_months_after_index_are_held_at_last_value():
    """지수를 만든 구간 이후(= 미래)는 마지막 값으로 고정한다."""
    df = deals([(f"2024-01-{d:02d}", "강남구", 100.0, 1_000_000_000) for d in range(1, 12)])
    index = MarketIndex(min_deals=1).fit(df)
    future = deals([("2030-06-01", "강남구", 100.0, 1.0)])
    assert index.transform(future)[0] == pytest.approx(np.log(1_000_000_000 / 100.0))


def test_unknown_district_falls_back_to_overall_mean():
    df = deals([(f"2024-01-{d:02d}", "강남구", 100.0, 1_000_000_000) for d in range(1, 12)])
    index = MarketIndex(min_deals=1).fit(df)
    other = deals([("2024-02-01", "노원구", 100.0, 1.0)])
    assert np.isfinite(index.transform(other)[0])


def test_errors():
    df = deals([("2024-01-01", "강남구", 100.0, 1_000_000_000)])
    with pytest.raises(ValueError, match="lag_months"):
        MarketIndex(lag_months=0)
    with pytest.raises(MarketIndexError, match="필요한 컬럼"):
        MarketIndex().fit(df.drop(columns=[S.DISTRICT]))
    with pytest.raises(MarketIndexError, match="fit"):
        MarketIndex().transform(df)


def test_saved_model_from_other_code_version_is_refused(tmp_path):
    """예측 방식이 다른 버전으로 저장된 파일은 거부한다 (조용히 틀리는 것보다 낫다)."""
    import joblib

    from apartment_ml.data.merge import attach_base_rate
    from apartment_ml.data.preprocess import preprocess_transactions
    from apartment_ml.models.base import ModelNotReadyError
    from apartment_ml.models.regression import PriceRegressionModel
    from tests.conftest import make_rows

    train = attach_base_rate(
        preprocess_transactions(make_rows(40)).data,
        pd.DataFrame({"effective_date": ["2022-01-01"], "base_rate": [3.0]}),
    )
    path = tmp_path / "price.joblib"
    PriceRegressionModel("linear_regression").fit(train).save(path)
    assert PriceRegressionModel.load(path).is_fitted

    artifact = joblib.load(path)
    artifact["metadata"]["prediction_contract"] = 1
    joblib.dump(artifact, path)
    with pytest.raises(ModelNotReadyError, match="다른 버전"):
        PriceRegressionModel.load(path)
