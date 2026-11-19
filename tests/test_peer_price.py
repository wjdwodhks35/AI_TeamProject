"""거래 특성·기준가·이상거래 레이블 테스트."""

import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.data.preprocess import (
    DEFAULT_ANOMALY_THRESHOLDS,
    SchemaValidationError,
    add_deal_features,
    add_peer_price,
    label_anomalies,
)

PER_M2 = 10_000_000  # ㎡당 1천만원


def deals(rows: list[dict]) -> pd.DataFrame:
    base = {
        "district": "강남구",
        "dong": "대치동",
        "complex_id": "11680-1000",
        "contract_date": "2024-01-15",
        "area_m2": 59.0,
        "price_krw": int(59.0 * PER_M2),
        "floor": 10,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    return df


# ---------------------------------------------------------------------------
# 거래 특성
# ---------------------------------------------------------------------------


def test_deal_type_mapping():
    df = deals([{}, {}, {}])
    df["deal_type"] = ["직거래", "중개거래", ""]
    out = add_deal_features(df)
    assert out[S.IS_DIRECT_DEAL].tolist()[:2] == [1, 0]
    assert pd.isna(out[S.IS_DIRECT_DEAL].iloc[2])  # 2021년 자료에는 거래유형이 없다


def test_agent_location():
    df = deals([{}, {}, {}])
    df["agent_district"] = ["서울 강남구", "서울 영등포구", ""]
    out = add_deal_features(df)
    assert out[S.AGENT_LOCATION].tolist() == [1, 2, 0]


def test_land_leasehold():
    df = deals([{}, {}, {}])
    df["land_leasehold_raw"] = ["Y", "N", ""]
    out = add_deal_features(df)
    assert out[S.LAND_LEASEHOLD].tolist()[:2] == [1, 0]
    assert pd.isna(out[S.LAND_LEASEHOLD].iloc[2])


def test_deal_features_without_source_columns():
    """원본 컬럼이 없으면 그냥 넘어간다 (기존 샘플 데이터 호환)."""
    out = add_deal_features(deals([{}]))
    assert S.IS_DIRECT_DEAL not in out.columns


# ---------------------------------------------------------------------------
# 기준가
# ---------------------------------------------------------------------------


def test_peer_price_uses_area_adjusted_median():
    """단지 전체(2단계)로 기준을 만들 때 면적을 보정한다.

    같은 단지의 59㎡와 115㎡는 가격이 두 배 차이난다. 가격 중앙값을 그대로
    기준으로 삼으면 작은 평형은 전부 싸게, 큰 평형은 전부 비싸게 잡힌다.
    """
    rows = [
        # 1분기: 평형대별 2건씩이라 1단계(단지+평형)는 표본 부족 → 2단계(단지)로 내려간다
        {"contract_date": "2024-01-10", "area_m2": 59.0, "price_krw": int(59.0 * PER_M2)},
        {"contract_date": "2024-02-10", "area_m2": 59.0, "price_krw": int(59.0 * PER_M2)},
        {"contract_date": "2024-01-20", "area_m2": 115.0, "price_krw": int(115.0 * PER_M2)},
        {"contract_date": "2024-02-20", "area_m2": 115.0, "price_krw": int(115.0 * PER_M2)},
        # 2분기: ㎡당 가격이 1분기와 같은 59㎡ 거래
        {"contract_date": "2024-04-10", "area_m2": 59.0, "price_krw": int(59.0 * PER_M2)},
    ]
    out = add_peer_price(deals(rows))
    last = out.iloc[4]
    assert last[S.PEER_TIER] == 2
    # 면적 보정이 되면 기준가 = 1천만 x 59 = 5.9억, 비율 1.0
    assert last[S.PEER_MEDIAN_PRICE_KRW] == pytest.approx(59.0 * PER_M2)
    assert last[S.PRICE_RATIO] == pytest.approx(1.0)


def test_peer_price_prefers_narrow_tier():
    """같은 단지·같은 평형 표본이 3건 이상이면 1단계를 쓴다."""
    rows = [{"contract_date": f"2024-0{m}-10", "price_krw": int(59.0 * PER_M2)} for m in (1, 2, 3)]
    rows.append({"contract_date": "2024-04-10"})
    out = add_peer_price(deals(rows))
    assert out.iloc[3][S.PEER_TIER] == 1


def test_peer_price_falls_back_to_district():
    """단지·법정동 표본이 없으면 자치구+평형대(4단계)까지 내려간다."""
    rows = [
        {
            "contract_date": "2024-01-10",
            "complex_id": f"11680-{i}",
            "dong": f"동{i}",
            "price_krw": int(59.0 * PER_M2),
        }
        for i in range(5)
    ]
    rows.append({"contract_date": "2024-04-10", "complex_id": "11680-999", "dong": "새동"})
    out = add_peer_price(deals(rows))
    assert out.iloc[5][S.PEER_TIER] == 4


def test_peer_price_does_not_look_ahead():
    """직전 기간만 보므로 첫 기간 거래에는 기준가가 없다."""
    out = add_peer_price(deals([{"contract_date": "2024-01-10"}]))
    assert out.iloc[0][S.PEER_TIER] == 0
    assert pd.isna(out.iloc[0][S.PEER_MEDIAN_PRICE_KRW])


def test_peer_price_requires_columns():
    df = deals([{}]).drop(columns=["price_krw"])
    with pytest.raises(SchemaValidationError, match="기준가 계산에 필요한 컬럼"):
        add_peer_price(df)


# ---------------------------------------------------------------------------
# 이상거래 레이블
# ---------------------------------------------------------------------------


def test_label_uses_tier_specific_threshold():
    """1단계는 ±20%, 2단계는 ±30%. 넓은 단계일수록 기준이 흔들리기 때문이다."""
    df = pd.DataFrame(
        {
            S.PEER_TIER: pd.array([1, 1, 2, 2], dtype="Int64"),
            S.PRICE_RATIO: [1.25, 1.10, 1.25, 1.35],
        }
    )
    out = label_anomalies(df)
    assert out[S.ANOMALY_LABEL].tolist() == [1, 0, 0, 1]


def test_label_skips_coarse_tiers():
    """3·4단계는 레이블을 달지 않는다 (기준이 거칠어 잡음이 된다)."""
    df = pd.DataFrame(
        {
            S.PEER_TIER: pd.array([3, 4, 0], dtype="Int64"),
            S.PRICE_RATIO: [1.5, 0.5, None],
        }
    )
    out = label_anomalies(df)
    assert out[S.ANOMALY_LABEL].isna().all()


def test_label_custom_thresholds():
    df = pd.DataFrame({S.PEER_TIER: pd.array([1], dtype="Int64"), S.PRICE_RATIO: [1.15]})
    assert label_anomalies(df, thresholds={1: 0.10})[S.ANOMALY_LABEL].tolist() == [1]
    assert label_anomalies(df, thresholds={1: 0.20})[S.ANOMALY_LABEL].tolist() == [0]


def test_label_requires_peer_price():
    with pytest.raises(SchemaValidationError, match="add_peer_price"):
        label_anomalies(pd.DataFrame({"price_krw": [1]}))


def test_default_thresholds_cover_narrow_tiers_only():
    assert set(DEFAULT_ANOMALY_THRESHOLDS) == {1, 2}


def test_peer_columns_are_forbidden_as_features():
    """레이블을 역산할 수 있는 컬럼은 입력에서 막혀 있어야 한다."""
    for column in (S.PEER_MEDIAN_PRICE_KRW, S.PRICE_RATIO, S.PEER_TIER):
        assert column in S.REGRESSION_FORBIDDEN_FEATURES
        assert column in S.CLASSIFICATION_FORBIDDEN_FEATURES
