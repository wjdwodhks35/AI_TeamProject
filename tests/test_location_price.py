"""위치 타깃 인코딩 — 누출 방지 장치가 실제로 동작하는지."""

import numpy as np
import pandas as pd
import pytest

from apartment_ml.models.location_price import LocationPriceEncoder


def frame(complexes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"complex_id": complexes})


def test_fit_transform_excludes_own_row():
    """학습 행의 값은 out-of-fold 로 채운다 — 자기 거래가 자기 설명변수에 섞이면 안 된다."""
    # 한 행만 값이 크게 다르면, 그 행 자신의 값은 자기 값이 빠진 표로 채워진다.
    X = frame(["a"] * 10)
    y = np.array([0.0] * 9 + [100.0])
    enc = LocationPriceEncoder(levels=("complex_id",), smoothing=0.0, n_folds=5)
    oof = enc.fit_transform(X, y)[:, 0]
    assert np.allclose(enc.transform(X)[:, 0], 10.0)  # 전체 평균 = 100/10
    assert oof[9] == pytest.approx(0.0)  # 자기 값 100 이 빠지면 나머지는 모두 0
    assert oof.max() > 10.0  # 100 이 포함된 표로 채워진 행은 전체 평균보다 크다


def test_unseen_location_gets_overall_mean():
    enc = LocationPriceEncoder(levels=("complex_id",), smoothing=0.0)
    enc.fit(frame(["a", "a", "b", "b"]), np.array([1.0, 1.0, 3.0, 3.0]))
    out = enc.transform(frame(["처음보는단지"]))
    assert out[0, 0] == pytest.approx(2.0)  # 전체 평균
    assert out[0, 1] == 0.0  # 학습 거래 수 0 — 모델이 '믿을 값이 아니다' 를 알 수 있다


def test_small_group_is_pulled_toward_overall_mean():
    """거래가 적은 위치는 전체 평균 쪽으로 끌어당긴다."""
    X = frame(["big"] * 20 + ["tiny"])
    y = np.array([0.0] * 20 + [100.0])
    enc = LocationPriceEncoder(levels=("complex_id",), smoothing=10.0)
    enc.fit(X, y)
    tiny = enc.transform(frame(["tiny"]))[0, 0]
    assert 0.0 < tiny < 100.0
    # 평활 없이 계산하면 100 이 그대로 나온다.
    raw = LocationPriceEncoder(levels=("complex_id",), smoothing=0.0).fit(X, y)
    assert raw.transform(frame(["tiny"]))[0, 0] == pytest.approx(100.0)


def test_requires_target_and_reports_names():
    enc = LocationPriceEncoder(levels=("complex_id",))
    with pytest.raises(ValueError, match="정답값"):
        enc.fit(frame(["a"]))
    with pytest.raises(ValueError, match="NaN"):
        enc.fit(frame(["a"]), np.array([np.nan]))
    enc.fit(frame(["a", "a"]), np.array([1.0, 2.0]))
    assert list(enc.get_feature_names_out()) == ["loc_complex_id_mean", "loc_complex_id_n"]
