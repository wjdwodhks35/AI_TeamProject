"""테스트 전용 가짜 예측 객체. ⚠️ 실제 모델이 아니며 src/ 에서 import 금지."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FakePriceModel:
    """가격 = 1억 × 면적/10 − 금리 × 1천만 (테스트 계산 검증용 임의 공식)."""

    name = "fake_price_model_TEST_ONLY"
    is_fitted = True
    feature_columns = ["area_m2", "base_rate", "district"]
    feature_ranges = {"base_rate": (0.5, 3.5)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rate = X["base_rate"].astype(float).fillna(0.0).to_numpy()
        return X["area_m2"].astype(float).to_numpy() * 10_000_000 - rate * 10_000_000


class FakeAnomalyModel:
    name = "fake_anomaly_model_TEST_ONLY"
    is_fitted = True
    threshold = 0.5
    feature_columns = ["price_per_m2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.9)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)
