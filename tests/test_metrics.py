import math

import numpy as np
import pytest

from apartment_ml.evaluation.metrics import (
    MetricInputError,
    classification_metrics,
    regression_metrics,
)


def test_regression_metrics_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m["mae"] == pytest.approx(50 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt((100 + 100 + 900) / 3))
    assert m["n"] == 3


def test_regression_perfect():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert m["mae"] == 0 and m["rmse"] == 0
    assert m["r2"] == pytest.approx(1.0) and m["mape"] == 0


def test_regression_r2_and_mape():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    # SST = 20000, SSE = 1100
    assert m["r2"] == pytest.approx(1 - 1100 / 20000)
    assert m["mape"] == pytest.approx((10 / 100 + 10 / 200 + 30 / 300) / 3 * 100)


def test_regression_r2_undefined_when_target_constant():
    m = regression_metrics([5.0, 5.0], [4.0, 6.0])
    assert math.isnan(m["r2"])


def test_regression_mape_skips_zero_target():
    m = regression_metrics([0.0, 100.0], [10.0, 110.0])
    assert m["mape"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "y_true,y_pred",
    [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), (["a", "b"], [1, 2])],
)
def test_regression_invalid(y_true, y_pred):
    with pytest.raises(MetricInputError):
        regression_metrics(y_true, y_pred)


def test_classification_metrics_values():
    y_true = [1, 1, 1, 0, 0, 0, 0, 1]
    y_pred = [1, 1, 0, 0, 0, 1, 0, 0]
    m = classification_metrics(y_true, y_pred)
    # TP=2, FP=1, FN=2, TN=3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert m["confusion_matrix"] == [[3, 1], [2, 2]]


def test_classification_no_positive_prediction():
    m = classification_metrics([0, 1], [0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0


@pytest.mark.parametrize(
    "y_true,y_pred", [([0, 2], [0, 1]), ([0, 1], [0, -1]), ([], []), ([0, 1], [0])]
)
def test_classification_invalid(y_true, y_pred):
    with pytest.raises(MetricInputError):
        classification_metrics(y_true, y_pred)
