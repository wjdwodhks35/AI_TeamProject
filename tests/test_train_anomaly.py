"""합성 테스트 케이스로 분류 학습 경로와 백엔드 연동을 검증한다."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from scripts.train_anomaly import train_anomaly

import apartment_ml.api.main as api
from apartment_ml import schemas as S
from apartment_ml.config import get_settings
from apartment_ml.models.base import LabelsNotAvailableError, TrainingDisabledError
from tests.conftest import make_rows
from tests.test_api import FEATURES


@pytest.fixture
def labelled_csv(tmp_path):
    rows = make_rows(60)
    # 코드 경로 검증용 테스트 픽스처. 임시 디렉터리 안에서만 사용한다.
    rows[S.ANOMALY_LABEL] = [0, 1] * 30
    rows.loc[3, S.ANOMALY_LABEL] = None
    path = tmp_path / "fixture.csv"
    rows.sample(frac=1, random_state=42).to_csv(path, index=False)
    return path


@pytest.fixture
def training_settings(tmp_path):
    return replace(
        get_settings(),
        classification_training_enabled=True,
        anomaly_model_path=tmp_path / "anomaly_model.joblib",
    )


def test_train_compare_save_and_serve(labelled_csv, training_settings, monkeypatch):
    report = train_anomaly(labelled_csv, "pytest fixture", training_settings)
    splits = report["splits"]
    assert splits["train"]["end"] < splits["valid"]["start"]
    assert splits["valid"]["end"] < splits["test"]["start"]
    assert splits["train"]["unlabelled"] == 1
    valid, test = report["validation_metrics"], report["test_metrics"]
    assert (
        set(valid)
        == set(test)
        == {
            "decision_tree",
            "random_forest",
            "hist_gradient_boosting",
            "voting",
        }
    )
    assert report["selected_model"] == max(
        valid, key=lambda name: (valid[name]["f1"], valid[name]["recall"], valid[name]["precision"])
    )
    for metrics in test.values():
        matrix = metrics["confusion_matrix"]
        assert [sum(row) for row in matrix] == [
            splits["test"]["normal"],
            splits["test"]["suspicious"],
        ]
    report_path = training_settings.anomaly_model_path.with_name("anomaly_model_metrics.json")
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(api, "get_settings", lambda: training_settings)
    with TestClient(api.create_app()) as client:
        assert client.get("/health").json()["models"]["anomaly"] is True
        response = client.post("/predict/anomaly", json={**FEATURES, "price_krw": 900_000_000})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["suspicion_score"] <= 1
    assert body["is_suspicious"] == (body["suspicion_score"] >= body["threshold"])


def test_rejects_repository_sample(training_settings):
    with pytest.raises(LabelsNotAvailableError, match="TEST_ONLY"):
        train_anomaly(training_settings.path("sample_transactions"), "sample", training_settings)
    assert not training_settings.anomaly_model_path.exists()


def test_requires_enabled_training(labelled_csv, training_settings):
    settings = replace(training_settings, classification_training_enabled=False)
    with pytest.raises(TrainingDisabledError):
        train_anomaly(labelled_csv, "pytest fixture", settings)
    assert not settings.anomaly_model_path.exists()
