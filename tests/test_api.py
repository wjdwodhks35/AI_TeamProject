import pytest
from fastapi.testclient import TestClient

from apartment_ml.api.main import create_app
from tests.fakes import FakeAnomalyModel, FakePriceModel

FEATURES = {
    "district": "강남구",
    "contract_date": "2024-05-20",
    "area_m2": 84.9,
    "floor": 12,
    "built_year": 2008,
    "base_rate": 3.5,
}


@pytest.fixture
def client():
    with TestClient(create_app(load_from_disk=False)) as c:
        yield c


@pytest.fixture
def ready_client():
    app = create_app(FakePriceModel(), FakeAnomalyModel(), load_from_disk=False)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"] == {"price": False, "anomaly": False}


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/predict/price", FEATURES),
        ("/predict/anomaly", {**FEATURES, "price_krw": 2_500_000_000}),
        (
            "/scenario/interest-rate",
            {"features": FEATURES, "base_rate_before": 3.5, "base_rate_after": 2.5},
        ),
    ],
)
def test_model_not_ready_returns_503(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "MODEL_NOT_READY"
    assert "predicted_price_krw" not in r.json()  # 가짜 예측값 없음


@pytest.mark.parametrize(
    "payload",
    [
        {**FEATURES, "district": "부산진구"},
        {**FEATURES, "area_m2": -1},
        {**FEATURES, "contract_date": "2024-02-31"},
        {k: v for k, v in FEATURES.items() if k != "floor"},
        {**FEATURES, "price_krw": 1},  # 가격 예측 입력에 실거래가 포함 금지
    ],
)
def test_validation_error_422(client, payload):
    assert client.post("/predict/price", json=payload).status_code == 422


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/predict/price", "/predict/anomaly", "/scenario/interest-rate"):
        assert p in paths


def test_endpoints_with_injected_fake_models(ready_client):
    """API 배선 확인용 — 가짜 모델은 tests/ 에만 존재한다."""
    assert ready_client.get("/health").json()["models"] == {"price": True, "anomaly": True}
    r = ready_client.post("/predict/price", json=FEATURES)
    assert r.status_code == 200
    assert r.json()["model_name"].endswith("TEST_ONLY")

    r = ready_client.post("/predict/anomaly", json={**FEATURES, "price_krw": 900_000_000})
    assert r.status_code == 200
    body = r.json()
    assert body["is_suspicious"] is True and body["reference_price_krw"] is not None

    r = ready_client.post(
        "/scenario/interest-rate",
        json={"features": FEATURES, "base_rate_before": 3.5, "base_rate_after": 2.5},
    )
    assert r.status_code == 200
    assert r.json()["price_diff_krw"] == pytest.approx(10_000_000)


def test_validation_error_contract(client):
    response = client.post("/predict/price", json={**FEATURES, "district": "unknown"})
    detail = response.json()["detail"]
    assert detail["code"] == "VALIDATION_ERROR"
    assert detail["issues"][0]["location"] == ["body", "district"]
    assert response.headers["X-Request-ID"]
    assert (
        client.post(
            "/predict/price", content="{", headers={"Content-Type": "application/json"}
        ).json()["detail"]["code"]
        == "VALIDATION_ERROR"
    )


def test_cors(client):
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/predict/price", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == headers["Origin"]
    headers["Origin"] = "https://untrusted.example"
    assert (
        "access-control-allow-origin"
        not in client.options("/predict/price", headers=headers).headers
    )


@pytest.mark.parametrize("failure", [RuntimeError("private server detail"), float("nan")])
def test_prediction_failure_is_safe(failure):
    class BrokenModel(FakePriceModel):
        def predict(self, X):
            if isinstance(failure, Exception):
                raise failure
            return [failure]

    with TestClient(
        create_app(BrokenModel(), load_from_disk=False), raise_server_exceptions=False
    ) as client:
        response = client.post(
            "/predict/price", json=FEATURES, headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 500
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert response.json()["detail"]["code"] == "INTERNAL_ERROR"
        assert "private server detail" not in response.text
        assert response.headers["X-Request-ID"]


def test_anomaly_requires_real_price(client):
    assert client.post("/predict/anomaly", json=FEATURES).status_code == 422


def test_anomaly_without_price_model():
    with TestClient(create_app(anomaly_model=FakeAnomalyModel(), load_from_disk=False)) as c:
        result = c.post("/predict/anomaly", json={**FEATURES, "price_krw": 900_000_000})
        assert result.status_code == 200
        assert result.json()["reference_price_krw"] is None
        assert result.json()["warnings"]


def test_unsupported_scenario():
    model = FakePriceModel()
    model.feature_columns = ["area_m2"]
    with TestClient(create_app(model, load_from_disk=False)) as c:
        response = c.post(
            "/scenario/interest-rate",
            json={"features": FEATURES, "base_rate_before": 3, "base_rate_after": 2},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "SCENARIO_NOT_SUPPORTED"


def test_broken_artifact_does_not_stop_server(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from apartment_ml.api import main

    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"not a joblib model")
    missing = tmp_path / "missing.joblib"
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(price_model_path=broken, anomaly_model_path=missing),
    )
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["models"] == {"price": False, "anomaly": False}
        assert client.post("/predict/price", json=FEATURES).status_code == 503


def test_model_not_ready_during_prediction():
    from apartment_ml.models.base import ModelNotReadyError

    class Unready(FakePriceModel):
        def predict(self, X):
            raise ModelNotReadyError("unavailable")

    with TestClient(create_app(Unready(), load_from_disk=False)) as client:
        assert client.post("/predict/price", json=FEATURES).status_code == 503
