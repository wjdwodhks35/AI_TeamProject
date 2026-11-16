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
