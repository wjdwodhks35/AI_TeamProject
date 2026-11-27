"""규제 자동 입력. 자치구와 계약일만 있으면 정해지는 값이라 사용자가 고르지 않는다."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from apartment_ml.api.main import create_app
from apartment_ml.data.regulation_lookup import RegulationLookup
from tests.fakes import FakeAnomalyModel, FakePriceModel


@pytest.fixture(scope="module")
def lookup() -> RegulationLookup:
    return RegulationLookup.load()


def test_lookup_reflects_designation_history(lookup):
    """2023-06 강남구는 세 규제 모두 걸려 있었고, 노원구는 모두 풀려 있었다."""
    assert all(v == 1 for v in lookup.at("강남구", date(2023, 6, 1)).values())
    assert all(v == 0 for v in lookup.at("노원구", date(2023, 6, 1)).values())


@pytest.fixture
def client():
    app = create_app(FakePriceModel(), FakeAnomalyModel(), load_from_disk=False)
    app.state.regulation_lookup = RegulationLookup.load()
    with TestClient(app) as c:
        yield c


def test_endpoint_returns_flags(client):
    body = client.get("/regulation/강남구", params={"on": "2023-06-01"}).json()
    assert body["speculation_overheated"] is True
    assert body["on"] == "2023-06-01"


def test_endpoint_rejects_non_seoul(client):
    assert client.get("/regulation/수원시").status_code == 404


def test_prediction_fills_regulation_without_user_input(client):
    """화면이 규제를 안 보내도 모델은 그날의 실제 규제 상태로 예측해야 한다."""
    r = client.post(
        "/predict/price",
        json={"district": "강남구", "contract_date": "2023-06-01", "area_m2": 84.9, "floor": 12},
    )
    assert r.status_code == 200
    assert r.json()["predicted_price_krw"] > 0
