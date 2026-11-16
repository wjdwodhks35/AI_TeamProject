"""공통 fixture. 테스트 중 외부 네트워크 호출을 차단한다."""

from __future__ import annotations

import socket

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*args, **kwargs):
        raise RuntimeError("테스트에서 외부 네트워크 호출은 금지되어 있습니다.")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def make_rows(n: int = 10, start: str = "2023-01-01", freq: str = "D") -> pd.DataFrame:
    """테스트용 합성 거래 (가짜 데이터)."""
    dates = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {
            "district": ["강남구", "마포구"] * (n // 2) + ["강남구"] * (n % 2),
            "contract_date": dates.strftime("%Y-%m-%d"),
            "area_m2": [59.9 + i for i in range(n)],
            "floor": [i % 20 + 1 for i in range(n)],
            "built_year": [2000 + i % 20 for i in range(n)],
            "price_krw": [500_000_000 + i * 10_000_000 for i in range(n)],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_rows()
