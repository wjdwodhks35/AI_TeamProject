"""공통 fixture. 테스트 중 외부 네트워크 호출을 차단한다."""

from __future__ import annotations

import socket
import threading

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair
    local = threading.local()

    def guard(*args, **kwargs):
        if getattr(local, "creating_socketpair", False):
            return original_connect(*args, **kwargs)
        raise RuntimeError("테스트에서 외부 네트워크 호출은 금지되어 있습니다.")

    def local_socketpair(*args, **kwargs):
        # Windows asyncio의 내부 wakeup socketpair만 허용한다.
        local.creating_socketpair = True
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            local.creating_socketpair = False

    monkeypatch.setattr(socket, "socketpair", local_socketpair)
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
            # 회귀의 위치 인코딩이 그룹을 만들 수 있도록 동·단지를 돌려 쓴다.
            "dong": [["역삼동", "서교동"][i % 2] for i in range(n)],
            "complex_id": [f"cx{i % 5}" for i in range(n)],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_rows()
