"""자치구·계약일로 규제 상태를 찾아준다.

왜 필요한가
-----------
투기과열지구·조정대상지역·토지거래허가구역은 **고시로 정해진 사실**이다. 사용자가
알아맞혀 입력할 값이 아니고, 틀리게 넣으면 예측만 나빠진다. 자치구와 계약일만 있으면
``configs/regulation_events.csv`` 에서 바로 결정된다.

학습 때 쓰는 ``merge.attach_regulation`` 을 그대로 다시 쓴다. 따로 구현하면 학습과
서빙의 판정이 어긋날 수 있다.

한계: 판정 단위가 자치구다. 토지거래허가구역처럼 실제로는 동·단지 단위인 규제는
그 자치구 전체로 퍼진다 (docs/DATA_SCHEMA.md).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.data.collect import (
    REGULATION_EVENTS_PATH,
    load_regulation_events,
    to_regulation_history,
)
from apartment_ml.data.merge import attach_regulation


class RegulationLookupError(RuntimeError):
    """규제 이력을 읽을 수 없을 때."""


class RegulationLookup:
    """고시 이력을 한 번 읽어 두고 (자치구, 날짜) 로 조회한다."""

    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    @classmethod
    def load(cls, path: str | Path = REGULATION_EVENTS_PATH) -> RegulationLookup:
        try:
            return cls(to_regulation_history(load_regulation_events(path)))
        except (OSError, ValueError) as exc:
            raise RegulationLookupError(f"규제 이력을 읽을 수 없습니다: {path}") from exc

    def at(self, district: str, on: date) -> dict[str, int]:
        """그 날 그 자치구에 걸려 있던 규제. 값은 0 또는 1."""
        frame = pd.DataFrame({S.DISTRICT: [str(district)], S.CONTRACT_DATE: [pd.Timestamp(on)]})
        out = attach_regulation(frame, self._history)
        return {col: int(out.at[0, col]) for col in S.REGULATION_COLUMNS}
