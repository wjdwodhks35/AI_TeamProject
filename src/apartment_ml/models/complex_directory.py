"""자치구 → 동 → 단지 목록. API 의 단지 선택 입력을 위한 참조 표.

왜 필요한가
-----------
회귀 모델의 단지 식별자(`complex_id`)는 국토부 단지 일련번호(`11680-2234` 같은 값)다.
사용자가 타이핑할 수 있는 값이 아니므로, 프론트가 **단지명으로 고르고 일련번호를 보내도록**
목록을 내려줘야 한다.

왜 중요한가
-----------
단지를 못 받으면 위치 타깃 인코딩이 '처음 보는 단지'로 떨어져 전체 평균을 쓴다.
같은 모델로 test 를 재면:

- 동·단지 있음 : MAE 1.41억, R² 0.935
- 동·단지 없음 : MAE 4.18억, R² 0.499  ← 기준선(0.603)보다도 낮다

변수 중요도 2위가 `complex_id`(섞으면 MAE +2.52억)인데 그걸 매 요청마다 버리는 셈이었다.

목록은 **학습 데이터에 실제로 등장한 단지**만 담는다. 모델이 본 적 없는 단지를 고르게
해도 추정이 좋아지지 않으므로, 고를 수 있는 것과 모델이 아는 것을 일치시킨다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from apartment_ml import schemas as S

DIRECTORY_FORMAT = 2  # 2: built_year 추가


class ComplexDirectoryError(RuntimeError):
    """단지 목록을 만들거나 읽을 수 없을 때."""


@dataclass(frozen=True)
class ComplexEntry:
    """단지 하나.

    `n_deals` 는 학습 데이터에서의 거래 수(목록 정렬·신뢰도 판단용).
    `built_year` 는 단지마다 고정된 값이라 사용자가 입력할 이유가 없다. 화면에서
    단지를 고르면 바로 채워 넣는다.
    """

    complex_id: str
    apartment_name: str
    dong: str
    n_deals: int
    built_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "complex_id": self.complex_id,
            "apartment_name": self.apartment_name,
            "dong": self.dong,
            "n_deals": self.n_deals,
            "built_year": self.built_year,
        }


class ComplexDirectory:
    """자치구별 단지 목록. `build` 로 만들고 `save`/`load` 로 파일에 담는다."""

    def __init__(self, by_district: dict[str, list[ComplexEntry]]) -> None:
        self._by_district = {k: list(v) for k, v in by_district.items()}

    # -- 생성 ---------------------------------------------------------------
    @classmethod
    def build(cls, df: pd.DataFrame) -> ComplexDirectory:
        """학습 데이터에서 목록을 만든다. 같은 단지가 여러 동으로 들어온 경우 최빈 동을 쓴다."""
        needed = (S.DISTRICT, S.DONG, S.COMPLEX_ID, S.APARTMENT_NAME)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ComplexDirectoryError(f"단지 목록에 필요한 컬럼이 없습니다: {missing}")

        work = df.loc[:, list(needed)].astype("string").dropna()
        # 준공연도는 있으면 쓰고 없으면 비워 둔다 (옛 데이터에는 없을 수 있다).
        built = (
            pd.to_numeric(df.get(S.BUILT_YEAR), errors="coerce").reindex(work.index)
            if S.BUILT_YEAR in df.columns
            else pd.Series(pd.NA, index=work.index, dtype="Float64")
        )
        work = work.assign(_built=built)
        if work.empty:
            raise ComplexDirectoryError("단지 목록을 만들 수 있는 행이 없습니다.")

        grouped = (
            work.groupby([S.DISTRICT, S.COMPLEX_ID], as_index=False)
            .agg(
                apartment_name=(S.APARTMENT_NAME, lambda s: s.mode().iat[0]),
                dong=(S.DONG, lambda s: s.mode().iat[0]),
                n_deals=(S.COMPLEX_ID, "size"),
                built_year=("_built", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
            )
            .sort_values(["dong", "apartment_name"], kind="stable")
        )

        by_district: dict[str, list[ComplexEntry]] = {}
        for district, rows in grouped.groupby(S.DISTRICT, sort=True):
            by_district[str(district)] = [
                ComplexEntry(
                    complex_id=str(r.complex_id),
                    apartment_name=str(r.apartment_name),
                    dong=str(r.dong),
                    n_deals=int(r.n_deals),
                    built_year=None if pd.isna(r.built_year) else int(r.built_year),
                )
                for r in rows.itertuples(index=False)
            ]
        return cls(by_district)

    # -- 조회 ---------------------------------------------------------------
    @property
    def districts(self) -> list[str]:
        return sorted(self._by_district)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_district.values())

    def complexes(self, district: str) -> list[ComplexEntry]:
        """그 자치구의 단지 목록. 없는 자치구면 빈 목록."""
        return list(self._by_district.get(district.strip(), ()))

    def dongs(self, district: str) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self.complexes(district):
            seen.setdefault(entry.dong, None)
        return sorted(seen)

    def has(self, district: str, complex_id: str) -> bool:
        return any(e.complex_id == complex_id for e in self.complexes(district))

    # -- 저장 / 불러오기 -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": DIRECTORY_FORMAT,
            "n_complexes": len(self),
            "districts": {
                k: [e.to_dict() for e in v] for k, v in sorted(self._by_district.items())
            },
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ComplexDirectory:
        source = Path(path)
        if not source.exists():
            raise ComplexDirectoryError(f"단지 목록 파일이 없습니다: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComplexDirectoryError(f"단지 목록 파일을 읽을 수 없습니다: {source}") from exc
        if payload.get("format") != DIRECTORY_FORMAT:
            raise ComplexDirectoryError(
                f"단지 목록 형식이 다릅니다 (파일 {payload.get('format')!r}, "
                f"현재 {DIRECTORY_FORMAT}): {source}"
            )
        districts: dict[str, list[ComplexEntry]] = {}
        for district, rows in payload.get("districts", {}).items():
            districts[district] = [_entry(row) for row in rows]
        return cls(districts)


def _entry(row: dict[str, Any]) -> ComplexEntry:
    try:
        return ComplexEntry(
            complex_id=str(row["complex_id"]),
            apartment_name=str(row["apartment_name"]),
            dong=str(row["dong"]),
            n_deals=int(row["n_deals"]),
            built_year=None if row.get("built_year") is None else int(row["built_year"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ComplexDirectoryError(f"단지 항목이 올바르지 않습니다: {row!r}") from exc


def build_from_frames(frames: Iterable[pd.DataFrame]) -> ComplexDirectory:
    """여러 분할(train/valid 등)을 합쳐 목록을 만든다."""
    return ComplexDirectory.build(pd.concat(list(frames), ignore_index=True))
