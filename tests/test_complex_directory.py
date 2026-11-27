"""단지 목록: 만들기 · 조회 · 파일 왕복."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from apartment_ml.models.complex_directory import ComplexDirectory, ComplexDirectoryError


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": ["강남구", "강남구", "강남구", "마포구"],
            "dong": ["대치동", "대치동", "역삼동", "서교동"],
            # 같은 단지가 두 번 거래됐고, 한 번은 동이 잘못 들어온 경우를 섞는다
            "complex_id": ["11680-1", "11680-1", "11680-2", "11440-1"],
            "apartment_name": ["은마", "은마", "래미안", "메세나폴리스"],
        }
    )


def test_build_groups_by_district_and_complex():
    directory = ComplexDirectory.build(frame())
    assert directory.districts == ["강남구", "마포구"]
    assert len(directory) == 3  # 같은 단지의 두 거래는 한 항목으로 합쳐진다

    gangnam = directory.complexes("강남구")
    assert [e.apartment_name for e in gangnam] == ["은마", "래미안"]  # 동 → 단지명 순
    assert gangnam[0].n_deals == 2
    assert directory.dongs("강남구") == ["대치동", "역삼동"]
    assert directory.has("강남구", "11680-1")
    assert not directory.has("마포구", "11680-1")  # 자치구가 다르면 없는 것으로 본다
    assert directory.complexes("없는구") == []


def test_save_load_roundtrip(tmp_path):
    original = ComplexDirectory.build(frame())
    path = original.save(tmp_path / "complexes.json")
    restored = ComplexDirectory.load(path)
    assert restored.districts == original.districts
    assert [e.to_dict() for e in restored.complexes("강남구")] == [
        e.to_dict() for e in original.complexes("강남구")
    ]


def test_rejects_bad_input(tmp_path):
    with pytest.raises(ComplexDirectoryError, match="컬럼"):
        ComplexDirectory.build(pd.DataFrame({"district": ["강남구"]}))
    with pytest.raises(ComplexDirectoryError, match="행이 없습니다"):
        ComplexDirectory.build(frame().iloc[0:0])
    with pytest.raises(ComplexDirectoryError, match="없습니다"):
        ComplexDirectory.load(tmp_path / "nope.json")

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"format": 999, "districts": {}}), encoding="utf-8")
    with pytest.raises(ComplexDirectoryError, match="형식이 다릅니다"):
        ComplexDirectory.load(stale)
