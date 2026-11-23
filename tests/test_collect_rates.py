"""금리 수집 스크립트의 응답 처리 테스트.

ECOS 는 일부 필드를 null 로 보낸다. 예전에 ``get(key, "")`` 로 읽다가 f-string 포맷에서
``unsupported format string passed to NoneType.__format__`` 로 죽은 적이 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import collect_rates as cr  # noqa: E402

from apartment_ml import schemas as S  # noqa: E402
from apartment_ml.data.collect import CollectError  # noqa: E402


def test_targets_match_schema_names():
    """저장할 이름이 RATE_SERIES 의 한글 이름과 같아야 read_rate_series 가 찾는다."""
    known = {name for _, name in S.RATE_SERIES}
    assert {name for _, _, name in cr.TARGETS} == known


def test_text_of_turns_null_into_empty_string():
    assert cr.text_of({"UNIT_NAME": None}, "UNIT_NAME") == ""
    assert cr.text_of({}, "UNIT_NAME") == ""
    assert cr.text_of({"UNIT_NAME": "%"}, "UNIT_NAME") == "%"


def test_strip_footnote_removes_trailing_marker():
    assert cr.strip_footnote("예금은행 가중평균금리1)") == "예금은행 가중평균금리"
    assert cr.strip_footnote("국고채(3년)") == "국고채(3년)"
    assert cr.strip_footnote("") == ""


def test_request_series_raises_on_api_error(monkeypatch: pytest.MonkeyPatch):
    """인증키가 틀리면 ECOS 가 RESULT 로 오류를 돌려준다. 조용히 넘기지 않는다."""

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}}

    monkeypatch.setattr(cr.requests, "get", lambda *a, **k: Response())

    with pytest.raises(CollectError, match="INFO-100"):
        cr.request_series("key", "722Y001", "0101000", "202101", "202112")


def test_request_series_returns_rows(monkeypatch: pytest.MonkeyPatch):
    rows = [{"TIME": "202101", "DATA_VALUE": "0.5", "UNIT_NAME": None}]

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"StatisticSearch": {"row": rows}}

    monkeypatch.setattr(cr.requests, "get", lambda *a, **k: Response())

    assert cr.request_series("key", "722Y001", "0101000", "202101", "202112") == rows
