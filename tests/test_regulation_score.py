"""전국 공통 규제 상태·규제 점수·이상거래 레이블 방향 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.data.collect import (
    load_regulation_events,
    to_national_regulation_state,
    to_regulation_history,
)
from apartment_ml.data.merge import (
    add_regulation_score,
    attach_national_regulation,
    regulation_score,
)


def events(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """(시행일, 대상, 항목, 값) 목록을 load_regulation_events 결과 모양으로 만든다."""
    frame = pd.DataFrame(rows, columns=["effective_date", "districts", "regulation_type", "value"])
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    return frame.sort_values("effective_date").reset_index(drop=True)


def deals(dates: list[str], district: str = "강남구") -> pd.DataFrame:
    return pd.DataFrame({"district": [district] * len(dates), "contract_date": dates})


# ---------------------------------------------------------------------------
# 전국 공통 규제 상태
# ---------------------------------------------------------------------------


def test_national_state_tracks_multi_level_values():
    """DSR 은 0/1 이 아니라 단계가 올라간다. 나중 이벤트가 앞 값을 덮어쓴다."""
    state = to_national_regulation_state(
        events(
            [
                ("2021-07-01", "ALL", S.REG_DSR_STAGE, 1),
                ("2022-01-01", "ALL", S.REG_DSR_STAGE, 2),
                ("2022-07-01", "ALL", S.REG_DSR_STAGE, 3),
            ]
        )
    )

    assert list(state[S.REG_DSR_STAGE]) == [0, 1, 2, 3]


def test_national_state_is_empty_without_national_events():
    """구별 규제만 있으면 빈 표를 준다 (컬럼 모양은 유지)."""
    state = to_national_regulation_state(
        events([("2021-01-01", "ALL", S.REG_ADJUSTMENT_TARGET, 1)])
    )

    assert state.empty
    assert S.REG_DSR_STAGE in state.columns


# ---------------------------------------------------------------------------
# 이벤트 파일 읽기
# ---------------------------------------------------------------------------


def test_load_events_accepts_multi_level_national_values():
    """실제 파일에 DSR 3단계·주담대 한도 같은 0/1 아닌 값이 들어 있다."""
    loaded = load_regulation_events()
    national = loaded[loaded["regulation_type"].isin(S.NATIONAL_REGULATION_COLUMNS)]

    assert (national["value"] > 1).any()


def test_load_events_rejects_non_binary_district_value(tmp_path):
    """구별 규제는 0/1 이어야 한다. 2 가 들어오면 멈춘다."""
    path = tmp_path / "events.csv"
    path.write_text(
        "effective_date,districts,regulation_type,value,description,source\n"
        f"2021-01-01,ALL,{S.REG_ADJUSTMENT_TARGET},2,설명,출처\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="0 또는 1"):
        load_regulation_events(path)


# ---------------------------------------------------------------------------
# 결합 (계약일 기준)
# ---------------------------------------------------------------------------


def test_attach_national_regulation_does_not_look_ahead():
    """10·15 대책은 10월 15일부터다. 10월 14일 계약에 붙으면 미래 정보가 새는 것이다."""
    state = to_national_regulation_state(
        events(
            [
                ("2025-06-28", "ALL", S.REG_MORTGAGE_CAP, 6),
                ("2025-10-15", "ALL", S.REG_MORTGAGE_CAP, 4),
            ]
        )
    )

    out = attach_national_regulation(deals(["2025-10-14", "2025-10-15"]), state)

    assert out[S.REG_MORTGAGE_CAP].iloc[0] == 6
    assert out[S.REG_MORTGAGE_CAP].iloc[1] == 4


def test_attach_national_regulation_uses_default_before_first_event():
    state = to_national_regulation_state(events([("2025-06-28", "ALL", S.REG_MORTGAGE_CAP, 6)]))

    out = attach_national_regulation(deals(["2021-03-01"]), state)

    assert out[S.REG_MORTGAGE_CAP].iloc[0] == S.MORTGAGE_CAP_NONE
    assert out[S.REG_DSR_STAGE].iloc[0] == 0


def test_attach_national_regulation_raises_when_column_missing():
    bad = pd.DataFrame({"effective_date": ["2025-01-01"], S.REG_DSR_STAGE: [1]})

    with pytest.raises(ValueError, match="전국 규제 상태표에 컬럼이 없습니다"):
        attach_national_regulation(deals(["2025-02-01"]), bad)


# ---------------------------------------------------------------------------
# 규제 점수
# ---------------------------------------------------------------------------


def test_regulation_score_is_zero_without_any_regulation():
    frame = pd.DataFrame({c: [0] for c in S.REGULATION_COLUMNS})
    for column, default in S.NATIONAL_REGULATION_DEFAULTS.items():
        frame[column] = [default]

    assert regulation_score(frame).iloc[0] == pytest.approx(0.0)


def test_regulation_score_sums_every_active_regulation():
    """모든 규제가 걸린 상태의 점수. 가중치 표가 바뀌면 이 값도 함께 바뀐다."""
    frame = pd.DataFrame(
        {
            S.REG_ADJUSTMENT_TARGET: [1],  # -1.0
            S.REG_SPECULATION_OVERHEATED: [1],  # -1.0
            S.REG_LAND_PERMIT: [1],  # -2.0
            S.REG_DSR_STAGE: [3],  # -1.5
            S.REG_STRESS_DSR: [2],  # -1.0
            S.REG_MORTGAGE_CAP: [4],  # -2.0
            S.REG_TRANSFER_TAX_HEAVY: [1],  # -1.0
            S.REG_POLICY_LOAN: [1],  # +1.0
            S.REG_MATURITY_LIMIT: [1],  # -0.5
        }
    )

    assert regulation_score(frame).iloc[0] == pytest.approx(-9.0)


def test_regulation_score_mortgage_cap_takes_lowest_band():
    """한도 4억은 6억 구간에도 해당하지만 더 센 쪽 한 번만 센다."""
    frame = pd.DataFrame({S.REG_MORTGAGE_CAP: [S.MORTGAGE_CAP_NONE, 6, 4]})
    score = regulation_score(frame)

    assert score.iloc[0] == pytest.approx(0.0)
    assert score.iloc[1] == pytest.approx(-1.0)
    assert score.iloc[2] == pytest.approx(-2.0)


def test_regulation_score_change_is_zero_in_stable_period():
    """규제가 바뀌지 않은 구간에서는 전월 대비 변화가 0이다."""
    ev = events(
        [
            ("2021-01-01", "ALL", S.REG_ADJUSTMENT_TARGET, 1),
            ("2021-07-01", "ALL", S.REG_DSR_STAGE, 1),
        ]
    )
    state = to_national_regulation_state(ev)
    out = add_regulation_score(
        deals(["2023-05-10"]), history=to_regulation_history(ev), national_state=state
    )

    assert out[S.REG_SCORE_CHANGE].iloc[0] == pytest.approx(0.0)


def test_regulation_score_change_catches_a_new_regulation():
    """시행 직후 계약은 한 달 전과 점수가 다르다."""
    ev = events(
        [
            ("2021-01-01", "ALL", S.REG_ADJUSTMENT_TARGET, 1),
            ("2025-10-15", "ALL", S.REG_LAND_PERMIT, 1),
        ]
    )
    state = to_national_regulation_state(ev)
    out = add_regulation_score(
        deals(["2025-10-20"]), history=to_regulation_history(ev), national_state=state
    )

    assert out[S.REG_SCORE_CHANGE].iloc[0] == pytest.approx(-2.0)
