"""금리 10종 읽기·결합 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apartment_ml import schemas as S
from apartment_ml.data.collect import CollectError, read_rate_series
from apartment_ml.data.merge import RATE_SERIES_LAG_MONTHS, attach_rate_series


def write_series(folder: Path, name: str, values: dict[str, str]) -> None:
    """수집 스크립트가 저장하는 모양대로 파일 하나를 만든다."""
    payload = {
        "stat_code": "722Y001",
        "item_code": "0101000",
        "name": name,
        "cycle": "M",
        "count": len(values),
        "rows": [{"ym": ym, "value": v} for ym, v in values.items()],
    }
    (folder / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def months(start: int, count: int, step: float = 0.0, first: float = 1.0) -> dict[str, str]:
    out = {}
    year, month = divmod(start, 100)
    for i in range(count):
        out[f"{year}{month:02d}"] = str(round(first + step * i, 3))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def test_read_rate_series_builds_one_column_per_item(tmp_path: Path):
    """파일 하나가 금리 한 종이고, 한글 항목명으로 내부 컬럼명을 찾는다."""
    write_series(tmp_path, "기준금리", {"202101": "0.5", "202102": "0.75"})
    write_series(tmp_path, "국고채3년", {"202101": "0.975", "202102": "0.995"})

    out = read_rate_series(tmp_path)

    assert list(out.columns) == ["ym", *S.RATE_COLUMNS]
    assert out.loc[0, S.BASE_RATE] == pytest.approx(0.5)
    assert out.loc[1, S.TREASURY_3Y] == pytest.approx(0.995)
    # 받지 못한 항목은 칸만 만들어 두고 비워 둔다.
    assert out[S.GROUP_LOAN_RATE].isna().all()


def test_read_rate_series_skips_blank_values(tmp_path: Path):
    """한국은행이 아직 올리지 않은 달은 빈 문자열로 온다. 그 달은 비워 둔다."""
    write_series(tmp_path, "주담대", {"202101": "2.63", "202102": "", "202103": "2.70"})

    out = read_rate_series(tmp_path).set_index("ym")

    assert out.loc["202101", S.MORTGAGE_RATE] == pytest.approx(2.63)
    assert "202102" not in out.index
    assert out.loc["202103", S.MORTGAGE_RATE] == pytest.approx(2.70)


def test_read_rate_series_ignores_unknown_item_names(tmp_path: Path):
    """RATE_SERIES 에 없는 항목은 건너뛴다 (나중에 다른 통계를 더 받아도 깨지지 않는다)."""
    write_series(tmp_path, "기준금리", {"202101": "0.5"})
    write_series(tmp_path, "회사채BBB", {"202101": "8.1"})

    out = read_rate_series(tmp_path)

    assert out.loc[0, S.BASE_RATE] == pytest.approx(0.5)
    assert "회사채BBB" not in out.columns


def test_read_rate_series_raises_when_folder_is_empty(tmp_path: Path):
    with pytest.raises(CollectError, match="금리 JSON 이 없습니다"):
        read_rate_series(tmp_path)


def test_read_rate_series_raises_when_nothing_matches(tmp_path: Path):
    """파일은 있는데 아는 항목이 하나도 없으면 조용히 빈 표를 주지 않고 멈춘다."""
    write_series(tmp_path, "회사채BBB", {"202101": "8.1"})

    with pytest.raises(CollectError, match="읽을 수 있는 금리 항목이 없습니다"):
        read_rate_series(tmp_path)


RATES = pd.DataFrame(
    {
        "ym": [f"2022{m:02d}" for m in range(1, 13)] + [f"2023{m:02d}" for m in range(1, 4)],
        S.MORTGAGE_RATE: [3.0 + 0.1 * i for i in range(15)],
    }
)


def deals(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"district": ["강남구"] * len(dates), "contract_date": dates})


def test_attach_rate_series_lags_by_one_month():
    """대출금리는 다음 달에 공개되므로 계약 월이 아니라 전월 값이 붙는다."""
    out = attach_rate_series(deals(["2023-02-20"]), RATES, columns=[S.MORTGAGE_RATE], months=())

    assert RATE_SERIES_LAG_MONTHS == 1
    # 2023-02 계약 -> 2023-01 금리 (= 3.0 + 0.1*12)
    assert out[S.MORTGAGE_RATE].iloc[0] == pytest.approx(4.2)


def test_attach_rate_series_computes_year_on_year_change():
    """변화량은 월별 시계열에서 12개월 전과 비교해 만든다."""
    out = attach_rate_series(deals(["2023-02-20"]), RATES, columns=[S.MORTGAGE_RATE], months=(12,))

    column = S.rate_change_column(S.MORTGAGE_RATE, 12)
    # 2023-01(4.2) - 2022-01(3.0) = 1.2
    assert out[column].iloc[0] == pytest.approx(1.2)


def test_attach_rate_series_leaves_null_before_series_starts():
    """금리 자료보다 이른 계약은 비워 둔다 (앞의 값을 끌어다 쓰지 않는다)."""
    out = attach_rate_series(deals(["2021-05-01"]), RATES, columns=[S.MORTGAGE_RATE], months=())

    assert pd.isna(out[S.MORTGAGE_RATE].iloc[0])


def test_attach_rate_series_keeps_row_count():
    out = attach_rate_series(
        deals(["2023-02-01", "2023-03-01", "2021-01-01"]),
        RATES,
        columns=[S.MORTGAGE_RATE],
        months=(12,),
    )
    assert len(out) == 3


def test_attach_rate_series_raises_without_ym_column():
    with pytest.raises(ValueError, match="'ym' 컬럼이 필요합니다"):
        attach_rate_series(deals(["2023-02-01"]), RATES.drop(columns=["ym"]))


def test_attach_rate_series_raises_when_column_missing():
    with pytest.raises(ValueError, match="금리 표에 컬럼이 없습니다"):
        attach_rate_series(deals(["2023-02-01"]), RATES, columns=[S.TREASURY_3Y])
