"""거래 데이터 + 외부 데이터(기준금리, 규제 이력) 병합.

시점 정합성(Point-in-time) 원칙
-------------------------------
- 기준금리: ``effective_date``(변경 적용일) 이 계약일 **이전 또는 같은 날** 인 가장 최근 값을 붙인다.
  금통위 발표일과 적용일이 다를 수 있으므로 반드시 '적용일' 을 사용한다.
- 규제: ``effective_from <= contract_date < effective_to`` 이면 1, 아니면 0.
  ``announced_date``(발표일)은 기록용이며 적용 판정에는 사용하지 않는다.
  ``effective_to`` 가 비어 있으면 현재까지 유효한 것으로 본다.
- 계약일 이후에 공개된 정보를 붙이지 않는다 (미래 정보 누출 방지).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from apartment_ml import schemas as S

BASE_RATE_COLUMNS = ("effective_date", S.BASE_RATE)
REGULATION_HISTORY_COLUMNS = (
    S.DISTRICT,
    "regulation_type",  # schemas.REGULATION_COLUMNS 중 하나
    "announced_date",  # 선택
    "effective_from",
    "effective_to",  # 선택 (null = 현재까지)
)


def _as_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="raise").dt.normalize().astype("datetime64[ns]")


def attach_base_rate(transactions: pd.DataFrame, base_rates: pd.DataFrame) -> pd.DataFrame:
    """계약일 시점에 유효한 기준금리를 ``base_rate`` 컬럼으로 붙인다.

    첫 적용일보다 이른 계약은 NaN. 기존 ``base_rate`` 컬럼은 덮어쓴다. 원래 행 순서를 유지한다.
    """
    missing = [c for c in BASE_RATE_COLUMNS if c not in base_rates.columns]
    if missing:
        raise ValueError(f"기준금리 데이터에 컬럼이 없습니다: {missing}")
    rates = base_rates.loc[:, list(BASE_RATE_COLUMNS)].copy()
    rates["effective_date"] = _as_datetime(rates["effective_date"])
    rates[S.BASE_RATE] = pd.to_numeric(rates[S.BASE_RATE], errors="raise").astype("float64")
    if rates["effective_date"].duplicated().any():
        raise ValueError("기준금리 데이터에 같은 적용일이 중복되어 있습니다.")
    rates = rates.sort_values("effective_date")

    left = transactions.drop(columns=[S.BASE_RATE], errors="ignore").copy()
    left["_order"] = np.arange(len(left))
    left["_cd"] = _as_datetime(left[S.CONTRACT_DATE])
    merged = pd.merge_asof(
        left.sort_values("_cd", kind="mergesort"),
        rates.rename(columns={"effective_date": "_cd_rate"}),
        left_on="_cd",
        right_on="_cd_rate",
        direction="backward",
    )
    merged = merged.sort_values("_order").drop(columns=["_order", "_cd", "_cd_rate"])
    merged.index = transactions.index
    return merged


def _rate_as_of(dates: pd.Series, rates: pd.DataFrame) -> pd.Series:
    """주어진 날짜들에 유효한 기준금리. 첫 적용일보다 이른 날짜는 NaN."""
    left = pd.DataFrame({"_cd": _as_datetime(dates), "_order": np.arange(len(dates))})
    merged = pd.merge_asof(
        left.sort_values("_cd", kind="mergesort"),
        rates.rename(columns={"effective_date": "_cd_rate", S.BASE_RATE: "_rate"}),
        left_on="_cd",
        right_on="_cd_rate",
        direction="backward",
    )
    merged = merged.sort_values("_order")
    return pd.Series(merged["_rate"].to_numpy(), index=dates.index, dtype="float64")


def attach_base_rate_changes(
    transactions: pd.DataFrame,
    base_rates: pd.DataFrame,
    *,
    months: tuple[int, ...] = (12, 3),
) -> pd.DataFrame:
    """계약일과 그 N개월 전의 기준금리 차이를 붙인다 (%포인트).

    금리는 '수준' 으로 모델에 넣으면 안 된다. 금리가 오르는 시기는 경기가 좋은
    시기이기도 해서, 수준끼리 비교하면 "금리가 높을 때 집값도 높다" 는 거꾸로 된
    관계가 나온다. 수집 자료(2021-01~2026-08, 68개월) 를 월별 평균 평당가로 집계하면
    기준금리 수준과의 상관은 +0.209 (부호가 뒤집힘) 이고, 전년 대비 변화는 -0.752 다.

    다만 월별 시계열은 추세가 강해 상관이 부풀려진다. 이 숫자는 효과 크기가 아니라
    '수준이 아니라 변화를 써야 한다' 는 근거로만 쓴다 (docs/DATA_QUALITY.md).

    ``base_rate`` 가 없으면 ``attach_base_rate`` 를 먼저 적용한다.
    """
    missing = [c for c in BASE_RATE_COLUMNS if c not in base_rates.columns]
    if missing:
        raise ValueError(f"기준금리 데이터에 컬럼이 없습니다: {missing}")

    out = transactions
    if S.BASE_RATE not in out.columns or out[S.BASE_RATE].isna().all():
        out = attach_base_rate(out, base_rates)
    out = out.copy()

    rates = base_rates.loc[:, list(BASE_RATE_COLUMNS)].copy()
    rates["effective_date"] = _as_datetime(rates["effective_date"])
    rates[S.BASE_RATE] = pd.to_numeric(rates[S.BASE_RATE], errors="raise").astype("float64")
    rates = rates.sort_values("effective_date")

    contract = _as_datetime(out[S.CONTRACT_DATE])
    current = pd.to_numeric(out[S.BASE_RATE], errors="coerce").astype("float64")
    for month in months:
        past = _rate_as_of(contract - pd.DateOffset(months=month), rates)
        out[f"base_rate_change_{month}m"] = current - past
    return out


def attach_regulation(transactions: pd.DataFrame, regulations: pd.DataFrame) -> pd.DataFrame:
    """규제 이력 테이블을 자치구·계약일 기준으로 0/1 컬럼(REGULATION_COLUMNS)으로 변환해 붙인다.

    현재 판정 단위는 '자치구' 이다. 동/단지 단위 규제(예: 일부 토지거래허가구역)는
    표현할 수 없으므로 docs/DATA_SCHEMA.md 의 미해결 사항을 참고한다.
    """
    required = [
        c for c in REGULATION_HISTORY_COLUMNS if c not in ("announced_date", "effective_to")
    ]
    missing = [c for c in required if c not in regulations.columns]
    if missing:
        raise ValueError(f"규제 이력 데이터에 컬럼이 없습니다: {missing}")
    unknown = set(regulations["regulation_type"]) - set(S.REGULATION_COLUMNS)
    if unknown:
        raise ValueError(f"알 수 없는 regulation_type: {sorted(unknown)}")

    out = transactions.copy()
    contract = _as_datetime(out[S.CONTRACT_DATE])
    district = out[S.DISTRICT].astype("string")
    start = _as_datetime(regulations["effective_from"])
    if "effective_to" in regulations.columns:
        end = pd.to_datetime(regulations["effective_to"]).astype("datetime64[ns]")
    else:
        end = pd.Series(pd.NaT, index=regulations.index, dtype="datetime64[ns]")

    for col in S.REGULATION_COLUMNS:
        flag = pd.Series(False, index=out.index)
        rows = regulations.index[regulations["regulation_type"] == col]
        for i in rows:
            active = (district == regulations.at[i, S.DISTRICT]) & (contract >= start[i])
            if pd.notna(end[i]):
                active &= contract < end[i]
            flag |= active.fillna(False).astype(bool)
        out[col] = flag.astype("Int64")
    return out


# ---------------------------------------------------------------------------
# 금리 10종
# ---------------------------------------------------------------------------
# 기준금리(base_rate)는 위의 attach_base_rate/attach_base_rate_changes 가 변경일 기준으로
# 정확히 붙인다. 여기서 다루는 것은 나머지 9종이다.
#
# 공개 시차: 주담대·집단대출·전세자금대출·가계대출은 한국은행이 월 가중평균을 다음 달에
# 공개한다. 10월 거래에 10월 금리를 붙이면 계약 시점에 없던 정보를 쓰게 되므로 한 달
# 밀려서 붙인다. 국고채·CD 는 매일 관측되지만 여기 표가 월평균이라 같은 기준을 적용한다.

RATE_SERIES_LAG_MONTHS = 1
MONTHLY_RATE_COLUMNS: tuple[str, ...] = tuple(c for c in S.RATE_COLUMNS if c != S.BASE_RATE)


def _ym(dates: pd.Series) -> pd.Series:
    return _as_datetime(dates).dt.strftime("%Y%m").astype("string")


def _shift_ym(ym: pd.Series, months: int) -> pd.Series:
    when = pd.to_datetime(ym.astype("string"), format="%Y%m")
    return (when + pd.DateOffset(months=months)).dt.strftime("%Y%m").astype("string")


def attach_rate_series(
    transactions: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    columns: Sequence[str] = MONTHLY_RATE_COLUMNS,
    months: tuple[int, ...] = S.RATE_CHANGE_MONTHS,
    lag: int = RATE_SERIES_LAG_MONTHS,
) -> pd.DataFrame:
    """월별 금리 표(``ym`` × 금리) 를 계약 월에 붙이고 변화량을 함께 만든다.

    변화량은 월별 시계열에서 먼저 계산한다 (12개월 전·3개월 전 대비). 거래별로 계산하면
    같은 달 거래끼리 값이 달라질 이유가 없는데도 계산량만 늘어난다.

    ``lag`` 개월 밀려서 붙는다. 기본 1개월인 이유는 위 주석 참고.
    """
    if "ym" not in rates.columns:
        raise ValueError("금리 표에 'ym' 컬럼이 필요합니다.")
    missing = [c for c in columns if c not in rates.columns]
    if missing:
        raise ValueError(f"금리 표에 컬럼이 없습니다: {missing}")

    table = rates.loc[:, ["ym", *columns]].copy()
    table["ym"] = table["ym"].astype("string")
    table = table.sort_values("ym").reset_index(drop=True)
    for column in columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("float64")
        for month in months:
            table[S.rate_change_column(column, month)] = table[column].diff(month)

    table["ym"] = _shift_ym(table["ym"], lag)

    out = transactions.copy()
    key = _ym(out[S.CONTRACT_DATE])
    merged = pd.DataFrame({"ym": key}).merge(table, on="ym", how="left")
    for column in table.columns:
        if column != "ym":
            out[column] = merged[column].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 전국 공통 규제 + 규제 점수
# ---------------------------------------------------------------------------


def _state_as_of(dates: pd.Series, state: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``effective_date`` 가 각 날짜 이전(같은 날 포함) 인 가장 최근 상태를 고른다."""
    table = state.loc[:, ["effective_date", *columns]].copy()
    table["effective_date"] = _as_datetime(table["effective_date"])
    table = table.sort_values("effective_date").reset_index(drop=True)

    want = pd.DataFrame({"_when": _as_datetime(dates)})
    want["_order"] = range(len(want))
    picked = pd.merge_asof(
        want.sort_values("_when"),
        table.rename(columns={"effective_date": "_when"}),
        on="_when",
        direction="backward",
    )
    return picked.sort_values("_order").loc[:, list(columns)].reset_index(drop=True)


def attach_national_regulation(
    transactions: pd.DataFrame, national_state: pd.DataFrame
) -> pd.DataFrame:
    """전국 공통 규제를 계약일 기준으로 붙인다.

    기준금리와 같은 방식이다. 시행일이 계약일 이전(같은 날 포함) 인 가장 최근 상태를 쓴다.
    월 단위로 붙이면 10·15 대책이 10월 1~14일 계약에도 붙어 미래 정보가 섞인다.
    """
    columns = list(S.NATIONAL_REGULATION_COLUMNS)
    missing = [c for c in columns if c not in national_state.columns]
    if missing:
        raise ValueError(f"전국 규제 상태표에 컬럼이 없습니다: {missing}")

    out = transactions.copy()
    picked = _state_as_of(out[S.CONTRACT_DATE], national_state, columns)
    for column in columns:
        value = pd.to_numeric(picked[column], errors="coerce")
        # 상태표의 첫 줄보다 이른 계약은 아직 아무 규제도 없던 시점이다.
        value = value.fillna(S.NATIONAL_REGULATION_DEFAULTS[column])
        out[column] = value.astype("Int64").to_numpy()
    return out


def regulation_score(frame: pd.DataFrame) -> pd.Series:
    """그 시점에 걸려 있는 규제들의 가중합.

    누적 합계가 아니라 '지금 걸려 있는 것' 만 더한다. 누적으로 만들면 값이 한 방향으로만
    움직여서 모델이 규제가 아니라 경과 시간을 학습한다.
    """
    score = pd.Series(0.0, index=frame.index)
    for column, weight in S.REGULATION_WEIGHTS.items():
        if column in frame.columns:
            value = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("float64")
            score = score + weight * value

    if S.REG_MORTGAGE_CAP in frame.columns:
        cap = pd.to_numeric(frame[S.REG_MORTGAGE_CAP], errors="coerce")
        applied = pd.Series(0.0, index=frame.index)
        decided = pd.Series(False, index=frame.index)
        for limit, weight in S.MORTGAGE_CAP_WEIGHTS:  # 낮은 한도부터 본다
            hit = (~decided) & cap.notna() & (cap <= limit)
            applied[hit] = weight
            decided = decided | hit
        score = score + applied
    return score.round(2)


def add_regulation_score(
    transactions: pd.DataFrame,
    *,
    history: pd.DataFrame,
    national_state: pd.DataFrame,
) -> pd.DataFrame:
    """규제 점수와 전월 대비 변화를 붙인다.

    점수를 계산할 규제 플래그는 이 함수가 직접 붙인다. 호출하는 쪽에서 이미 붙여
    두었는지에 따라 값이 달라지면 안 되기 때문이다.

    변화량은 계약일 한 달 전 시점의 규제 상태를 다시 구해서 뺀다. 월별 표를 만들어
    diff 를 쓰면 월 중간에 시행된 규제가 한 달 밀리거나 당겨진다.
    """
    out = transactions.copy()
    keys = out.loc[:, [S.DISTRICT, S.CONTRACT_DATE]].copy()

    def state_at(when: pd.Series) -> pd.DataFrame:
        frame = keys.copy()
        frame[S.CONTRACT_DATE] = when
        frame = attach_regulation(frame, history)
        return attach_national_regulation(frame, national_state)

    now = _as_datetime(keys[S.CONTRACT_DATE])
    out[S.REG_SCORE] = regulation_score(state_at(now)).to_numpy()
    earlier = regulation_score(state_at(now - pd.DateOffset(months=1))).to_numpy()
    out[S.REG_SCORE_CHANGE] = (out[S.REG_SCORE] - earlier).round(2)
    return out
