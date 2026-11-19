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
    관계가 나온다. 실제 수집 데이터(서울 68개월)에서 기준금리 수준과 평당가의
    상관은 +0.137 이었고, 전년 대비 변화로 바꾸면 -0.310 이었다.

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
