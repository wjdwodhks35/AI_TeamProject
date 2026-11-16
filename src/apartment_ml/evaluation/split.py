"""계약일 기준 시간순 데이터 분할.

- '분할 기준으로서의 계약일' 과 '모델 입력 변수로서의 계약일(연·월 파생변수)' 은 별개다.
  이 모듈은 전자만 다루며, 입력 변수 구성에는 관여하지 않는다.
- 같은 계약일의 거래는 항상 같은 구간에 들어간다 (경계에서 임의로 섞이지 않음).
- 구간 순서: train(가장 과거) < valid < test(가장 최근). 미래 거래는 과거 구간에 들어가지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apartment_ml.schemas import CONTRACT_DATE


class InsufficientDataError(ValueError):
    """분할에 필요한 데이터가 부족할 때."""


@dataclass
class TimeSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    date_col: str = CONTRACT_DATE

    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "valid": self.valid, "test": self.test}

    def summary(self) -> pd.DataFrame:
        """구간별 건수와 날짜 범위."""
        rows = []
        for name, part in self.parts().items():
            dates = part[self.date_col]
            rows.append(
                {
                    "split": name,
                    "rows": len(part),
                    "start": dates.min() if len(part) else pd.NaT,
                    "end": dates.max() if len(part) else pd.NaT,
                }
            )
        return pd.DataFrame(rows).set_index("split")


def _prepare(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"날짜 컬럼이 없습니다: {date_col}")
    if df.empty:
        raise InsufficientDataError("분할할 데이터가 비어 있습니다.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    if out[date_col].isna().any():
        raise ValueError(f"{date_col} 에 비어 있거나 잘못된 날짜가 있습니다.")
    return out.sort_values(date_col, kind="mergesort")  # 안정 정렬: 같은 날짜 내 원래 순서 유지


def _check(split: TimeSplit, min_rows: int) -> TimeSplit:
    for name, part in split.parts().items():
        if len(part) < min_rows:
            raise InsufficientDataError(
                f"{name} 구간 데이터가 부족합니다 ({len(part)}건 < 최소 {min_rows}건). "
                "데이터를 늘리거나 분할 비율/경계 날짜를 조정하세요."
            )
    tr_end, va_start = split.train[split.date_col].max(), split.valid[split.date_col].min()
    va_end, te_start = split.valid[split.date_col].max(), split.test[split.date_col].min()
    if not (tr_end < va_start and va_end < te_start):  # 내부 불변식
        raise AssertionError("시간순 분할 불변식 위반")
    return split


def split_by_dates(
    df: pd.DataFrame,
    valid_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    date_col: str = CONTRACT_DATE,
    min_rows: int = 1,
) -> TimeSplit:
    """경계 날짜로 분할: train < valid_start <= valid < test_start <= test."""
    vs, ts = pd.Timestamp(valid_start).normalize(), pd.Timestamp(test_start).normalize()
    if not vs < ts:
        raise ValueError("valid_start 는 test_start 보다 이전이어야 합니다.")
    data = _prepare(df, date_col)
    d = data[date_col]
    return _check(
        TimeSplit(
            train=data[d < vs],
            valid=data[(d >= vs) & (d < ts)],
            test=data[d >= ts],
            date_col=date_col,
        ),
        min_rows,
    )


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = CONTRACT_DATE,
    min_rows: int = 1,
) -> TimeSplit:
    """비율로 시간순 분할. 경계는 날짜 단위로 맞춘다.

    정렬 후 ``round(n * train_ratio)`` 번째 행의 계약일을 valid 시작일로,
    ``round(n * (train_ratio + valid_ratio))`` 번째 행의 계약일을 test 시작일로 삼는다.
    경계 날짜의 거래는 모두 뒤 구간에 들어가므로 실제 비율은 근사값이다.
    """
    ratios = (train_ratio, valid_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"비율은 모두 양수이고 합이 1 이어야 합니다: {ratios}")
    data = _prepare(df, date_col)
    n = len(data)
    if n < 3 * min_rows:
        raise InsufficientDataError(f"데이터가 {n}건으로 3개 구간 분할에 부족합니다.")
    dates = data[date_col].to_numpy()
    i_valid = min(max(int(round(n * train_ratio)), 1), n - 1)
    i_test = min(max(int(round(n * (train_ratio + valid_ratio))), i_valid), n - 1)
    if not dates[0] < dates[i_valid] < dates[i_test]:
        raise InsufficientDataError(
            "고유 계약일이 부족해 같은 날짜를 나누지 않고는 3개 구간을 만들 수 없습니다."
        )
    return split_by_dates(data, dates[i_valid], dates[i_test], date_col=date_col, min_rows=min_rows)


def expanding_window_splits(
    df: pd.DataFrame, n_splits: int, date_col: str = CONTRACT_DATE
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """하이퍼파라미터 조정용 시간순 교차검증 (확장 윈도우).

    고유 계약일을 ``n_splits + 1`` 개 구간으로 나눈 뒤, k 번째 fold 는
    앞의 k 개 구간으로 학습하고 다음 1 개 구간으로 검증한다. 같은 날짜는 같은 구간.
    반환되는 인덱스는 ``df`` 의 **위치(position) 인덱스** 이다 (``df.iloc`` 용).
    학습 데이터(train) 안에서만 호출하여 test 구간을 참조하지 않도록 한다.
    """
    if n_splits < 1:
        raise ValueError("n_splits 는 1 이상이어야 합니다.")
    if date_col not in df.columns:
        raise KeyError(f"날짜 컬럼이 없습니다: {date_col}")
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{date_col} 에 비어 있거나 잘못된 날짜가 있습니다.")
    unique = np.sort(dates.unique())
    if len(unique) < n_splits + 1:
        raise InsufficientDataError(
            f"고유 계약일 {len(unique)}개로 {n_splits}-fold 시간순 검증을 할 수 없습니다."
        )
    blocks = np.array_split(unique, n_splits + 1)
    values = dates.to_numpy()
    for k in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:k])
        valid_dates = blocks[k]
        train_idx = np.flatnonzero(np.isin(values, train_dates))
        valid_idx = np.flatnonzero(np.isin(values, valid_dates))
        yield train_idx, valid_idx
