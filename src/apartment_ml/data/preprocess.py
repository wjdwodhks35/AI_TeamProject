"""공통 전처리.

설계 원칙 (데이터 누출 방지)
----------------------------
1. 이 모듈의 ``preprocess_transactions`` 단계는 **행 단위(stateless)** 변환만 수행한다.
   (타입 변환, 유효성 검사, 중복 제거, 날짜 파생변수) → 분할 전에 적용해도 누출이 없다.
2. 데이터로부터 통계를 학습하는 단계(결측 대체, 스케일링, 범주 인코딩)는
   ``build_feature_preprocessor`` 가 만드는 scikit-learn ``ColumnTransformer`` 로만 수행하고,
   반드시 **학습(train) 데이터에만 fit** 한다. (모델 Pipeline 내부에 포함 — models/regression.py)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apartment_ml import schemas as S

InvalidPolicy = Literal["raise", "drop"]


class SchemaValidationError(ValueError):
    """데이터가 공통 스키마를 위반할 때."""


@dataclass
class PreprocessReport:
    input_rows: int = 0
    output_rows: int = 0
    invalid_rows: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    missing_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "invalid_rows": dict(self.invalid_rows),
            "duplicate_rows": self.duplicate_rows,
            "missing_counts": dict(self.missing_counts),
        }


@dataclass
class PreprocessResult:
    data: pd.DataFrame
    report: PreprocessReport


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------


def load_transactions(path: str | Path) -> pd.DataFrame:
    """CSV 또는 Parquet 파일을 읽는다 (타입 변환은 하지 않음)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        # 모든 값을 문자열로 읽은 뒤 coerce_types 에서 명시적으로 변환한다.
        return pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8")
    raise ValueError(f"지원하지 않는 파일 형식입니다 (csv/parquet): {path}")


# ---------------------------------------------------------------------------
# 검증 / 타입 변환
# ---------------------------------------------------------------------------


def validate_required_columns(
    df: pd.DataFrame, required: Iterable[str] = S.REQUIRED_COLUMNS
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"필수 컬럼이 없습니다: {missing}")


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        series = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(series, errors="coerce")


def manwon_to_krw(series: pd.Series) -> pd.Series:
    """'125,000'(만원) → 1_250_000_000(원). 변환 불가 값은 NA."""
    return (_to_numeric(series) * S.MANWON_TO_KRW).round().astype("Int64")


def normalize_dates(series: pd.Series) -> pd.Series:
    """여러 날짜 표기(YYYY-MM-DD, YYYYMMDD, YYYY.MM.DD)를 시간정보 없는 datetime64[ns] 로."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"[./]", "-", regex=True)
    compact = s.str.fullmatch(r"\d{8}").fillna(False)
    s = s.where(~compact, s.str.slice(0, 4) + "-" + s.str.slice(4, 6) + "-" + s.str.slice(6, 8))
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return parsed.dt.normalize().astype("datetime64[ns]")


def coerce_types(df: pd.DataFrame, invalid: InvalidPolicy = "raise") -> pd.DataFrame:
    """표준 자료형으로 변환한다.

    - 필수 컬럼에서 변환 실패(잘못된 날짜/숫자)가 있으면 ``invalid='raise'`` 시 오류,
      ``'drop'`` 시 해당 행 제거.
    """
    validate_required_columns(df)
    out = df.copy()

    raw_dates = out[S.CONTRACT_DATE]
    out[S.CONTRACT_DATE] = normalize_dates(raw_dates)
    out[S.DISTRICT] = out[S.DISTRICT].astype("string").str.strip()
    out[S.AREA_M2] = _to_numeric(out[S.AREA_M2]).astype("float64")
    out[S.FLOOR] = _to_numeric(out[S.FLOOR]).round().astype("Int64")
    out[S.PRICE_KRW] = _to_numeric(out[S.PRICE_KRW]).round().astype("Int64")

    for col in (S.BUILT_YEAR, S.ANOMALY_LABEL, *S.REGULATION_COLUMNS):
        if col in out.columns:
            out[col] = _to_numeric(out[col]).round().astype("Int64")
    if S.BASE_RATE in out.columns:
        out[S.BASE_RATE] = _to_numeric(out[S.BASE_RATE]).astype("float64")

    bad = pd.Series(False, index=out.index)
    for col in S.REQUIRED_COLUMNS:
        bad |= out[col].isna()
    if bad.any():
        if invalid == "raise":
            examples = df.loc[bad, list(S.REQUIRED_COLUMNS)].head(3).to_dict("records")
            raise SchemaValidationError(
                f"필수 컬럼 값이 비었거나 변환할 수 없는 행 {int(bad.sum())}건: {examples}"
            )
        out = out.loc[~bad]
    return out


def find_invalid_rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    """규칙별 위반 행 마스크. ``coerce_types`` 이후 호출해야 한다."""
    rules: dict[str, pd.Series] = {
        "unknown_district": ~df[S.DISTRICT].isin(S.SEOUL_DISTRICTS),
        "non_positive_price": df[S.PRICE_KRW] < S.MIN_PRICE_KRW,
        "price_too_large": df[S.PRICE_KRW] > S.MAX_PRICE_KRW,
        "area_out_of_range": ~df[S.AREA_M2].between(S.MIN_AREA_M2, S.MAX_AREA_M2),
        "floor_out_of_range": ~df[S.FLOOR].between(S.MIN_FLOOR, S.MAX_FLOOR),
    }
    if S.BUILT_YEAR in df.columns:
        by = df[S.BUILT_YEAR]
        rules["built_year_out_of_range"] = by.notna() & ~by.between(
            S.MIN_BUILT_YEAR, S.MAX_BUILT_YEAR
        )
    if S.BASE_RATE in df.columns:
        br = df[S.BASE_RATE]
        rules["base_rate_out_of_range"] = br.notna() & ~br.between(S.MIN_BASE_RATE, S.MAX_BASE_RATE)
    if S.ANOMALY_LABEL in df.columns:
        lab = df[S.ANOMALY_LABEL]
        rules["invalid_anomaly_label"] = lab.notna() & ~lab.isin(S.ANOMALY_LABEL_VALUES)
    for col in S.REGULATION_COLUMNS:
        if col in df.columns:
            rules[f"invalid_{col}"] = df[col].notna() & ~df[col].isin((0, 1))
    return {k: v.fillna(True).astype(bool) for k, v in rules.items()}


def validate_values(
    df: pd.DataFrame, invalid: InvalidPolicy = "raise", report: PreprocessReport | None = None
) -> pd.DataFrame:
    rules = find_invalid_rows(df)
    any_bad = pd.Series(False, index=df.index)
    for name, mask in rules.items():
        if mask.any():
            if report is not None:
                report.invalid_rows[name] = int(mask.sum())
            any_bad |= mask
    if any_bad.any():
        if invalid == "raise":
            summary = {k: int(v.sum()) for k, v in rules.items() if v.any()}
            raise SchemaValidationError(f"유효성 검사 실패: {summary}")
        return df.loc[~any_bad].copy()
    return df


# ---------------------------------------------------------------------------
# 중복 / 결측 / 파생변수
# ---------------------------------------------------------------------------


def duplicate_key_columns(df: pd.DataFrame) -> list[str]:
    return [*S.DUPLICATE_KEY, *(c for c in S.DUPLICATE_KEY_OPTIONAL if c in df.columns)]


def drop_duplicate_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """중복 거래 제거 (첫 행 유지). 반환: (결과, 제거 건수)."""
    key = duplicate_key_columns(df)
    dup = df.duplicated(subset=key, keep="first")
    return df.loc[~dup].copy(), int(dup.sum())


def missing_value_report(df: pd.DataFrame, columns: Iterable[str] | None = None) -> dict[str, int]:
    cols = list(columns) if columns is not None else list(df.columns)
    return {c: int(df[c].isna().sum()) for c in cols if c in df.columns}


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """계약연도·월, 건물연식 생성 (행 단위 계산, 누출 없음)."""
    out = df.copy()
    dates = pd.to_datetime(out[S.CONTRACT_DATE])
    out[S.CONTRACT_YEAR] = dates.dt.year.astype("Int64")
    out[S.CONTRACT_MONTH] = dates.dt.month.astype("Int64")
    if S.BUILT_YEAR in out.columns:
        out[S.BUILDING_AGE] = (out[S.CONTRACT_YEAR] - out[S.BUILT_YEAR]).astype("Int64")
    else:
        out[S.BUILDING_AGE] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """가격 파생변수(㎡당 가격). **분류 전용** — 회귀 입력 금지 (schemas 참고)."""
    out = df.copy()
    out[S.PRICE_PER_M2] = out[S.PRICE_KRW].astype("float64") / out[S.AREA_M2]
    return out


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """선택 컬럼이 없으면 null 로 채워 스키마 모양을 통일한다."""
    out = df.copy()
    for col in (S.BUILT_YEAR, S.ANOMALY_LABEL, *S.REGULATION_COLUMNS):
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if S.BASE_RATE not in out.columns:
        out[S.BASE_RATE] = np.nan
    return out


def preprocess_transactions(df: pd.DataFrame, invalid: InvalidPolicy = "raise") -> PreprocessResult:
    """행 단위 공통 전처리 전체 실행.

    순서: 필수 컬럼 검증 → 타입 변환 → 값 검증 → 중복 제거 → 선택 컬럼 보완 → 날짜 파생변수.
    결과는 계약일 오름차순으로 정렬된다.
    """
    report = PreprocessReport(input_rows=len(df))
    out = coerce_types(df, invalid=invalid)
    if invalid == "drop" and len(out) < len(df):
        report.invalid_rows["unparseable_required"] = len(df) - len(out)
    out = validate_values(out, invalid=invalid, report=report)
    out, report.duplicate_rows = drop_duplicate_transactions(out)
    out = ensure_optional_columns(out)
    out = add_date_features(out)
    out = out.sort_values(S.CONTRACT_DATE, kind="mergesort").reset_index(drop=True)
    report.output_rows = len(out)
    report.missing_counts = missing_value_report(out, [*S.REQUIRED_COLUMNS, *S.OPTIONAL_COLUMNS])
    return PreprocessResult(data=out, report=report)


# ---------------------------------------------------------------------------
# 학습형 전처리 (train 에만 fit)
# ---------------------------------------------------------------------------


def assert_no_forbidden_features(features: Sequence[str], forbidden: Iterable[str]) -> None:
    leaked = sorted(set(features) & set(forbidden))
    if leaked:
        raise ValueError(f"입력 변수에 사용할 수 없는 컬럼(정답/정답 파생)이 포함됨: {leaked}")


def to_model_frame(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """모델 입력용 DataFrame: 지정 컬럼만, 숫자는 float64(NA→NaN), 범주는 object."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"모델 입력 컬럼이 없습니다: {missing}")
    out = pd.DataFrame(index=df.index)
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("float64")
        else:
            out[col] = s.astype(object).where(s.notna(), np.nan)
    return out


def build_feature_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """결측 대체 + 스케일링 + 원-핫 인코딩. 반드시 Pipeline 안에서 train 데이터로만 fit 할 것."""
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), list(numeric_features)),
            ("cat", categorical, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
