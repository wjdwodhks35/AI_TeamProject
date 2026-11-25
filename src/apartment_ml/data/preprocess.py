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
    for col in (
        S.BUILT_YEAR,
        S.ANOMALY_LABEL,
        *S.REGULATION_COLUMNS,
        *S.NATIONAL_REGULATION_COLUMNS,
        *S.DEAL_FEATURE_COLUMNS,
    ):
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for col in (*S.RATE_COLUMNS, *S.RATE_CHANGE_COLUMNS, *S.REGULATION_SCORE_COLUMNS):
        if col not in out.columns:
            out[col] = np.nan
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


# ---------------------------------------------------------------------------
# 거래 특성 (국토교통부 상세 자료)
# ---------------------------------------------------------------------------

DIRECT_DEAL_TEXT = "직거래"
BROKER_DEAL_TEXT = "중개거래"
AREA_BAND = "_area_band"  # 기준가 계산용 임시 컬럼 (10㎡ 단위)


def add_deal_features(df: pd.DataFrame) -> pd.DataFrame:
    """상세 자료의 거래유형·중개사 소재지·토지임대부를 숫자 컬럼으로 만든다.

    - ``is_direct_deal``: 직거래 1, 중개거래 0, 표기 없으면 null
      (2021년 자료에는 거래유형이 거의 없다)
    - ``agent_location``: 0 소재지 없음 / 1 같은 자치구 / 2 다른 자치구
      중개사가 다른 구에 있는 거래는 해제율이 더 높았다 (8.1% 대 5.2%)
    - ``land_leasehold``: 토지임대부 1. 땅값이 빠져 같은 평형이라도 훨씬 싸므로
      이 값을 넣지 않으면 전부 '이상하게 싼 거래' 로 잡힌다
    """
    out = df.copy()

    if "deal_type" in out.columns:
        text = out["deal_type"].astype("string").str.strip()
        direct = pd.Series(pd.NA, index=out.index, dtype="Int64")
        direct[text == DIRECT_DEAL_TEXT] = 1
        direct[text == BROKER_DEAL_TEXT] = 0
        out[S.IS_DIRECT_DEAL] = direct

    if "agent_district" in out.columns and S.DISTRICT in out.columns:
        agent = out["agent_district"].astype("string").fillna("").str.strip()
        district = out[S.DISTRICT].astype("string").fillna("")
        same = [bool(d) and bool(a) and d in a for d, a in zip(district, agent, strict=True)]
        location = np.where(agent == "", 0, np.where(same, 1, 2))
        out[S.AGENT_LOCATION] = pd.Series(location, index=out.index, dtype="Int64")

    if "land_leasehold_raw" in out.columns:
        raw = out["land_leasehold_raw"].astype("string").str.strip().str.upper()
        leasehold = pd.Series(pd.NA, index=out.index, dtype="Int64")
        leasehold[raw == "Y"] = 1
        leasehold[raw == "N"] = 0
        out[S.LAND_LEASEHOLD] = leasehold

    return out


# ---------------------------------------------------------------------------
# 기준가 (이상거래 레이블의 재료)
# ---------------------------------------------------------------------------
# 좁은 그룹부터 보고 표본이 모자라면 넓은 그룹으로 올라간다.
# 실제 수집 데이터에서 '같은 단지·같은 평형' 만 쓰면 45.7% 밖에 기준을 못 만들었고,
# 네 단계를 두면 92.4% 가 됐다.

PEER_TIERS: tuple[tuple[int, tuple[str, ...], int], ...] = (
    (1, (S.COMPLEX_ID, AREA_BAND), 3),
    (2, (S.COMPLEX_ID,), 3),
    (3, (S.DONG, AREA_BAND), 5),
    (4, (S.DISTRICT, AREA_BAND), 5),
)
AREA_BAND_WIDTH = 10.0


def _peer_one_tier(
    df: pd.DataFrame, keys: tuple[str, ...], min_count: int, period: str
) -> pd.Series:
    """직전 기간의 같은 그룹 ㎡당 가격 중앙값 x 이 거래의 면적.

    가격 중앙값을 그대로 쓰면 안 된다. 같은 단지라도 59㎡와 115㎡는 가격이 두 배
    차이나므로, 단지 중앙값을 기준으로 삼으면 작은 평형은 전부 싸게, 큰 평형은
    전부 비싸게 잡힌다. ㎡당 가격으로 기준을 만들고 면적을 다시 곱한다.
    """
    work = df.copy()
    work["_period"] = work[S.CONTRACT_DATE].dt.to_period(period)
    work["_per_m2"] = work[S.PRICE_KRW].astype("float64") / work[S.AREA_M2]

    grouped = (
        work.groupby([*keys, "_period"], dropna=True)["_per_m2"]
        .agg(_median="median", _count="size")
        .reset_index()
    )
    grouped = grouped.loc[grouped["_count"] >= min_count].copy()
    grouped["_period"] = grouped["_period"] + 1  # 다음 기간에 붙인다 (미래를 보지 않는다)

    merged = work.merge(grouped[[*keys, "_period", "_median"]], on=[*keys, "_period"], how="left")
    merged.index = df.index
    return merged["_median"] * df[S.AREA_M2]


def add_peer_price(df: pd.DataFrame, *, period: str = "Q") -> pd.DataFrame:
    """직전 기간의 같은 조건 시세를 ``peer_median_price_krw`` 로 붙인다.

    좁은 그룹부터 보고 표본이 모자라면 넓은 그룹으로 올라가며, 몇 단계에서 만들었는지를
    ``peer_tier`` 에 남긴다. 단계가 넓을수록 기준 자체가 흔들리므로 레이블을 달 때
    단계를 봐야 한다 (``label_anomalies`` 참고).

    직전 기간만 보기 때문에 미래 정보가 들어가지 않는다. 학습형 전처리가 아니라
    시점 기준 집계이므로 분할 전에 계산해도 누출이 생기지 않는다.
    """
    required = [S.CONTRACT_DATE, S.AREA_M2, S.PRICE_KRW, S.DISTRICT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"기준가 계산에 필요한 컬럼이 없습니다: {missing}")

    out = df.copy()
    out[AREA_BAND] = (out[S.AREA_M2] // AREA_BAND_WIDTH * AREA_BAND_WIDTH).astype("float64")

    peer = pd.Series(np.nan, index=out.index, dtype="float64")
    tier = pd.Series(0, index=out.index, dtype="Int64")
    for number, keys, min_count in PEER_TIERS:
        if any(k not in out.columns for k in keys):
            continue  # complex_id 나 dong 이 없는 데이터도 받을 수 있게 한다
        candidate = _peer_one_tier(out, keys, min_count, period)
        fill = peer.isna() & candidate.notna()
        peer[fill] = candidate[fill]
        tier[fill] = number

    out[S.PEER_MEDIAN_PRICE_KRW] = peer
    out[S.PEER_TIER] = tier
    out[S.PRICE_RATIO] = out[S.PRICE_KRW].astype("float64") / peer
    return out.drop(columns=[AREA_BAND])


# ---------------------------------------------------------------------------
# 이상거래 레이블 (후보 — ML 담당자와 정의 확정 전)
# ---------------------------------------------------------------------------
# 기준가 단계마다 기준이 흔들리는 정도가 다르다. 실제 수집 데이터에서 시세비율의
# 사분위폭이 1단계 0.077, 2단계 0.226, 3단계 0.313, 4단계 0.454 였다.
# 자치구 단위(4단계)로 판정하면 이상으로 뽑힌 거래의 직거래 비율이 그 구 평균과
# 같아져(배수 1.0) 신호가 아니라 잡음이었다. 그래서 1·2단계만 레이블을 단다.

DEFAULT_ANOMALY_THRESHOLDS: dict[int, float] = {1: 0.20, 2: 0.30}


ANOMALY_DIRECTIONS = ("below", "both")


def label_anomalies(
    df: pd.DataFrame,
    *,
    thresholds: dict[int, float] | None = None,
    direction: str = "both",
) -> pd.DataFrame:
    """기준가에서 많이 벗어난 거래에 ``anomaly_label`` 을 단다.

    주의: 레이블 정의는 머신러닝 담당자와 합의해 확정한다 (docs/CONTRIBUTING.md).
    이 함수는 확정 전 후보이며, 호출하지 않으면 ``anomaly_label`` 은 null 로 남는다.
    ``configs/settings.yaml`` 의 ``classification.training_enabled`` 도 함께 확인한다.

    ``thresholds`` 는 기준가 단계별 허용 범위다. 기본값은 1단계 20%, 2단계 30% 이고,
    여기에 없는 단계는 레이블을 달지 않는다 (null 로 남긴다).

    이 레이블이 뜻하는 것은 **"같은 조건의 주변 거래에서 가격이 크게 벗어났다"** 이지
    "불법이다" 가 아니다. 싸게 팔린 데에도 비싸게 팔린 데에도 정상적인 이유가 있을 수 있다.

    ``direction``
        ``"both"`` (기본) 양쪽으로 벗어난 거래를 모두 본다.
        ``"below"``       기준가보다 **싼** 거래만 본다.

    두 방향은 성격이 다르다. 레이블을 만들 때 쓰지 않은 직거래 비율로 보면
    (전체 2.74% 기준):

        1단계 싸게   1,172건  직거래 67.03%  = 24.4배
        1단계 비싸게 2,121건  직거래  3.88%  =  1.4배
        2단계 싸게   1,527건  직거래 23.42%  =  8.5배
        2단계 비싸게 2,759건  직거래  1.36%  =  0.5배   <- 평균보다 낮다

    싼 쪽에는 직거래가 몰려 있고 비싼 쪽에는 그렇지 않다. 해제율과 단기 재매매로도
    같은 결과가 나왔다 (docs/DATA_QUALITY.md). 비싼 쪽이 왜 갈리지 않는지는 설명하지
    못한다. 한 레이블로 묶으면 성격이 다른 두 집단이 섞이므로, 모델 담당자는
    ``direction="below"`` 로 만든 레이블과 성능을 비교해 보는 것이 좋다.
    """
    if direction not in ANOMALY_DIRECTIONS:
        raise ValueError(f"direction 은 {ANOMALY_DIRECTIONS} 중 하나여야 합니다: {direction!r}")
    if S.PRICE_RATIO not in df.columns or S.PEER_TIER not in df.columns:
        raise SchemaValidationError("먼저 add_peer_price 로 기준가를 만들어야 합니다.")

    limits = DEFAULT_ANOMALY_THRESHOLDS if thresholds is None else thresholds
    out = df.copy()
    label = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for tier, limit in limits.items():
        target = (out[S.PEER_TIER] == tier) & out[S.PRICE_RATIO].notna()
        ratio = out.loc[target, S.PRICE_RATIO]
        hit = ratio < 1 - limit
        if direction == "both":
            hit = hit | (ratio > 1 + limit)
        label[target] = hit.astype("Int64")
    out[S.ANOMALY_LABEL] = label
    return out
