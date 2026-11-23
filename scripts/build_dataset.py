"""수집해 둔 원본(data/raw) → 학습에 쓸 한 장의 표(data/processed).

``scripts/collect_molit.py`` 와 ``scripts/collect_rates.py`` 로 받아 둔 원본이 있어야 한다.
외부 네트워크를 쓰지 않으므로 API 키는 필요 없다.

실행
----
    python scripts/build_dataset.py
    python scripts/build_dataset.py --label          # 이상거래 레이블까지 생성
    python scripts/build_dataset.py --keep-cancelled # 해제 거래를 남긴다 (검증용)

만드는 것
---------
1. ``data/processed/transactions_<코드>_<자치구>.csv`` — 자치구별 정규화 결과
2. ``data/processed/training_dataset.csv``            — 위를 모두 합쳐 외부 데이터까지 붙인 표

주의: ``--label`` 로 만드는 이상거래 레이블은 우리가 정한 기준이고 정답표가 아니다.
레이블 정의는 머신러닝 담당자와 합의해 확정한다 (docs/CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.config import get_settings
from apartment_ml.data.collect import (
    SEOUL_LAWD_CODES,
    load_regulation_events,
    normalize_molit_api_frame,
    read_rate_series,
    to_national_regulation_state,
    to_regulation_history,
)
from apartment_ml.data.merge import (
    add_regulation_score,
    attach_base_rate,
    attach_base_rate_changes,
    attach_national_regulation,
    attach_rate_series,
    attach_regulation,
)
from apartment_ml.data.preprocess import (
    ANOMALY_DIRECTIONS,
    add_deal_features,
    add_peer_price,
    add_price_features,
    label_anomalies,
    preprocess_transactions,
)

RATE_DIR_NAME = "금리"
BASE_RATE_FILE = "base_rate.csv"
DISTRICT_CSV_PREFIX = "transactions_"
OUTPUT_NAME = "training_dataset.csv"


def district_folders(raw_dir: Path) -> list[Path]:
    """``data/raw`` 아래 자치구 폴더 목록. 금리·규제 폴더는 제외한다."""
    codes = set(SEOUL_LAWD_CODES)
    found = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir() and path.name.split("_")[0] in codes:
            found.append(path)
    return found


def read_district(folder: Path, *, drop_cancelled: bool) -> pd.DataFrame:
    """자치구 폴더의 월별 JSON 을 모두 읽어 정규화한다.

    행마다 들어 있는 필드가 다를 수 있다 (해제된 거래에만 cdealType 이 붙는 식).
    DataFrame 으로 만들면 없는 칸은 빈 값으로 채워진다.
    """
    items: list[dict[str, str]] = []
    files = sorted(folder.glob("*.json"))
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.extend(payload.get("items", []))
    if not items:
        return pd.DataFrame()
    raw = pd.DataFrame(items)
    return normalize_molit_api_frame(raw, drop_cancelled=drop_cancelled)


def is_fresh(csv_path: Path, folder: Path, *, drop_cancelled: bool) -> bool:
    """이미 만들어 둔 자치구 CSV 를 그대로 쓸 수 있는지 본다.

    원본보다 새것이어야 하고, 해제 거래 처리 방식도 같아야 한다. 방식이 다른 파일을
    재사용하면 --keep-cancelled 를 붙여도 해제 거래가 빠진 표가 나온다.
    해제 거래를 남긴 파일에는 was_cancelled 칸이 있다.
    """
    if not csv_path.exists():
        return False
    newest = max((p.stat().st_mtime for p in folder.glob("*.json")), default=0.0)
    if csv_path.stat().st_mtime < newest:
        return False
    with csv_path.open(encoding="utf-8-sig") as handle:
        header = handle.readline()
    return (S.WAS_CANCELLED in header) is not drop_cancelled


def build_district_csvs(
    raw_dir: Path, out_dir: Path, *, drop_cancelled: bool, rebuild: bool = False
) -> list[Path]:
    """자치구별 원본 JSON 을 정규화해 CSV 로 저장한다.

    이미 만들어 둔 CSV 가 원본보다 새것이면 건너뛴다. 25개 구를 한 번에 읽으면 몇 분이
    걸리므로, 중간에 끊겨도 다시 실행하면 남은 구만 읽는다. ``--rebuild`` 로 전부 다시 만든다.
    """
    folders = district_folders(raw_dir)
    if not folders:
        raise SystemExit(
            f"자치구 폴더가 없습니다. 먼저 scripts/collect_molit.py 를 돌리세요. ({raw_dir})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    total = 0
    print(f"[1] 자치구별 원본 읽기 — 폴더 {len(folders)}개", flush=True)
    for folder in folders:
        path = out_dir / f"{DISTRICT_CSV_PREFIX}{folder.name}.csv"
        if not rebuild and is_fresh(path, folder, drop_cancelled=drop_cancelled):
            written.append(path)
            print(f"    {folder.name:16} 그대로 둠 (원본이 그대로)", flush=True)
            continue

        frame = read_district(folder, drop_cancelled=drop_cancelled)
        if frame.empty:
            print(f"    {folder.name:16} 받은 파일 없음", flush=True)
            continue
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
        total += len(frame)
        months = len(list(folder.glob("*.json")))
        print(
            f"    {folder.name:16} {months:3}개월  {len(frame):7,}건  칸 {len(frame.columns)}",
            flush=True,
        )
    if total:
        print(f"    새로 만든 것 합계 {total:,}건 -> {out_dir}", flush=True)
    return written


def base_rate_changes(raw_dir: Path, rates: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """기준금리 변경일 표. 일별 수집본이 있으면 그것을 쓴다.

    월별 표에서 유도하면 금통위가 월 중간에 열리므로 변경일이 최대 3주쯤 앞당겨진다.
    정확한 날짜가 필요하면 scripts/collect_base_rate.py 를 돌려 두면 된다.
    """
    exact = raw_dir / BASE_RATE_FILE
    if exact.exists():
        table = pd.read_csv(exact, encoding="utf-8-sig")
        return table, f"{BASE_RATE_FILE} (일별 수집, 변경일 정확)"

    monthly = rates.loc[:, ["ym", S.BASE_RATE]].dropna().reset_index(drop=True)
    changed = monthly[S.BASE_RATE].ne(monthly[S.BASE_RATE].shift())
    out = monthly.loc[changed].copy()
    out["effective_date"] = pd.to_datetime(out["ym"], format="%Y%m")
    out = out.loc[:, ["effective_date", S.BASE_RATE]].reset_index(drop=True)
    return out, "월별 표에서 유도 (변경일이 월초로 당겨짐)"


def combine(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, encoding="utf-8-sig") for p in paths]
    return pd.concat(frames, ignore_index=True)


def build(
    *, label: bool, keep_cancelled: bool, rebuild: bool = False, direction: str = "below"
) -> Path:
    settings = get_settings()
    raw_dir = Path(settings.path("raw_dir"))
    out_dir = Path(settings.path("processed_dir"))

    paths = build_district_csvs(
        raw_dir, out_dir, drop_cancelled=not keep_cancelled, rebuild=rebuild
    )
    frame = combine(paths)
    print()
    print(f"[2] 합치기 — {len(frame):,}행")

    result = preprocess_transactions(frame, invalid="drop")
    frame = result.data
    report = result.report
    print(f"[3] 공통 전처리 — {report.input_rows:,} -> {report.output_rows:,}행")
    if report.invalid_rows:
        print(f"    값이 이상해서 버린 행 {report.invalid_rows}")
    print(f"    중복 제거 {report.duplicate_rows:,}건")

    frame = add_price_features(frame)
    frame = add_deal_features(frame)
    filled = {
        name: int(frame[col].notna().sum())
        for name, col in (
            ("직거래", S.IS_DIRECT_DEAL),
            ("중개사위치", S.AGENT_LOCATION),
            ("토지임대부", S.LAND_LEASEHOLD),
        )
        if col in frame.columns
    }
    print(f"[4] 거래 조건 — 값이 있는 건수 {filled}")

    frame = add_peer_price(frame)
    counts = frame[S.PEER_TIER].value_counts().sort_index()
    tiers = ", ".join(f"{int(tier)}단계 {int(n):,}" for tier, n in counts.items())
    covered = float((frame[S.PEER_TIER] > 0).mean() * 100)
    print(f"[5] 기준가 — {tiers}  (기준을 만든 비율 {covered:.1f}%)", flush=True)

    rates = read_rate_series(raw_dir / RATE_DIR_NAME)
    changes, source = base_rate_changes(raw_dir, rates)
    frame = attach_base_rate(frame, changes)
    frame = attach_base_rate_changes(frame, changes)
    frame = attach_rate_series(frame, rates)
    print(f"[6] 금리 {len(S.RATE_COLUMNS)}종 — 기준금리 출처: {source}")
    print(f"    기준금리가 비어 있는 거래 {int(frame[S.BASE_RATE].isna().sum()):,}건")

    events = load_regulation_events()
    frame = attach_regulation(frame, to_regulation_history(events))
    national = to_national_regulation_state(events)
    frame = attach_national_regulation(frame, national)
    frame = add_regulation_score(
        frame, history=to_regulation_history(events), national_state=national
    )
    moved = int((frame[S.REG_SCORE_CHANGE] != 0).sum())
    print(f"[7] 규제 — 이벤트 {len(events)}건, 점수가 전월과 다른 거래 {moved:,}건")

    if label:
        frame = label_anomalies(frame, direction=direction)
        labelled = int(frame[S.ANOMALY_LABEL].notna().sum())
        flagged = int((frame[S.ANOMALY_LABEL] == 1).sum())
        rate = flagged / labelled * 100 if labelled else 0.0
        print(f"[8] 이상거래 레이블 — 대상 {labelled:,}건 중 {flagged:,}건 ({rate:.2f}%)")
        print("    주의: 우리가 정한 기준이다. 정답표가 아니고 팀 합의가 필요하다.")
    else:
        print("[8] 이상거래 레이블 — 건너뜀 (--label 로 생성)")

    drop = list(S.TRAINING_DROP_COLUMNS)
    if not keep_cancelled:
        drop += list(S.CANCELLATION_RAW_COLUMNS)
    present = [c for c in drop if c in frame.columns]
    frame = frame.drop(columns=present)
    print()
    print(f"[9] 빼는 칸 {len(present)}개 — 계약 시점에 없거나 정답을 다시 담은 원본 칸")
    print(f"    {', '.join(present)}")
    if keep_cancelled:
        marked = int(pd.to_numeric(frame[S.WAS_CANCELLED], errors="coerce").fillna(0).sum())
        print(f"    해제 표시 {marked:,}건을 남겼다 (레이블 검증용. 학습 입력 금지)")

    frame = frame.sort_values(S.CONTRACT_DATE).reset_index(drop=True)
    path = out_dir / OUTPUT_NAME
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print()
    print(f"[10] 저장 — {len(frame):,}행 x {len(frame.columns)}칸 -> {path}")
    first = frame[S.CONTRACT_DATE].min()
    last = frame[S.CONTRACT_DATE].max()
    print(f"    기간 {first:%Y-%m-%d} ~ {last:%Y-%m-%d}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 → 학습 데이터")
    parser.add_argument(
        "--label",
        action="store_true",
        help="이상거래 레이블을 함께 만든다 (팀 합의 전에는 기본 꺼짐)",
    )
    parser.add_argument(
        "--direction",
        choices=ANOMALY_DIRECTIONS,
        default="below",
        help="below: 기준가보다 싼 거래만 (기본) / both: 양쪽 다",
    )
    parser.add_argument(
        "--keep-cancelled",
        action="store_true",
        help="해제된 거래를 남긴다 (레이블 검증용. 학습에는 쓰지 않는다)",
    )
    parser.add_argument("--rebuild", action="store_true", help="자치구별 CSV 를 전부 다시 만든다")
    args = parser.parse_args()
    build(
        label=args.label,
        keep_cancelled=args.keep_cancelled,
        rebuild=args.rebuild,
        direction=args.direction,
    )


if __name__ == "__main__":
    main()
