"""국토교통부 아파트 매매 실거래가 수집.

서울 25개 자치구를 월 단위로 받아 ``data/raw/<자치구>/<YYYYMM>.json`` 에 원본 그대로 저장한다.
이미 받은 달은 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 받는다.
기간을 늘릴 때도 ``--end`` 만 바꿔서 다시 실행하면 없는 달만 받는다.

실행
----
    export MOLIT_API_KEY=<공공데이터포털 디코딩 키>
    python scripts/collect_molit.py
    python scripts/collect_molit.py --start 202609 --end 202612

개발계정 기준 하루 10,000회이고, 2021-01 ~ 2026-08 전체가 약 1,700회다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

from apartment_ml.config import get_api_key, get_settings
from apartment_ml.data.collect import (
    SEOUL_LAWD_CODES,
    CollectError,
    MolitApiCollector,
    month_range,
    raw_month_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="국토교통부 실거래가 수집")
    parser.add_argument("--start", default=settings.analysis_start.strftime("%Y%m"))
    parser.add_argument("--end", default=settings.analysis_end.strftime("%Y%m"))
    parser.add_argument("--sleep", type=float, default=0.15, help="요청 사이 대기 시간(초)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    raw_dir = settings.path("raw_dir")
    months = month_range(args.start, args.end)

    todo = [
        (code, ym)
        for code in SEOUL_LAWD_CODES
        for ym in months
        if not raw_month_path(raw_dir, code, ym).exists()
    ]
    total_slots = len(SEOUL_LAWD_CODES) * len(months)
    print(f"{args.start} ~ {args.end}   {len(months)}개월 x {len(SEOUL_LAWD_CODES)}개 구")
    print(f"이미 받음 {total_slots - len(todo)}   받을 것 {len(todo)}")
    print(f"저장 위치 {raw_dir}")
    if not todo:
        print("전부 받아져 있습니다.")
        return 0

    collector = MolitApiCollector(get_api_key("MOLIT_API_KEY"), sleep_seconds=args.sleep)
    started = time.time()
    rows_total = 0

    for done, (code, year_month) in enumerate(todo, 1):
        name = SEOUL_LAWD_CODES[code]
        try:
            frame = collector.fetch(code, year_month)
        except CollectError as exc:
            print(f"\n중단: {name} {year_month} — {exc}", file=sys.stderr)
            print("고친 뒤 다시 실행하면 없는 달만 받습니다.", file=sys.stderr)
            return 1

        path = raw_month_path(raw_dir, code, year_month)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lawd_cd": code,
            "district": name,
            "deal_ymd": year_month,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(frame),
            "items": frame.to_dict(orient="records"),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        rows_total += len(frame)
        print(f"[{done}/{len(todo)}] {name:5} {year_month}  {len(frame):5}건")
        time.sleep(args.sleep)

    print(f"\n완료. {rows_total:,}건  {time.time() - started:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
