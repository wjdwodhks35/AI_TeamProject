"""한국은행 기준금리 수집.

일별 자료를 받아 **값이 바뀐 날만** 추려 ``data/raw/base_rate.csv`` 에 저장한다.
금리 변경일을 그대로 쓰기 때문에 ``merge.attach_base_rate`` 가 계약일 시점의
금리를 정확히 붙일 수 있다. 월별로 받으면 적용일이 한 달 단위로 뭉개진다.

실행
----
    export ECOS_API_KEY=<한국은행 ECOS 인증키>
    python scripts/collect_base_rate.py
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from apartment_ml.config import get_api_key, get_settings
from apartment_ml.data.collect import EcosBaseRateCollector

OUTPUT_NAME = "base_rate.csv"


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="한국은행 기준금리 수집")
    parser.add_argument("--start", type=_as_date, default=settings.analysis_start)
    parser.add_argument("--end", type=_as_date, default=settings.analysis_end)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    # 계약일 시점 금리를 붙이려면 분석 시작 이전의 마지막 변경값도 필요하다.
    start = date(args.start.year - 2, args.start.month, 1)
    collector = EcosBaseRateCollector(get_api_key("ECOS_API_KEY"))
    changes = collector.fetch(start, args.end)

    raw_dir = settings.path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / OUTPUT_NAME
    changes.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"기준금리 변경 {len(changes)}회  {start} ~ {args.end}")
    for _, row in changes.iterrows():
        print(f"  {row['effective_date'].date()}  {row['base_rate']:.2f}%")
    print(f"\n저장 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
