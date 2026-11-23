"""한국은행 ECOS 금리 10종 수집.

``data/raw/금리/<이름>_<통계표>_<항목>.json`` 에 월별 원본을 그대로 저장한다.
이미 받은 항목은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 받는다.

기준금리 하나만으로는 부족하다. 가계가 실제로 부담하는 대출금리는 기준금리와 다르게
움직이고, 특히 대출금리는 6개월 전의 변화가 지금 가격과 더 강하게 맞물린다
(docs/DATA_QUALITY.md 5절).

실행
----
    export ECOS_API_KEY=<한국은행 ECOS 인증키>
    python scripts/collect_rates.py
    python scripts/collect_rates.py --end 202612

기준금리는 ``scripts/collect_base_rate.py`` 로 **일별** 로도 받아 둘 수 있다. 금통위는 월
중간에 열리므로, 일별 자료가 있으면 변경 적용일이 정확해진다.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from apartment_ml import schemas as S
from apartment_ml.config import get_api_key, get_settings
from apartment_ml.data.collect import CollectError

ECOS_API = "https://ecos.bok.or.kr/api"
CYCLE = "M"
DEFAULT_START = "202101"
DEFAULT_END = "202610"
RATE_DIR_NAME = "금리"
MAX_ROWS = 10_000
RETRIES = 3

# (통계표코드, 항목코드, 저장할 이름 — schemas.RATE_SERIES 의 한글 이름과 같아야 한다)
TARGETS: tuple[tuple[str, str, str], ...] = (
    ("722Y001", "0101000", "기준금리"),
    ("121Y006", "BECBLA0302", "주담대"),
    ("121Y006", "BECBLA030201", "주담대_고정"),
    ("121Y006", "BECBLA030202", "주담대_변동"),
    ("121Y006", "BECBLA03052", "집단대출"),
    ("121Y006", "BECBLA03041", "전세자금대출"),
    ("121Y006", "BECBLA03", "가계대출"),
    ("721Y001", "5020000", "국고채3년"),
    ("721Y001", "5050000", "국고채10년"),
    ("721Y001", "2010000", "CD91일"),
)


def text_of(payload: dict[str, Any], key: str) -> str:
    """ECOS 는 일부 필드를 null 로 보낸다. ``get(key, "")`` 만으로는 None 이 샌다."""
    value = payload.get(key)
    return "" if value is None else str(value)


def request_series(
    api_key: str, stat: str, item: str, start: str, end: str
) -> list[dict[str, Any]]:
    url = f"{ECOS_API}/StatisticSearch/{api_key}/json/kr/1/{MAX_ROWS}/{stat}/{CYCLE}/{start}/{end}/{item}"
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as error:  # noqa: BLE001 - 네트워크·JSON 오류를 함께 재시도한다
            last = error
            if attempt == RETRIES - 1:
                raise CollectError(f"ECOS 요청 실패: {error}") from error
            time.sleep(1.5 * (attempt + 1))
    else:  # pragma: no cover - 위 break/raise 로 끝난다
        raise CollectError(f"ECOS 요청 실패: {last}")

    if "RESULT" in data:  # 인증키 오류 등
        code = text_of(data["RESULT"], "CODE")
        message = text_of(data["RESULT"], "MESSAGE")
        raise CollectError(f"ECOS 오류 {code}: {message}")
    if "StatisticSearch" not in data:
        raise CollectError(f"예상과 다른 응답: {str(data)[:200]}")
    return data["StatisticSearch"].get("row", [])


def strip_footnote(name: str) -> str:
    """항목 이름 끝에 '1)' 같은 각주가 붙어 온다."""
    return re.sub(r"\s*\d+\)\s*$", "", name).strip()


def collect(*, start: str, end: str) -> Path:
    api_key = get_api_key("ECOS_API_KEY")
    settings = get_settings()
    out_dir = Path(settings.path("raw_dir")) / RATE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    known = {name for _, name in S.RATE_SERIES}
    unknown = [name for _, _, name in TARGETS if name not in known]
    if unknown:
        raise CollectError(f"schemas.RATE_SERIES 에 없는 이름입니다: {unknown}")

    calls = 0
    for stat, item, name in TARGETS:
        path = out_dir / f"{name}_{stat}_{item}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            print(f"  {name:14} 이미 있음  {saved['count']}개월", flush=True)
            continue

        rows = request_series(api_key, stat, item, start, end)
        calls += 1
        if not rows:
            print(f"  {name:14} 데이터 없음 — 통계표·항목 코드와 기간을 확인하세요", flush=True)
            continue

        values = [{"ym": text_of(row, "TIME"), "value": text_of(row, "DATA_VALUE")} for row in rows]
        payload = {
            "stat_code": stat,
            "item_code": item,
            "name": name,
            "item_name": strip_footnote(text_of(rows[0], "ITEM_NAME1")) or name,
            "unit": text_of(rows[0], "UNIT_NAME"),
            "cycle": CYCLE,
            "start": start,
            "end": end,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": len(values),
            "rows": values,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"  {name:14} {len(values):4}개월  {values[0]['ym']}~{values[-1]['ym']}"
            f"  {payload['unit']:5} {payload['item_name'][:24]}",
            flush=True,
        )
        time.sleep(0.2)

    print()
    print(f"호출 {calls}회. 저장 위치 {out_dir}")
    print("다음: python scripts/build_dataset.py")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="한국은행 ECOS 금리 10종 수집")
    parser.add_argument("--start", default=DEFAULT_START, help="시작 연월 (YYYYMM)")
    parser.add_argument("--end", default=DEFAULT_END, help="끝 연월 (YYYYMM)")
    args = parser.parse_args()
    collect(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
