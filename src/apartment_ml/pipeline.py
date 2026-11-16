"""최소 실행 경로 (Smoke pipeline).

외부 API·학습 모델 없이 샘플 데이터로 공통 기반을 점검한다:
로딩 → 스키마 검증/전처리 → 기준금리·규제 병합 → 계약일 기준 시간순 분할 → 요약 출력.

실행: ``python -m apartment_ml.pipeline``
⚠️ 샘플 데이터는 합성(가짜) 데이터이므로 이 결과로 모델 성능을 논하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from apartment_ml.config import Settings, get_settings
from apartment_ml.data.merge import attach_base_rate, attach_regulation
from apartment_ml.data.preprocess import (
    PreprocessReport,
    load_transactions,
    preprocess_transactions,
)
from apartment_ml.evaluation.split import TimeSplit, time_based_split


@dataclass
class SmokeResult:
    data: pd.DataFrame
    report: PreprocessReport
    split: TimeSplit


def run_smoke_pipeline(settings: Settings | None = None) -> SmokeResult:
    settings = settings or get_settings()
    raw = load_transactions(settings.path("sample_transactions"))
    result = preprocess_transactions(raw, invalid="drop")
    data = attach_base_rate(result.data, pd.read_csv(settings.path("sample_base_rates")))
    data = attach_regulation(data, pd.read_csv(settings.path("sample_regulations")))
    split = time_based_split(
        data,
        train_ratio=settings.split.train_ratio,
        valid_ratio=settings.split.valid_ratio,
        test_ratio=settings.split.test_ratio,
        min_rows=settings.split.min_rows_per_split,
    )
    return SmokeResult(data=data, report=result.report, split=split)


def main() -> None:
    res = run_smoke_pipeline()
    print("[전처리 리포트]")
    for k, v in res.report.as_dict().items():
        print(f"  {k}: {v}")
    print("\n[시간순 분할 요약]")
    print(res.split.summary().to_string())
    print("\n※ 합성 샘플 데이터 기반 점검 결과입니다. 실제 거래/모델 성능과 무관합니다.")


if __name__ == "__main__":
    main()
