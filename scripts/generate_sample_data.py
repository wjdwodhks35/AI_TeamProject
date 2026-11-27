"""합성(가짜) 샘플 데이터 생성기 — 테스트·최소 실행 경로 전용.

⚠️ 생성되는 모든 값은 난수 기반의 가짜 값이며 실제 거래·실제 기준금리·실제 정책과 무관합니다.
   모델 학습 결과나 성능 평가에 사용하지 마세요.

실행: python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from apartment_ml.schemas import SEOUL_DISTRICTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
SEED = 20240101
N_ROWS = 120


def make_transactions(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2021-01-01", "2026-10-31", freq="D")
    dates = np.sort(rng.choice(days, size=N_ROWS, replace=True))
    district = rng.choice(SEOUL_DISTRICTS, size=N_ROWS)
    area = rng.choice([39.6, 59.9, 74.5, 84.9, 101.3, 114.7], size=N_ROWS)
    floor = rng.integers(1, 36, size=N_ROWS)
    built = rng.integers(1985, 2024, size=N_ROWS).astype(object)
    built[rng.random(N_ROWS) < 0.05] = None  # 일부 건축연도 결측
    price_manwon = (area * rng.uniform(800, 2500, size=N_ROWS)).round(-2).astype(int)
    df = pd.DataFrame(
        {
            "district": district,
            "contract_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "area_m2": area,
            "floor": floor,
            "built_year": built,
            "price_krw": price_manwon * 10_000,
            "apartment_name": [f"SAMPLE단지{i % 17:02d}" for i in range(N_ROWS)],
            "anomaly_label": None,  # 미확보 — 임의 레이블 생성 금지
            "source": "synthetic_sample",
        }
    )
    # 전처리 검증용 의도적 결함: 중복 1건 + 음수 가격 1건
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    bad = df.iloc[[20]].copy()
    bad["price_krw"] = -1
    bad["contract_date"] = df.iloc[21]["contract_date"]
    return pd.concat([df, bad], ignore_index=True)


def make_base_rates() -> pd.DataFrame:
    # 가짜 금리 경로 (실제 한국은행 기준금리 아님)
    return pd.DataFrame(
        {
            "effective_date": ["2020-06-01", "2021-09-01", "2022-03-01", "2022-09-01",
                               "2023-03-01", "2024-09-01", "2025-06-01"],
            "base_rate": [0.50, 1.00, 1.75, 2.75, 3.50, 3.00, 2.50],
        }
    )  # fmt: skip


def make_regulations() -> pd.DataFrame:
    # 가짜 규제 이력 (실제 정책 아님)
    return pd.DataFrame(
        {
            "district": ["강남구", "서초구", "송파구", "용산구", "마포구"],
            "regulation_type": ["reg_speculation_overheated"] * 3
            + ["reg_land_permit", "reg_adjustment_target"],
            "announced_date": [
                "2020-12-01",
                "2020-12-01",
                "2020-12-01",
                "2023-01-10",
                "2021-01-05",
            ],
            "effective_from": [
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
                "2023-02-01",
                "2021-01-10",
            ],
            "effective_to": [None, None, "2024-06-30", None, "2023-01-05"],
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    make_transactions(rng).to_csv(OUT / "sample_transactions_TEST_ONLY.csv", index=False)
    make_base_rates().to_csv(OUT / "sample_base_rates_TEST_ONLY.csv", index=False)
    make_regulations().to_csv(OUT / "sample_regulations_TEST_ONLY.csv", index=False)
    print(f"합성 샘플 데이터 생성 완료: {OUT}")


if __name__ == "__main__":
    main()
