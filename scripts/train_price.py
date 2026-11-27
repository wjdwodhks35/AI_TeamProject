"""학습 데이터(data/processed/training_dataset.csv) → 가격 회귀 모델 학습·비교.

``scripts/build_dataset.py`` 로 만든 표가 있어야 한다. 외부 네트워크는 쓰지 않는다.

실행
----
    python scripts/train_price.py                      # 전체 비교 후 최고 모델 저장
    python scripts/train_price.py --models gradient_boosting
    python scripts/train_price.py --no-market          # 시장지수 없이 (절대가격 학습)
    python scripts/train_price.py --no-location        # 동·단지 인코딩 없이

평가 방식
---------
계약일 기준 시간순 70/15/15 분할. test 구간은 학습 구간보다 **미래**이므로,
무작위 분할보다 점수가 낮게 나오는 게 정상이고 실제 쓰임에 가까운 수치다.

기준선을 함께 출력한다: '자치구 + 평형대(10㎡ 단위) 중앙값' 조회. 학습이 전혀
없는 방식이므로, 모델이 이걸 못 넘기면 모델을 쓸 이유가 없다.

모델 비교 표는 전부 **train 만으로** 학습한 값이다(공정한 비교). 저장하는 배포본은
여기서 두 단계를 더 거친다.

1. 최고 모델의 로그 복원 편향 보정계수를 **valid** 에서 구한다.
2. **train+valid** 로 다시 학습하고 1의 계수를 그대로 적용해 저장한다.

valid 구간(최근 7개월)을 버리면 그 사이 거래된 단지를 못 본 상태로 예측하게 된다.
측정 결과 test MAE 1.34억 -> 1.12억, 편향 -0.41억 -> -0.02억 차이가 났다. 보정계수는
valid 를 학습에 넣기 전 모델에서 구하므로 test 정보는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.config import get_settings
from apartment_ml.evaluation.metrics import regression_metrics
from apartment_ml.evaluation.split import TimeSplit, time_based_split
from apartment_ml.models.complex_directory import ComplexDirectory
from apartment_ml.models.market_index import MarketIndex
from apartment_ml.models.regression import REGRESSOR_FACTORIES, PriceRegressionModel

DEFAULT_DATA = Path("data/processed/training_dataset.csv")
AREA_BAND = 10.0


def load_split(path: Path) -> TimeSplit:
    if not path.exists():
        raise SystemExit(
            f"학습 데이터가 없습니다: {path}\n먼저 scripts/build_dataset.py 를 실행하세요."
        )
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df[S.CONTRACT_DATE] = pd.to_datetime(df[S.CONTRACT_DATE])
    return time_based_split(df)


def baseline_lookup(split: TimeSplit) -> dict[str, np.ndarray]:
    """기준선: 학습 구간의 '자치구 + 평형대' 중앙값을 그대로 쓴다."""
    overall = float(split.train[S.REGRESSION_TARGET].median())

    def band(part: pd.DataFrame) -> pd.Series:
        return (part[S.AREA_M2] // AREA_BAND * AREA_BAND).astype("float64")

    table = split.train.groupby([split.train[S.DISTRICT], band(split.train)])[
        S.REGRESSION_TARGET
    ].median()

    def predict(part: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([part[S.DISTRICT], band(part)])
        return table.reindex(keys).fillna(overall).to_numpy("float64")

    return {"valid": predict(split.valid), "test": predict(split.test)}


def score(split: TimeSplit, pred: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("valid", "test"):
        part = getattr(split, name)
        truth = part[S.REGRESSION_TARGET].to_numpy("float64")
        guess = np.clip(pred[name], 1.0, None)
        for key, value in regression_metrics(truth, guess).items():
            out[f"{name}_{key}"] = value
        out[f"{name}_bias"] = float(np.mean(guess - truth))
    return out


def print_header() -> None:
    print(f"\n{'모델':30}{'valid MAE':>11}{'test MAE':>11}{'R2':>8}{'MAPE':>8}{'편향':>10}")
    print("-" * 78)


def print_row(label: str, m: dict[str, float], *, valid_in_train: bool = False) -> None:
    # valid 를 학습에 넣은 모델의 valid MAE 는 자기 학습 데이터 점수라 의미가 없다.
    valid = "         -" if valid_in_train else f"{m['valid_mae'] / 1e8:10.2f}억"
    print(
        f"{label:30}{valid}{m['test_mae'] / 1e8:10.2f}억"
        f"{m['test_r2']:8.3f}{m['test_mape']:7.1f}%{m['test_bias'] / 1e8:+9.2f}억",
        flush=True,
    )


def monthly_bias(split: TimeSplit, pred: np.ndarray) -> None:
    """test 구간의 월별 편향. 한쪽으로 쏠리면 시점 흐름을 못 따라가는 것이다."""
    frame = pd.DataFrame(
        {
            "ym": split.test[S.CONTRACT_DATE].dt.to_period("M"),
            "err": np.clip(pred, 1.0, None) - split.test[S.REGRESSION_TARGET].to_numpy("float64"),
        }
    )
    print("\ntest 월별 편향 (예측 - 실제)")
    for ym, group in frame.groupby("ym"):
        print(f"  {ym}  n={len(group):6,}  {group['err'].mean() / 1e8:+6.2f}억")


def main() -> None:
    parser = argparse.ArgumentParser(description="가격 회귀 모델 학습·비교")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # 기본 경로는 설정에서 가져온다. API 가 읽는 경로와 어긋나면 모델을 못 찾는다.
    parser.add_argument(
        "--out", type=Path, default=None, help="기본값: settings 의 price_model_path"
    )
    parser.add_argument(
        "--directory-out",
        type=Path,
        default=None,
        help="기본값: settings 의 complex_directory_path",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(REGRESSOR_FACTORIES), choices=list(REGRESSOR_FACTORIES)
    )
    parser.add_argument(
        "--no-market",
        action="store_true",
        help="시장지수 없이 절대가격을 학습한다 (시점 흐름을 못 따라간다 — 비교용)",
    )
    parser.add_argument("--no-location", action="store_true", help="동·단지 인코딩을 쓰지 않는다")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = load_split(args.data)
    for name in ("train", "valid", "test"):
        part = getattr(split, name)
        dates = part[S.CONTRACT_DATE]
        print(f"{name:5} {len(part):8,}건  {dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}")

    print_header()
    print_row("기준선: 자치구+평형 중앙값", score(split, baseline_lookup(split)))

    # 시장지수는 '예측 시점에 이미 신고가 끝난 직전월 거래' 로 만든다. 운영에서도
    # 같은 정보를 쓸 수 있으므로, 전 구간 이력으로 만들어 두고 각 행은 직전월 값만 본다.
    history = pd.concat([split.train, split.valid, split.test])
    market = None if args.no_market else MarketIndex().fit(history)
    location = () if args.no_location else S.REGRESSION_LOCATION_FEATURES

    best: tuple[float, str, PriceRegressionModel] | None = None
    for name in args.models:
        started = time.time()
        model = PriceRegressionModel(
            name,
            location_features=location,
            relative_to_market=market is not None,
            market_index=market,
            random_state=args.seed,
        ).fit(split.train)
        pred = {part: model.predict(getattr(split, part)) for part in ("valid", "test")}
        metrics = score(split, pred)
        print_row(f"{name} ({time.time() - started:.0f}초)", metrics)
        if best is None or metrics["valid_mae"] < best[0]:
            best = (metrics["valid_mae"], name, model)

    if best is None:
        return
    _, name, model = best

    # (1) 로그 복원 편향 보정계수를 valid 에서 구한다
    factor = model.calibrate(split.valid)
    print_row(
        f"{name} + 편향보정 (x{factor:.4f})",
        score(split, {p: model.predict(getattr(split, p)) for p in ("valid", "test")}),
    )

    # (2) valid 까지 넣어 다시 학습한다. 이게 저장되는 배포본이다.
    deployed = PriceRegressionModel(
        name,
        location_features=location,
        relative_to_market=market is not None,
        market_index=market,
        random_state=args.seed,
    ).fit(pd.concat([split.train, split.valid]))
    deployed.set_log_bias_factor(
        factor, residual_sd=model.log_residual_sd_, source=_date_range(split.valid)
    )
    pred_test = deployed.predict(split.test)
    print_row(
        f"{name} + 보정 + train+valid 학습",
        score(split, {"valid": model.predict(split.valid), "test": pred_test}),
        valid_in_train=True,
    )

    monthly_bias(split, pred_test)

    settings = get_settings()
    out = args.out or settings.price_model_path
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"\n저장: {name}, train+valid {deployed.metadata['n_train']:,}건 학습, "
        f"보정계수 {factor:.4f} -> {deployed.save(out)}"
    )

    # 단지 목록: API 가 프론트에 내려줄 선택 목록. 모델이 학습한 구간과 같게 만든다.
    directory = ComplexDirectory.build(pd.concat([split.train, split.valid]))
    directory_out = args.directory_out or settings.complex_directory_path
    print(f"단지 목록: {len(directory):,}개 단지 -> {directory.save(directory_out)}")


def _date_range(part: pd.DataFrame) -> list[str]:
    dates = part[S.CONTRACT_DATE]
    return [dates.min().date().isoformat(), dates.max().date().isoformat()]


if __name__ == "__main__":
    main()
