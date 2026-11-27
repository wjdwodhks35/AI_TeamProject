"""승인된 이상거래 레이블로 분류 모델을 학습하고 백엔드용 모델을 저장한다.

레이블은 기준가(``price_ratio``, ``peer_tier``)가 있으면 학습 시점에
``label_anomalies`` 로 다시 만든다. 어떤 기준으로 만든 레이블인지가 결과를 좌우하므로
CSV 에 들어 있던 값에 의존하지 않고 방향(``--direction``)을 명시한다. 기준가가 없는
데이터는 외부에서 승인받은 레이블을 받은 경우로 보고 CSV 값을 그대로 쓴다
(보고서의 ``label_direction`` 이 ``csv`` 가 된다).

출력에는 **학습이 전혀 없는 규칙 기준선**을 함께 적는다: "직거래면 이상". 모델이
이걸 못 넘기면 모델을 쓸 이유가 없다. 회귀의 '자치구+평형 중앙값' 기준선과 같은
역할이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from apartment_ml import schemas as S
from apartment_ml.config import Settings, get_settings
from apartment_ml.data.preprocess import (
    add_date_features,
    add_price_features,
    coerce_types,
    ensure_optional_columns,
    label_anomalies,
    load_transactions,
    validate_values,
)
from apartment_ml.evaluation.metrics import classification_metrics
from apartment_ml.evaluation.split import time_based_split
from apartment_ml.models.base import LabelsNotAvailableError, TrainingDisabledError
from apartment_ml.models.classification import AnomalyClassificationModel, validate_anomaly_labels
from apartment_ml.models.ensemble import DEFAULT_MEMBERS, build_voting_model

# 기존 API 요청 또는 요청에서 파생되는 값으로 학습 입력을 제한한다.
API_NUMERIC_FEATURES = (
    S.AREA_M2,
    S.FLOOR,
    S.BUILT_YEAR,
    S.BUILDING_AGE,
    S.CONTRACT_YEAR,
    S.CONTRACT_MONTH,
    S.BASE_RATE,
    *S.REGULATION_COLUMNS,
    S.PRICE_KRW,
    S.PRICE_PER_M2,
)


def direct_deal_baseline(part: pd.DataFrame) -> dict:
    """학습 없는 규칙 기준선: 직거래면 이상으로 본다.

    레이블(기준가 대비 싼 거래)에는 직거래가 몰려 있다. 모델이 이 한 줄 규칙을
    못 넘기면, 학습한 것이 '직거래 여부' 말고 무엇인지 설명할 수 있어야 한다.
    """
    target = part[S.CLASSIFICATION_TARGET].astype(int).to_numpy()
    direct = pd.to_numeric(part[S.IS_DIRECT_DEAL], errors="coerce").fillna(0)
    return classification_metrics(target, (direct == 1).astype(int).to_numpy())


def train_anomaly(
    data_path: Path, label_source: str, settings: Settings, direction: str = "below"
) -> dict:
    """공통 시간순 분할에서 모델 네 개를 비교하고 최종 모델과 지표를 저장한다."""
    data = load_transactions(data_path)
    if data_path.stem.upper().endswith("_TEST_ONLY") or (
        S.SOURCE in data and data[S.SOURCE].eq("synthetic_sample").any()
    ):
        raise LabelsNotAvailableError("TEST_ONLY 합성 데이터는 이상거래 학습에 사용할 수 없습니다.")
    target = S.CLASSIFICATION_TARGET
    # 기준가가 있으면 여기서 기준을 정해 레이블을 다시 만든다. 어떤 기준으로 만든
    # 레이블인지가 결과를 좌우하므로 CSV 에 들어 있던 값에 의존하지 않는다.
    # 기준가가 없으면 외부에서 승인받은 레이블을 받은 경우이므로 그대로 쓴다.
    if {S.PRICE_RATIO, S.PEER_TIER} <= set(data.columns):
        data = label_anomalies(data, direction=direction)
    else:
        direction = "csv"
    if target not in data or data[target].isna().all():
        raise LabelsNotAvailableError("anomaly_label 컬럼이 없거나 전부 결측입니다.")
    if not settings.classification_training_enabled:
        raise TrainingDisabledError(
            "레이블 기준 확정 후 classification.training_enabled를 true로 설정하세요."
        )
    if not label_source.strip():
        raise ValueError("--label-source에 팀이 승인한 레이블 출처와 기준을 입력하세요.")

    # 일반 전처리는 레이블을 정수로 반올림하므로 원본 값을 먼저 검증한다.
    data[target] = pd.to_numeric(data[target], errors="raise")
    if not data[target].dropna().isin(S.ANOMALY_LABEL_VALUES).all():
        raise ValueError("anomaly_label에는 0, 1 또는 결측만 사용할 수 있습니다.")
    data = validate_values(coerce_types(data))
    data = add_price_features(add_date_features(ensure_optional_columns(data)))
    split = time_based_split(
        data,
        train_ratio=settings.split.train_ratio,
        valid_ratio=settings.split.valid_ratio,
        test_ratio=settings.split.test_ratio,
        min_rows=settings.split.min_rows_per_split,
    )
    labelled = {name: part.dropna(subset=[target]) for name, part in split.parts().items()}
    for name in ("train", "valid"):
        validate_split_labels(name, labelled[name][target])

    seed = settings.random_seed
    models = {
        name: AnomalyClassificationModel(
            name, numeric_features=API_NUMERIC_FEATURES, random_state=seed
        )
        for name in DEFAULT_MEMBERS
    }
    models["voting"] = build_voting_model(numeric_features=API_NUMERIC_FEATURES, random_state=seed)
    for model in models.values():
        model.fit(labelled["train"], training_enabled=settings.classification_training_enabled)

    valid = {name: model.evaluate(labelled["valid"]) for name, model in models.items()}
    selected = max(
        valid, key=lambda name: (valid[name]["f1"], valid[name]["recall"], valid[name]["precision"])
    )
    validate_split_labels("test", labelled["test"][target])
    test = {name: model.evaluate(labelled["test"]) for name, model in models.items()}

    model = models[selected]
    model.metadata["label_source"] = label_source.strip()
    model.metadata["label_direction"] = direction
    model_path = settings.anomaly_model_path
    report_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    if data_path.resolve() in {model_path.resolve(), report_path.resolve()}:
        raise ValueError("학습 데이터와 모델·평가 결과는 서로 다른 파일이어야 합니다.")
    report = {
        "data_path": str(data_path),
        "label_source": label_source.strip(),
        "label_direction": direction,
        # 학습 없는 규칙. 모델이 이걸 못 넘기면 모델을 쓸 이유가 없다.
        "baseline_direct_deal": direct_deal_baseline(labelled["test"]),
        "splits": {
            name: {
                "start": part[S.CONTRACT_DATE].min().date().isoformat(),
                "end": part[S.CONTRACT_DATE].max().date().isoformat(),
                "rows": len(part),
                "unlabelled": int(part[target].isna().sum()),
                "normal": int(part[target].eq(0).sum()),
                "suspicious": int(part[target].eq(1).sum()),
            }
            for name, part in split.parts().items()
        },
        "threshold": model.threshold,
        "selected_model": selected,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "validation_metrics": valid,
        "test_metrics": test,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_path.parent) as directory:
        pending_model = model.save(Path(directory) / model_path.name)
        pending_report = Path(directory) / report_path.name
        pending_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pending_report.replace(report_path)
        pending_model.replace(model_path)
    return report


def validate_split_labels(name: str, labels: pd.Series) -> None:
    try:
        validate_anomaly_labels(labels)
    except LabelsNotAvailableError as exc:
        raise LabelsNotAvailableError(f"{name} 구간의 레이블이 부족합니다. {exc}") from exc


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=settings.path("processed_dir") / "training_dataset.csv"
    )
    parser.add_argument("--label-source", required=True, help="팀이 승인한 레이블 출처와 판정 기준")
    parser.add_argument(
        "--direction",
        choices=("below", "both"),
        default="below",
        help="below: 기준가보다 싼 거래만 이상 (기본). both: 양쪽 모두",
    )
    args = parser.parse_args()
    try:
        report = train_anomaly(args.data, args.label_source, settings, args.direction)
    except (OSError, ValueError, LabelsNotAvailableError, TrainingDisabledError) as exc:
        parser.exit(1, f"이상거래 학습 실패. {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
