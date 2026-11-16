"""FastAPI 앱.

실행: ``uvicorn apartment_ml.api.main:app --reload``  →  문서: http://127.0.0.1:8000/docs

모델 준비 규칙
--------------
- 시작 시 settings.yaml 의 ``models.*_model_path`` 파일이 있으면 불러온다. 없으면 '미준비'.
- 모델이 미준비이면 예측 엔드포인트는 **503 MODEL_NOT_READY** 를 반환한다 (가짜 예측값 없음).
- 테스트에서는 ``create_app(price_model=..., load_from_disk=False)`` 로 모델을 주입한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status

from apartment_ml import __version__
from apartment_ml import schemas as S
from apartment_ml.api.schemas import (
    AnomalyPredictionRequest,
    AnomalyPredictionResponse,
    ErrorResponse,
    HealthResponse,
    InterestRateScenarioRequest,
    InterestRateScenarioResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)
from apartment_ml.config import get_settings
from apartment_ml.data.preprocess import add_price_features
from apartment_ml.models.base import AnomalyDetector, ModelNotReadyError, PricePredictor
from apartment_ml.scenarios.interest_rate import (
    build_feature_frame,
    compare_interest_rate_scenario,
    out_of_range_warnings,
)

logger = logging.getLogger(__name__)

NOT_READY_RESPONSES = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "모델 미준비"},
}


def _not_ready(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "MODEL_NOT_READY", "message": message},
    )


def _load_models_from_disk() -> tuple[PricePredictor | None, AnomalyDetector | None]:
    # 지연 import: 모델 의존성이 없어도 /health 는 동작하도록
    from apartment_ml.models.classification import AnomalyClassificationModel
    from apartment_ml.models.regression import PriceRegressionModel

    settings = get_settings()
    price = anomaly = None
    if settings.price_model_path.exists():
        try:
            price = PriceRegressionModel.load(settings.price_model_path)
        except ModelNotReadyError as exc:
            logger.warning("가격 모델 로드 실패: %s", exc)
    if settings.anomaly_model_path.exists():
        try:
            anomaly = AnomalyClassificationModel.load(settings.anomaly_model_path)
        except ModelNotReadyError as exc:
            logger.warning("이상거래 모델 로드 실패: %s", exc)
    return price, anomaly


def create_app(
    price_model: PricePredictor | None = None,
    anomaly_model: AnomalyDetector | None = None,
    load_from_disk: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_from_disk and price_model is None and anomaly_model is None:
            app.state.price_model, app.state.anomaly_model = _load_models_from_disk()
        yield

    app = FastAPI(
        title="서울 아파트 가격 예측 및 이상거래 의심 탐지 API",
        version=__version__,
        description="Phase 0: API 계약. 모델이 준비되지 않으면 503 을 반환합니다.",
        lifespan=lifespan,
    )
    app.state.price_model = price_model
    app.state.anomaly_model = anomaly_model

    def _price_model(request: Request) -> PricePredictor:
        model = request.app.state.price_model
        if model is None or not model.is_fitted:
            raise _not_ready("가격 예측 모델이 아직 학습/배포되지 않았습니다.")
        return model

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        def ready(m: object) -> bool:
            return m is not None and bool(getattr(m, "is_fitted", False))

        return HealthResponse(
            version=__version__,
            models={
                "price": ready(request.app.state.price_model),
                "anomaly": ready(request.app.state.anomaly_model),
            },
        )

    @app.post(
        "/predict/price",
        response_model=PricePredictionResponse,
        responses=NOT_READY_RESPONSES,
        tags=["prediction"],
    )
    def predict_price(body: PricePredictionRequest, request: Request) -> PricePredictionResponse:
        model = _price_model(request)
        frame = build_feature_frame(body)
        pred = float(model.predict(frame)[0])
        warnings = out_of_range_warnings(model, frame)
        if pred <= 0:
            warnings.append(
                "모델 예측값이 0 이하입니다. 입력 조건이 학습 범위를 벗어났을 수 있습니다."
            )
        return PricePredictionResponse(
            predicted_price_krw=round(pred), model_name=model.name, warnings=warnings
        )

    @app.post(
        "/predict/anomaly",
        response_model=AnomalyPredictionResponse,
        responses=NOT_READY_RESPONSES,
        tags=["prediction"],
    )
    def predict_anomaly(
        body: AnomalyPredictionRequest, request: Request
    ) -> AnomalyPredictionResponse:
        model: AnomalyDetector | None = request.app.state.anomaly_model
        if model is None or not model.is_fitted:
            raise _not_ready(
                "이상거래 분류 모델이 준비되지 않았습니다 (이상거래 정답 레이블 미확보)."
            )
        features = S.ApartmentFeatures.model_validate(body.model_dump(exclude={"price_krw"}))
        frame = build_feature_frame(features)
        frame[S.PRICE_KRW] = pd.array([body.price_krw], dtype="Int64")
        frame = add_price_features(frame)
        score = float(model.predict_proba(frame)[0])
        threshold = float(getattr(model, "threshold", 0.5))

        warnings: list[str] = []
        reference = gap = None
        price_model = request.app.state.price_model
        if price_model is not None and price_model.is_fitted:
            reference = round(float(price_model.predict(frame)[0]))
            if reference > 0:
                gap = (body.price_krw - reference) / reference * 100.0
        else:
            warnings.append("가격 모델이 없어 기준가격을 제공하지 않습니다.")
        return AnomalyPredictionResponse(
            is_suspicious=score >= threshold,
            suspicion_score=score,
            threshold=threshold,
            reference_price_krw=reference,
            price_gap_pct=gap,
            model_name=model.name,
            warnings=warnings,
        )

    @app.post(
        "/scenario/interest-rate",
        response_model=InterestRateScenarioResponse,
        responses=NOT_READY_RESPONSES,
        tags=["scenario"],
    )
    def interest_rate_scenario(
        body: InterestRateScenarioRequest, request: Request
    ) -> InterestRateScenarioResponse:
        model = _price_model(request)
        try:
            result = compare_interest_rate_scenario(
                model, body.features, body.base_rate_before, body.base_rate_after
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "SCENARIO_NOT_SUPPORTED", "message": str(exc)},
            ) from exc
        return InterestRateScenarioResponse(**result.as_dict())

    return app


app = create_app()
