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
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apartment_ml import __version__
from apartment_ml import schemas as S
from apartment_ml.api.schemas import (
    AnomalyPredictionRequest,
    AnomalyPredictionResponse,
    ComplexItem,
    ErrorResponse,
    ForecastPointOut,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    InterestRateScenarioRequest,
    InterestRateScenarioResponse,
    LocationsResponse,
    PricePredictionRequest,
    PricePredictionResponse,
    RegulationStateResponse,
)
from apartment_ml.config import get_cors_origins, get_settings
from apartment_ml.data.preprocess import add_price_features
from apartment_ml.data.regulation_lookup import RegulationLookup, RegulationLookupError
from apartment_ml.models.anomaly_rules import RuleLevel, rule_verdict
from apartment_ml.models.base import AnomalyDetector, ModelNotReadyError, PricePredictor
from apartment_ml.models.complex_directory import ComplexDirectory, ComplexDirectoryError
from apartment_ml.models.forecast import (
    BAND_Z,
    ForecastError,
    MarketIndexForecaster,
    forecast_prices,
)
from apartment_ml.scenarios.interest_rate import (
    ScenarioNotSupportedError,
    build_feature_frame,
    compare_interest_rate_scenario,
    out_of_range_warnings,
)

logger = logging.getLogger(__name__)

NOT_READY_RESPONSES = {
    422: {"model": ErrorResponse, "description": "입력 검증 또는 시나리오 지원 오류"},
    500: {"model": ErrorResponse, "description": "서버 또는 모델 처리 실패"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "모델 미준비"},
}


NO_COMPLEX_WARNING = (
    "단지(complex_id)를 지정하지 않아 자치구 수준으로 추정했습니다. "
    "측정 기준 평균오차가 1.41억 -> 4.18억으로 늘어납니다. "
    "GET /locations/{자치구} 에서 단지를 골라 complex_id 를 함께 보내면 정확해집니다."
)


def _location_warnings(
    features: S.ApartmentFeatures, directory: ComplexDirectory | None
) -> list[str]:
    """단지 입력 상태에 대한 경고. 조용히 거친 추정을 주지 않기 위한 것이다."""
    complex_id = (features.complex_id or "").strip()
    if not complex_id:
        return [NO_COMPLEX_WARNING]
    if directory is not None and not directory.has(features.district, complex_id):
        return [
            f"단지 일련번호 {complex_id!r} 를 {features.district} 목록에서 찾지 못했습니다. "
            f"모델이 본 적 없는 단지이므로 자치구 수준 추정에 가깝습니다."
        ]
    return []


def _not_ready(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "MODEL_NOT_READY", "message": message},
    )


def _build_forecaster(price_model: object) -> MarketIndexForecaster | None:
    """가격 모델 안의 시장지수로 예측기를 만든다.

    지수가 모델 안에 들어 있으므로 원본 CSV 가 없어도 되고, 모델과 지수가 어긋날
    일도 없다. 백테스트까지 해도 0.2초면 끝난다.
    """
    market = getattr(price_model, "market_", None)
    if market is None:
        return None
    try:
        return MarketIndexForecaster.from_index(market)
    except ForecastError as exc:
        logger.warning("시세 예측기를 만들 수 없습니다: %s", exc)
        return None


def _load_regulation_lookup() -> RegulationLookup | None:
    """고시 이력을 읽어 둔다. 없으면 규제 자동 입력만 빠지고 나머지는 그대로 동작한다."""
    try:
        return RegulationLookup.load()
    except RegulationLookupError as exc:
        logger.warning("규제 이력 로드 실패: %s", exc)
        return None


def _apply_regulation(features: S.ApartmentFeatures, request: Request) -> S.ApartmentFeatures:
    """비어 있는 규제 항목을 고시 이력에서 채운다.

    투기과열지구 같은 값은 고시로 정해진 사실이라 사용자가 알아맞힐 값이 아니다.
    요청에 명시된 값이 있으면 그대로 둔다 (가정을 바꿔 보고 싶을 수 있다).
    """
    lookup: RegulationLookup | None = request.app.state.regulation_lookup
    if lookup is None:
        return features
    known = lookup.at(features.district, features.contract_date)
    current = features.regulation
    filled = S.RegulationInfo(
        speculation_overheated=(
            current.speculation_overheated
            if current and current.speculation_overheated is not None
            else bool(known[S.REG_SPECULATION_OVERHEATED])
        ),
        adjustment_target=(
            current.adjustment_target
            if current and current.adjustment_target is not None
            else bool(known[S.REG_ADJUSTMENT_TARGET])
        ),
        land_permit=(
            current.land_permit
            if current and current.land_permit is not None
            else bool(known[S.REG_LAND_PERMIT])
        ),
    )
    return features.model_copy(update={"regulation": filled})


def _load_complex_directory() -> ComplexDirectory | None:
    """단지 목록을 읽는다. 없으면 /locations 만 503 이 되고 예측은 그대로 동작한다."""
    try:
        return ComplexDirectory.load(get_settings().complex_directory_path)
    except ComplexDirectoryError as exc:
        logger.warning("단지 목록 로드 실패: %s", exc)
        return None


def _load_models_from_disk() -> tuple[PricePredictor | None, AnomalyDetector | None]:
    # 지연 import: 모델 의존성이 없어도 /health 는 동작하도록
    from apartment_ml.models.classification import AnomalyClassificationModel
    from apartment_ml.models.regression import PriceRegressionModel

    settings = get_settings()
    price = anomaly = None
    if settings.price_model_path.exists():
        try:
            price = PriceRegressionModel.load(settings.price_model_path)
        except Exception as exc:
            logger.warning("가격 모델 로드 실패: %s", exc)
    if settings.anomaly_model_path.exists():
        try:
            anomaly = AnomalyClassificationModel.load(settings.anomaly_model_path)
        except Exception as exc:
            logger.warning("이상거래 모델 로드 실패: %s", exc)
    return price, anomaly


def create_app(
    price_model: PricePredictor | None = None,
    anomaly_model: AnomalyDetector | None = None,
    load_from_disk: bool = True,
    complex_directory: ComplexDirectory | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_from_disk:
            loaded_price, loaded_anomaly = _load_models_from_disk()
            if price_model is None:
                app.state.price_model = loaded_price
            if anomaly_model is None:
                app.state.anomaly_model = loaded_anomaly
            if complex_directory is None:
                app.state.complex_directory = _load_complex_directory()
        app.state.regulation_lookup = _load_regulation_lookup()
        app.state.forecaster = (
            _build_forecaster(app.state.price_model) if app.state.price_model else None
        )
        yield

    app = FastAPI(
        title="서울 아파트 가격 예측 및 이상거래 의심 탐지 API",
        version=__version__,
        description="Phase 0: API 계약. 모델이 준비되지 않으면 503 을 반환합니다.",
        lifespan=lifespan,
    )
    app.state.price_model = price_model
    app.state.anomaly_model = anomaly_model
    app.state.complex_directory = complex_directory
    app.state.forecaster = None
    app.state.regulation_lookup = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "VALIDATION_ERROR",
                    "message": "요청 입력값을 확인해 주세요.",
                    "issues": [
                        {"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}
        )
        return JSONResponse(
            status_code=exc.status_code, content={"detail": detail}, headers=exc.headers
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready(request: Request, exc: ModelNotReadyError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": {"code": "MODEL_NOT_READY", "message": "모델이 준비되지 않았습니다."}
            },
        )

    @app.exception_handler(Exception)
    async def prediction_error(request: Request, exc: Exception):
        logger.error(
            "request_id=%s stage=backend_or_model path=%s",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
            exc_info=exc,
        )
        headers = {"X-Request-ID": getattr(request.state, "request_id", "unknown")}
        origin = request.headers.get("origin")
        if origin in get_cors_origins():
            headers.update(
                {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Expose-Headers": "X-Request-ID",
                    "Vary": "Origin",
                }
            )
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "INTERNAL_ERROR",
                    "message": "서버 처리에 실패했습니다.",
                }
            },
            headers=headers,
        )

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

    @app.get(
        "/regulation/{district}",
        response_model=RegulationStateResponse,
        responses={404: {"model": ErrorResponse, "description": "서울 25개 자치구가 아님"}},
        tags=["reference"],
    )
    def regulation_state(
        district: str, request: Request, on: date | None = None
    ) -> RegulationStateResponse:
        """그 날 그 자치구에 걸려 있던 규제. 입력칸 대신 이걸 보여주면 된다."""
        lookup: RegulationLookup | None = request.app.state.regulation_lookup
        if lookup is None:
            raise _not_ready("규제 이력을 읽을 수 없습니다 (configs/regulation_events.csv).")
        name = district.strip()
        if name not in S.SEOUL_DISTRICTS:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DISTRICT_NOT_FOUND",
                    "message": f"서울 25개 자치구 중 하나여야 합니다: {name!r}",
                },
            )
        when = on or date.today()
        flags = lookup.at(name, when)
        return RegulationStateResponse(
            district=name,
            on=when.isoformat(),
            speculation_overheated=bool(flags[S.REG_SPECULATION_OVERHEATED]),
            adjustment_target=bool(flags[S.REG_ADJUSTMENT_TARGET]),
            land_permit=bool(flags[S.REG_LAND_PERMIT]),
        )

    @app.get(
        "/locations/{district}",
        response_model=LocationsResponse,
        responses={
            404: {"model": ErrorResponse, "description": "서울 25개 자치구가 아님"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "단지 목록 미준비",
            },
        },
        tags=["reference"],
    )
    def locations(district: str, request: Request) -> LocationsResponse:
        """그 자치구의 동·단지 목록. 자치구를 고를 때 한 번 호출해 두 드롭다운을 채운다."""
        directory: ComplexDirectory | None = request.app.state.complex_directory
        if directory is None:
            raise _not_ready(
                "단지 목록이 준비되지 않았습니다 (scripts/train_price.py 를 실행하세요)."
            )
        name = district.strip()
        if name not in S.SEOUL_DISTRICTS:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DISTRICT_NOT_FOUND",
                    "message": f"서울 25개 자치구 중 하나여야 합니다: {name!r}",
                },
            )
        entries = directory.complexes(name)
        return LocationsResponse(
            district=name,
            dongs=directory.dongs(name),
            complexes=[ComplexItem(**e.to_dict()) for e in entries],
        )

    @app.post(
        "/predict/price",
        response_model=PricePredictionResponse,
        responses=NOT_READY_RESPONSES,
        tags=["prediction"],
    )
    def predict_price(body: PricePredictionRequest, request: Request) -> PricePredictionResponse:
        model = _price_model(request)
        frame = build_feature_frame(_apply_regulation(body, request))
        predictions = model.predict(frame)
        if getattr(predictions, "shape", None) != (1,):
            raise RuntimeError("Price predictions must have shape (1,)")
        pred = float(predictions[0])
        if not math.isfinite(pred):
            raise RuntimeError("Price prediction must be finite")
        warnings = _location_warnings(body, request.app.state.complex_directory)
        warnings += out_of_range_warnings(model, frame)
        if pred <= 0:
            warnings.append(
                "모델 예측값이 0 이하입니다. 입력 조건이 학습 범위를 벗어났을 수 있습니다."
            )
        return PricePredictionResponse(
            predicted_price_krw=round(pred), model_name=model.name, warnings=warnings
        )

    @app.post(
        "/forecast/price",
        response_model=ForecastResponse,
        responses=NOT_READY_RESPONSES,
        tags=["prediction"],
    )
    def forecast_price(body: ForecastRequest, request: Request) -> ForecastResponse:
        """향후 N개월 예상 거래가격.

        회귀가 시세 대비 상대가격을 학습하므로, 미래 가격은 시장지수만 내다보면 된다.
        구간은 백테스트로 잰 지평별 오차와 모델 잔차를 합쳐 만든 80% 범위다.
        """
        model = _price_model(request)
        forecaster: MarketIndexForecaster | None = request.app.state.forecaster
        if forecaster is None:
            raise _not_ready("시세 예측기가 준비되지 않았습니다 (가격 모델을 다시 학습하세요).")

        features = _apply_regulation(
            S.ApartmentFeatures(**body.model_dump(exclude={"months"})), request
        )
        frame = build_feature_frame(features)
        try:
            points = forecast_prices(model, frame, forecaster, months=body.months)
        except (ForecastError, ValueError) as exc:
            raise _not_ready(str(exc)) from exc

        warnings = _location_warnings(body, request.app.state.complex_directory)
        # 구간이 넓은 이유를 나눠서 알려 준다. 대부분은 시점이 아니라 개별 거래 편차다.
        base_pct = (math.exp(BAND_Z * getattr(model, "log_residual_sd_", 0.0)) - 1) * 100
        last_pct = (points[-1].high_krw / points[-1].price_krw - 1) * 100
        warnings.append(
            f"{body.months}개월 뒤 구간은 ±{last_pct:.0f}% 입니다. 이 가운데 ±{base_pct:.0f}%p 는 "
            f"시점과 무관하게 같은 조건에서도 거래마다 가격이 다르기 때문이고, 나머지가 "
            f"시세를 내다보는 몫입니다."
        )
        return ForecastResponse(
            base_month=str(forecaster.last_month),
            model_name=model.name,
            points=[
                ForecastPointOut(
                    month=p.month,
                    horizon=p.horizon,
                    predicted_price_krw=round(p.price_krw),
                    low_krw=round(p.low_krw),
                    high_krw=round(p.high_krw),
                )
                for p in points
            ],
            warnings=warnings,
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
        features = _apply_regulation(
            S.ApartmentFeatures.model_validate(body.model_dump(exclude={"price_krw"})), request
        )
        frame = build_feature_frame(features)
        frame[S.PRICE_KRW] = pd.array([body.price_krw], dtype="Int64")
        frame = add_price_features(frame)
        probabilities = model.predict_proba(frame)
        if getattr(probabilities, "shape", None) != (1,):
            raise RuntimeError("Anomaly probabilities must have shape (1,)")
        score = float(probabilities[0])
        threshold = float(getattr(model, "threshold", 0.5))
        if not (0 <= score <= 1 and 0 <= threshold <= 1):
            raise RuntimeError("Invalid anomaly probability or threshold")

        warnings: list[str] = ["이 결과는 이상거래 의심 판정이며 위법 여부를 확정하지 않습니다."]
        # 기준가격은 가격 회귀 모델이 내므로 단지 입력이 없으면 같이 거칠어진다.
        warnings += _location_warnings(features, request.app.state.complex_directory)
        reference = gap = None
        price_model = request.app.state.price_model
        if price_model is not None and price_model.is_fitted:
            reference = round(float(price_model.predict(build_feature_frame(features))[0]))
            if reference > 0:
                gap = (body.price_krw - reference) / reference * 100.0
        else:
            warnings.append("가격 모델이 없어 기준가격을 제공하지 않습니다.")

        # 규칙 판정을 같이 낸다. 모델의 레이블은 '기준가보다 싼 거래' 인데 분류 입력에
        # 기준가가 없어서 모델이 규칙을 복원하지 못한다. 기준가는 응답에 이미 있으므로
        # 규칙을 그대로 적용하면 된다. test F1 모델 0.194 vs 규칙 0.429.
        verdict = rule_verdict(gap, body.is_direct_deal)
        if verdict.level is not RuleLevel.NORMAL and not (score >= threshold):
            warnings.append(
                "규칙과 모델의 판정이 다릅니다. 측정상 규칙이 더 정확하므로 "
                "(F1 0.429 vs 0.194) 규칙 판정을 먼저 보세요."
            )
        return AnomalyPredictionResponse(
            is_suspicious=score >= threshold,
            suspicion_score=score,
            threshold=threshold,
            reference_price_krw=reference,
            price_gap_pct=gap,
            model_name=model.name,
            rule_level=str(verdict.level),
            rule_label=verdict.label,
            rule_reasons=verdict.reasons,
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
                model,
                _apply_regulation(body.features, request),
                body.base_rate_before,
                body.base_rate_after,
            )
        except ScenarioNotSupportedError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "SCENARIO_NOT_SUPPORTED", "message": str(exc)},
            ) from exc
        return InterestRateScenarioResponse(**result.as_dict())

    return app


app = create_app()
