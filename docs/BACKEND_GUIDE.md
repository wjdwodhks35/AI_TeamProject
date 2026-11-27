# 백엔드 실행 및 인수인계

## 실행

프로젝트 루트에서 Python 3.11 이상을 사용한다. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:APARTMENT_ML_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"
.\.venv\Scripts\python.exe -m uvicorn apartment_ml.api.main:app --reload --log-level info
```

API 주소는 `http://127.0.0.1:8000`, Swagger는 `/docs`, 명세 JSON은 `/openapi.json`이다.
`.env`는 자동 로딩하지 않는다. 환경변수 변경 후 서버를 재시작한다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = '{"district":"강남구","area_m2":84.5,"floor":12,"built_year":2015,"contract_date":"2026-10-15","base_rate":2.5}'
Invoke-RestMethod http://127.0.0.1:8000/predict/price -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

모델 파일이 없으면 health는 200, 예측은 503이다. 성공 예측값을 임의로 만들지 않는다.

## ML 담당자 계약

기존 `models/base.py`의 `PricePredictor`, `AnomalyDetector`를 따른다.
공통 입력은 `schemas.ApartmentFeatures`, 파생변수 생성은 `build_feature_frame`을 사용한다.
회귀 입력에 `price_krw`, `price_per_m2`를 포함하지 않는다.
분류 입력에는 실제 가격과 `price_per_m2`를 추가한다.

가격 모델은 `predict(DataFrame)`에서 원 단위의 유한한 `(n,)` NumPy 배열을 반환한다.
분류 모델은 `predict_proba(DataFrame)`에서 이상 클래스(1)의 `(n,)` 확률 배열을 반환한다.
일반 sklearn 분류기의 `(n,2)` 출력은 이 계약으로 변환한 래퍼가 필요하다.
`is_fitted`, `name`, `feature_columns`가 필요하며 분류 임계값은 `threshold`(기본 0.5)이다.
금리 시나리오는 `feature_columns`에 `base_rate`가 있어야 한다.

파일은 기존 `PriceRegressionModel.save()` / `AnomalyClassificationModel.save()`의
joblib 아티팩트 형식을 사용한다. 설정 경로는 `configs/settings.yaml`의 `models`에 있다.
다른 설정은 `APARTMENT_ML_CONFIG` 환경변수로 지정한다. 신뢰할 수 있는 모델 파일만 로딩한다.
파일 부재·손상은 모델별 미준비로 처리한다. 모델 교체 후 서버를 재시작한다.
백엔드는 학습 코드를 추가하지 않는다.

## 프론트엔드 담당자 계약

요청·응답 예시는 [API_CONTRACT](API_CONTRACT.md)와 Swagger를 참조한다.
가격 예측 요청에 실거래가격을 보내지 않는다. 이상거래 요청에는 `price_krw`가 필수다.
시나리오는 `features`, `base_rate_before`, `base_rate_after`를 보내며
`features.base_rate`는 비교 금리로 덮어쓴다.

모든 오류는 `detail.code`, `detail.message`를 사용한다.
422 `VALIDATION_ERROR`에는 필드별 `issues`가 추가된다.
503 `MODEL_NOT_READY`는 '모델 준비 중'으로 표시한다.
422 `SCENARIO_NOT_SUPPORTED`는 금리를 입력으로 사용하지 않는 모델이다.
500 `INTERNAL_ERROR`는 재시도 안내를 표시하고 `X-Request-ID`로 서버 로그를 조회한다.
404와 405도 같은 오류 구조를 사용한다.

금리 비교 결과는 인과효과가 아니므로 `disclaimer`를 화면에 표시한다.
이상거래는 확정 판정이 아니므로 `warnings`를 표시한다.
실제 프론트엔드는 제공되지 않았으며 브라우저 화면 연동은 수행하지 않았다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m apartment_ml.pipeline
```

테스트 가짜 모델은 `tests/fakes.py`에만 존재하며 실제 예측 성능을 의미하지 않는다.
TestClient는 API 내부 연결과 CORS를 검사하며 실제 UI 통합 테스트를 대체하지 않는다.
GitHub Actions 기존 3.11/3.12/3.13 검증 구성을 재사용한다.

## 구현 범위와 남은 작업

구현: 네 API, 공통 Pydantic 검증, 모델 로딩/호출 경로, 준비 상태, 통일된 오류,
CORS, 요청 ID/처리시간/실패 로그, 금리 비교, 테스트, 인수인계 문서.

미수행: 실제 학습 모델 배포 및 결과 검증, 레이블 기반 이상거래 모델 검증,
프론트엔드 협의·화면 연동, 데이터→실제 모델→UI 전체 통합, 원격 CI 실행·배포.
이 항목들은 관련 산출물이 전달된 후 수행한다. 로컬 검증 결과는 `BACKEND_REPORT.md`에 기록한다.
