# 아키텍처

단일 Python 패키지(`apartment_ml`)와 파일 기반 저장소로 구성합니다. DB, 메시지 큐, 마이크로서비스는 쓰지 않습니다.

## 데이터 흐름
```text
 [국토교통부 API/CSV]   [ECOS 기준금리]   [규제 이력(수작업)]
          │                    │                  │
     data.collect  ──────────────────────────────── (원본 → data/raw)
          │ normalize_molit_api_frame (원본 컬럼 → 내부 컬럼)
          ▼
   data.preprocess.preprocess_transactions   ← 행 단위: 검증·타입·중복·파생 (누출 없음)
          ▼
   data.merge.attach_base_rate / attach_regulation  ← 계약일 시점에 유효한 값만 붙임
          ▼                                          (→ data/processed)
   evaluation.split.time_based_split  → train / valid / test (계약일 기준)
          ▼
   models.regression / classification / ensemble
     └ Pipeline[ColumnTransformer(대체·스케일·인코딩) → 추정기]  ← train에만 fit
          │  evaluation.metrics (MAE/RMSE, P/R/F1/CM)
          │  save() → models/*.joblib
          ▼
   api.main (FastAPI)  ──  scenarios.interest_rate
          ▼
       웹 UI (feature/ui)
```

## 모듈 의존 관계
```text
schemas  ←  config
   ↑
data.preprocess  ←  data.collect, data.merge
   ↑
evaluation.metrics, evaluation.split
   ↑
models.base ← models.regression, models.classification ← models.ensemble
   ↑
scenarios.interest_rate   (models.base의 PricePredictor 계약에만 의존)
   ↑
api.main / api.schemas    (PricePredictor / AnomalyDetector 계약에만 의존)
```
- 의존은 아래쪽(하위 모듈) 방향으로만 갑니다. `schemas`는 다른 모듈을 import하지 않습니다.
- API와 시나리오는 구체 클래스가 아니라 **Protocol(`PricePredictor`, `AnomalyDetector`)** 에만 의존합니다. 따라서 ML 담당자가 모델을 바꿔도 API 코드는 수정하지 않아도 됩니다.

## 모듈별 역할
| 모듈 | 역할 | 주 담당 |
|---|---|---|
| `schemas.py` | 컬럼명, 유효 범위, 입력 변수/정답값 구분, 원본 컬럼 매핑, 단건 Pydantic 모델 | 공통 (변경 시 전원 리뷰) |
| `config.py` | settings.yaml 로딩, API 키는 환경변수에서만 읽음 | 공통 |
| `data/collect.py` | 수집기 Protocol, MOLIT 정규화 | feature/data |
| `data/merge.py` | 기준금리·규제를 시점에 맞춰 병합 | feature/data |
| `data/preprocess.py` | 행 단위 전처리, 학습형 전처리(ColumnTransformer) 팩토리 | feature/data (+ml) |
| `evaluation/split.py` | 시간순 분할, 확장 윈도우 교차검증 | feature/ml |
| `evaluation/metrics.py` | 회귀·분류 지표 | feature/ml |
| `models/*` | 모델 래퍼, 레지스트리, 저장/불러오기 | feature/ml |
| `scenarios/interest_rate.py` | 금리 조건 비교 | feature/scenario |
| `api/*` | FastAPI 엔드포인트, 요청·응답 스키마 | feature/ui |
| `pipeline.py` | 합성 샘플 데이터로 공통 기반을 점검하는 최소 실행 경로 | 공통 |

## 확장 지점 (기존 코드를 최소한으로 수정하는 방법)
- 회귀·분류 모델 추가: `REGRESSOR_FACTORIES` / `CLASSIFIER_FACTORIES`에 항목 하나를 추가합니다.
- 입력 변수 추가: `schemas.REGRESSION_*_FEATURES`에 추가하고, 파생 로직은 `preprocess`에 추가합니다.
- 수집기 추가: `collect`의 Protocol을 구현하는 클래스를 추가합니다.
- API 모델 교체: 모델 파일을 바꾸거나 `create_app(price_model=...)`으로 주입합니다.
