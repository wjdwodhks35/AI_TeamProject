# API 계약

- 구현: `src/apartment_ml/api/main.py`, 스키마: `src/apartment_ml/api/schemas.py`
- 실행 중에는 OpenAPI 문서를 `http://127.0.0.1:8000/docs`(Swagger)와 `/openapi.json`에서 확인할 수 있습니다.
- 모든 금액은 **원(KRW)**, 금리는 **%** 단위입니다. 날짜는 `YYYY-MM-DD` 형식입니다.

## 공통: 거래 조건 `ApartmentFeatures`
```json
{
  "district": "강남구",
  "contract_date": "2024-05-20",
  "area_m2": 84.9,
  "floor": 12,
  "built_year": 2008,
  "base_rate": 3.5,
  "regulation": {"speculation_overheated": true, "adjustment_target": true, "land_permit": false}
}
```
| 필드 | 필수 | 제약 |
|---|---|---|
| district | ✅ | 서울 25개 자치구 |
| contract_date | ✅ | 유효한 날짜 |
| area_m2 | ✅ | 10 ~ 400 |
| floor | ✅ | -5 ~ 100 |
| built_year | - | 1950 ~ 2030 |
| base_rate | - | -1 ~ 20 |
| regulation | - | 각 항목은 bool 또는 null |

정의되지 않은 필드는 거부됩니다 (`extra="forbid"`). 예를 들어 `/predict/price`에 `price_krw`를 넣으면 422가 반환됩니다.

## 오류 형식
```json
{"detail": {"code": "MODEL_NOT_READY", "message": "가격 예측 모델이 아직 학습/배포되지 않았습니다."}}
```
| HTTP | code | 의미 |
|---|---|---|
| 422 | `VALIDATION_ERROR` | 요청 스키마 검증 실패 (`detail.issues`에 location/message/type 제공) |
| 500 | `INTERNAL_ERROR` | 모델 예측 또는 서버 처리 실패 |
| 422 | `SCENARIO_NOT_SUPPORTED` | 금리를 입력 변수로 쓰지 않는 모델 등, 시나리오 계산 불가 |
| 503 | `MODEL_NOT_READY` | 모델이 학습·배포되지 않음. **가짜 예측값은 반환하지 않음** |

---

## `GET /health`
```json
{"status": "ok", "version": "0.1.0", "models": {"price": false, "anomaly": false}}
```

## `POST /predict/price`
- 요청: `ApartmentFeatures`
- 응답 200:
```json
{"predicted_price_krw": 2350000000, "model_name": "linear_regression", "warnings": []}
```
- `warnings`: 입력값이 학습 데이터 범위를 벗어났을 때(외삽) 등에 채워집니다.
- 503: 가격 모델이 없을 때

## `POST /predict/anomaly`
- 요청: `ApartmentFeatures` + `"price_krw": 2500000000` (실제 거래가격, 필수)
- 응답 200:
```json
{
  "is_suspicious": true,
  "suspicion_score": 0.82,
  "threshold": 0.5,
  "reference_price_krw": 2100000000,
  "price_gap_pct": 19.05,
  "model_name": "voting_decision_tree+random_forest+hist_gradient_boosting",
  "warnings": []
}
```
- `is_suspicious`는 *의심* 판정일 뿐, 위법 여부를 확정하는 값이 아닙니다.
- `reference_price_krw`(기준가격)는 가격 회귀 모델의 예상가격입니다. 가격 모델이 없으면 `null`이고 warning이 추가됩니다.
- 503: 분류 모델이 없거나 미학습인 경우입니다. 실제 모델이 배포되면 위 계약으로 판정합니다.

## `POST /scenario/interest-rate`
- 요청:
```json
{"features": { "...ApartmentFeatures..." }, "base_rate_before": 3.5, "base_rate_after": 2.5}
```
  `features.base_rate` 값은 무시됩니다. 금리를 뺀 나머지 조건은 그대로 고정됩니다.
- 응답 200:
```json
{
  "base_rate_before": 3.5,
  "base_rate_after": 2.5,
  "predicted_price_before_krw": 2300000000.0,
  "predicted_price_after_krw": 2350000000.0,
  "price_diff_krw": 50000000.0,
  "price_change_pct": 2.17,
  "model_name": "linear_regression",
  "warnings": [],
  "disclaimer": "모델 예측값의 조건별 비교 결과이며, 금리 변동이 실제 시장 가격에 미치는 인과효과를 의미하지 않습니다."
}
```
- `price_change_pct`: 기존 조건의 예측가격이 0이면 `null`
- 503: 가격 모델이 없을 때 / 422: 모델이 `base_rate`를 입력 변수로 쓰지 않을 때

## 모델 배포 방식
서버가 시작될 때 `configs/settings.yaml`의 `models.price_model_path`와 `models.anomaly_model_path` 파일이 있으면 해당 모델을 불러옵니다.
모델 파일은 `PriceRegressionModel.save()` / `AnomalyClassificationModel.save()`로 만듭니다 (joblib 형식, `format_version=1`).
joblib 파일은 신뢰할 수 있는 출처에서 받은 것만 사용해야 합니다.

## 변경 규칙
요청·응답 필드를 추가하거나 바꿀 때는 `api/schemas.py`, 이 문서, `tests/test_api.py`를 같은 PR에서 함께 수정합니다.
기존 필드를 삭제하거나 이름을 바꾸면 UI 담당자와 먼저 합의해야 합니다.
