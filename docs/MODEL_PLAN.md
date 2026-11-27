# 모델 계획

> 이 문서에는 **성능 수치가 없습니다.** 실제 데이터로 학습·평가한 결과만 기록하며, 합성 샘플 데이터로 얻은 수치는 기록하지 않습니다.

## 1. 가격 회귀 (기능 A)
| 구분 | 내용 |
|---|---|
| 정답값 | `price_krw` (원) |
| 입력 변수 | `schemas.REGRESSION_FEATURES` (면적, 층, 건축연도, 건물연식, 계약 연·월, 기준금리, 규제 플래그, 자치구) |
| 기준 모델 | `LinearRegression` (`PriceRegressionModel("linear_regression")`) |
| 비교 모델 | `RandomForestRegressor` (`"random_forest"`). 새 모델은 `REGRESSOR_FACTORIES`에 등록 |
| 전처리 | Pipeline 안의 `ColumnTransformer`: 중앙값 대체 + 표준화 + One-Hot. **학습 데이터에만 fit** |
| 지표 | MAE, RMSE (`evaluation.metrics.regression_metrics`) |

검토할 사항 (feature/ml)
- 가격 분포가 한쪽으로 치우쳐 있으므로 `log(price)` 변환을 검토합니다. 변환한다면 평가는 원 단위로 되돌린 뒤 수행합니다.
- ㎡당 가격을 정답값으로 쓰는 방식과 총액을 직접 예측하는 방식을 비교합니다.
- 계약 시점의 추세를 반영하는 방법: 연·월 변수 외에 기준 시점으로부터 경과한 개월 수를 추가할지 검토합니다.

## 2. 이상거래 의심 분류 (기능 B)
| 구분 | 내용 |
|---|---|
| 정답값 | `anomaly_label` (0 = 정상, 1 = 의심) — **미확보** |
| 입력 변수 | 자치구, 전용면적, 층, 건축연도, 건물연식, 계약 연·월, 기준금리, 자치구 규제 3종, 거래가격, ㎡당 가격 |
| 개별 모델 | `DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier` (Boosting) |
| 앙상블 | `VotingClassifier` (soft voting) — `ensemble.build_voting_model()` |
| 지표 | Precision, Recall, F1, Confusion Matrix (`classification_metrics`) |

### 레이블 미확보 문제 (블로커)
- 현재 `AnomalyClassificationModel.fit()`은 `training_enabled=True`를 넘기지 않으면 `TrainingDisabledError`를 냅니다. 레이블이 전부 결측이거나 한 클래스뿐이면 `LabelsNotAvailableError`를 냅니다.
- **임의로 레이블을 만들지 않습니다.** 정상/이상 기준이 정해지기 전까지 분류 성능 수치는 존재하지 않습니다.
- 레이블 확보 방안 후보 (팀 합의 필요)
  1. 공식 자료에서 확인된 사례: 해제(취소)된 거래, 국토부·지자체 조사로 적발된 사례가 공개된 경우
  2. 규칙 기반 레이블: 예) 같은 단지·면적·기간의 시세에서 크게 벗어난 거래. 이 경우 모델이 결국 *규칙을 재현*하게 된다는 한계를 문서화해야 합니다.
  3. 비지도 이상탐지(IsolationForest 등)로 방향을 바꾸는 방안. 범위가 바뀌므로 지도교수 및 팀과 협의가 필요합니다.
### 학습 및 백엔드 연동

1. 데이터 담당자와 레이블 출처 및 판정 기준을 확정한 뒤, 실제 거래가 담긴 전처리 CSV의 `anomaly_label`에 승인된 값을 제공받습니다. `0`은 정상, `1`은 이상거래 의심, 빈 값은 미판정입니다. 저장소의 `TEST_ONLY` 샘플 122행은 모두 미판정입니다. `scripts/build_dataset.py --label`의 가격 편차 규칙은 팀 검토가 필요한 후보입니다.
2. `configs/settings.yaml`에서 `classification.training_enabled`를 `true`로 설정하고 저장소 루트에서 실행합니다. `--label-source`에는 실제 레이블 출처와 승인된 기준을 적습니다.

   ```bash
   python scripts/train_anomaly.py --data data/processed/training_dataset.csv --label-source "<승인된 레이블 출처와 기준>"
   ```

3. 전체 거래를 계약일 기준으로 분할한 뒤 구간별로 레이블이 있는 행을 사용합니다. 각 구간에서 정상과 의심 거래가 모두 필요합니다. 결측 대체 및 인코딩은 학습 구간에서만 적합합니다.
4. 개별 모델 3종과 동일 가중치 soft voting에 공통 임계값 0.5를 사용합니다. 검증 F1이 가장 높은 모델을 선택합니다. 동률이면 Recall, Precision, 모델 선언 순서로 결정합니다. 선택 후 테스트 구간에서 네 모델의 Precision, Recall, F1 및 혼동행렬을 계산합니다. 행렬의 행은 실제값, 열은 예측값이며 순서는 `[0, 1]`입니다.
5. 기본 출력은 `models/anomaly_model.joblib`와 `models/anomaly_model_metrics.json`입니다. joblib에는 선택된 모델과 전처리기가 들어갑니다. JSON에는 모델별 검증·테스트 지표, 구간별 계약일 범위, 클래스별 건수 및 미판정 건수, 레이블 출처가 들어갑니다.

API 서버는 시작할 때 설정된 모델 파일을 불러옵니다. [API 계약](API_CONTRACT.md)의 `POST /predict/anomaly`에 거래 조건과 원 단위 실제 거래가격 `price_krw`를 보내면 `is_suspicious`, `suspicion_score`, `threshold`, `model_name`을 반환합니다. 파이썬에서는 `AnomalyClassificationModel.load(path)`로 모델을 불러오고, 해당 모델의 `feature_columns`를 포함한 DataFrame에 `predict` 또는 `predict_proba`를 호출할 수 있습니다. 학습 입력은 API 요청에서 제공하거나 파생할 수 있는 변수로 구성합니다.

## 3. 시간순 검증
- 분할: `evaluation.split.time_based_split` (비율, 기본 70/15/15) 또는 `split_by_dates` (경계 날짜 지정)
  - 같은 계약일의 거래는 같은 구간에 들어갑니다. train < valid < test 순서를 지킵니다.
- 하이퍼파라미터 조정: train 데이터 안에서 `expanding_window_splits` (확장 윈도우 교차검증)를 사용합니다.
  - 일반 `KFold`나 셔플 분할은 **사용하지 않습니다** (미래 데이터 누출).
- test 데이터는 최종 보고 직전에 한 번만 사용합니다.
- 분할 기준인 계약일과 입력 변수인 계약 연·월은 별개입니다. 입력 변수로 쓰더라도 분할은 항상 계약일 기준으로 합니다.
- 금리·규제 변수는 계약일 시점에 유효한 값만 붙입니다 (`DATA_SCHEMA.md` §4).

## 4. 금리 시나리오 (기능 C)
- `scenarios.interest_rate.compare_interest_rate_scenario(model, features, before, after)`
- 금리를 뺀 나머지 조건은 그대로 두고, 같은 모델로 두 번 예측해 차이와 변화율을 계산합니다.
- 입력값이 학습 데이터의 범위를 벗어나면 warning을 추가합니다 (`model.feature_ranges`).
- 이 결과는 **인과효과가 아닙니다.** 선형 모델에서는 금리 계수가 그대로 나타나고, 트리 모델에서는 학습 구간 밖에서 예측값이 변하지 않을 수 있습니다. 이 한계를 UI에도 표시합니다.
- 추가할 기능 (feature/scenario): 여러 금리 값에 대한 연속 비교(sweep), 결과 시각화

## 5. 향후 12개월 가격 예측 (기능 E) — 설계 미확정
아직 구현하지 않습니다. 결정해야 할 사항은 다음과 같습니다.
1. 예측 대상: 특정 단지·조건의 가격인지, 자치구 가격지수인지
2. 방식: (a) 현재 회귀 모델에 미래 `contract_date`를 넣어 외삽 — 추세 외삽의 신뢰성이 낮음 / (b) 월별 집계 시계열 모델 / (c) 미래 시점별 direct multi-horizon 모델
3. 미래 기준금리를 어떻게 가정할지: 현재 값 유지, 시나리오 입력, 외부 전망치
4. 평가 방법: rolling-origin 백테스트에서 horizon별 오차 측정
5. 불확실성 표현: 예측구간을 함께 제시할지 여부
이 사항들이 결정되면 `scenarios/` 또는 `models/forecast.py`를 추가하고 API 계약도 확장합니다.
