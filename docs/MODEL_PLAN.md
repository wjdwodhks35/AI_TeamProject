# 모델 계획

> 성능 수치는 **실제 데이터로 학습·평가한 결과만** 기록합니다. 합성 샘플 데이터로 얻은 수치는 기록하지 않습니다.
> 아래 회귀 수치는 `data/processed/training_dataset.csv` (2021-01 ~ 2026-08, 257,655건) 를
> 계약일 기준 시간순 70/15/15 로 나눠 측정한 값입니다. `python scripts/train_price.py` 로 재현됩니다.

## 1. 가격 회귀 (기능 A)
| 구분 | 내용 |
|---|---|
| 정답값 | `price_krw` (원). 학습에는 **직전월 시장지수 대비 로그 평당가**를 쓰고 예측값은 원으로 되돌립니다 (아래 §1.1) |
| 입력 변수 | `schemas.REGRESSION_FEATURES` + `REGRESSION_LOCATION_FEATURES` (자치구·동·단지) |
| 채택 모델 | `HistGradientBoostingRegressor` (`PriceRegressionModel("gradient_boosting")`) |
| 비교 모델 | `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`. 새 모델은 `REGRESSOR_FACTORIES`에 등록 |
| 전처리 | Pipeline 안의 `ColumnTransformer`: 중앙값 대체 + 표준화 + One-Hot + 위치 타깃 인코딩. **학습 데이터에만 fit** |
| 지표 | MAE, RMSE, R², MAPE (`evaluation.metrics.regression_metrics`) |
| 실행 | `python scripts/train_price.py` |

기준선: **자치구 + 평형대(10㎡ 단위) 중앙값 조회**. 학습이 전혀 없는 방식이므로, 모델이
이걸 못 넘기면 모델을 쓸 이유가 없습니다.

측정 결과 (train 180,279 / valid 38,666 / test 38,710건, 아래 §1.1·§1.2 구성 적용)

| 모델 | 학습시간 | valid MAE | test MAE | R² | MAPE | 편향 |
|---|---|---|---|---|---|---|
| 기준선: 자치구+평형 중앙값 | — | 3.26억 | 3.59억 | 0.603 | 28.0% | −2.05억 |
| `linear_regression` | 2초 | 1.73억 | 1.90억 | 0.871 | 18.8% | +0.65억 |
| `decision_tree` | 4초 | 1.59억 | 1.69억 | 0.902 | 13.8% | −0.82억 |
| `random_forest` | 283초 | **1.45억** | 1.52억 | 0.925 | 12.5% | −0.79억 |
| `gradient_boosting` | 16초 | 1.50억 | **1.41억** | **0.935** | **11.8%** | −0.65억 |

**모델 선택 기준**: `scripts/train_price.py` 는 **valid MAE** 로 고릅니다. test 를 보고 고르면
test 가 더 이상 '한 번도 안 쓴 데이터'가 아니게 되므로, 점수가 실제보다 좋게 나옵니다.
그 규칙대로면 `random_forest` 가 선택됩니다. 다만 valid 차이(1.45 vs 1.50억, 3%)는 사실상
동률이고 `gradient_boosting` 이 **18배 빠르며** test 에서는 더 낫습니다. valid 가 동률일 때
학습 비용으로 가르는 것은 test 를 건드리지 않으므로 허용되는 판단입니다 — 팀에서 정하세요.
(`--models gradient_boosting` 으로 하나만 학습할 수 있습니다.)

### 1.1 왜 '절대 가격'이 아니라 '시세 대비 상대가격'인가

처음에는 `price_krw` 를 그대로 맞혔습니다. 그런데 **트리 계열은 학습 때 본 범위 밖으로
외삽하지 못합니다.** 시간순 분할에서 test 구간(2026년)은 학습 구간(~2025-07)보다 가격이
높은데, 모델은 마지막으로 학습한 가격대에서 멈춥니다. 측정된 증상:

- test 편향(예측 − 실제) **−1.69억**, 그리고 그 편향이 2026-02 −1.26억 → 2026-08 −2.01억으로
  **시간이 갈수록 커졌습니다**.
- `contract_year`·`contract_month` 를 빼도 R² 0.807 → 0.806 (차이 없음).
- 직전월 시장지수를 **설명변수로** 추가해도 편향 −1.66억 (거의 그대로). 설명변수도 외삽이
  안 되기 때문입니다.

그래서 시점 흐름을 학습에 맡기지 않고, **지수로 직접 처리**합니다 (`models/market_index.py`).

- 지수 = 그 자치구의 **직전월** 평균 로그 평당가. 거래가 10건 미만인 자치구-월은 서울 전체
  평균으로, 그마저 없으면 마지막 알려진 값으로 대체합니다.
- 모델이 맞히는 값 = `log(가격/면적) − 지수` (= 그 시점 시세의 몇 배인가)
- 예측 = `exp(모델 출력 + 지수) × 면적`

결과: 편향 −1.69억 → **−0.67억**, 월별 추세도 사라졌습니다. R² 0.860 → **0.935**.

**시점 규칙에 대한 해명**: 지수는 계약일 기준 **직전월** 거래로만 만듭니다. 실거래가는
계약 후 30일 내 신고되므로, 예측 시점에 직전월 거래는 이미 공개돼 있습니다. 즉 예측
시점에 알 수 없는 정보는 들어가지 않습니다. 다만 test 구간의 지수는 test 구간 거래로
만들어지므로, 학습 구간만으로 지수를 고정한 **엄격 조건**도 함께 측정했습니다:
test MAE 1.97억 / R² 0.892 — 이 경우에도 절대가격 방식(2.07억 / 0.860)보다 낫습니다.
즉 **상대가격이라는 구조 자체가 더 맞고**, 최신 지수를 쓰면 더 좋아집니다.

### 1.2 위치를 어디까지 넣는가

자치구 25개만으로는 서울 아파트 가격을 설명하기 어렵습니다. 동(332개)·단지(7,615개)를
넣으면 좋아지지만, 원-핫으로는 변수가 폭발하고 `HistGradientBoosting` 의 범주형 직접
처리는 255개까지만 됩니다. 그래서 **'그 위치의 평균 가격수준' 숫자 하나로 요약**합니다
(`models/location_price.py`).

| 위치 수준 | test MAE | R² |
|---|---|---|
| 자치구만 | 2.63억 | 0.807 |
| + 동 | 2.41억 | 0.839 |
| + 동 + 단지 | 2.17억 | 0.866 |

누출 방지 장치 (테스트로 고정: `tests/test_location_price.py`)
- 값은 **학습 데이터에서만** 계산합니다.
- 학습 행 자신의 값은 **5겹 out-of-fold** 로 채웁니다. 자기 거래가 자기 설명변수에 섞이면
  학습 성능만 올라가고 실제 성능은 떨어집니다.
- 거래가 적은 위치는 전체 평균 쪽으로 평활하고, 학습에 없던 위치는 전체 평균을 씁니다.
  그 위치의 학습 거래 수를 함께 내보내므로 모델이 '이 값을 믿을 만한지' 판단할 수 있습니다
  (test 거래의 2.7% 가 학습에 없던 단지입니다).

⚠️ **분류에는 쓰지 않습니다.** `CLASSIFICATION_*_FEATURES` 는 `REGRESSION_*_FEATURES` 를
그대로 물려받으므로, 위치 변수는 `REGRESSION_LOCATION_FEATURES` 라는 별도 상수로 두었습니다.
분류 입력에는 `price_krw` 가 있어서, 단지 수준 기준가가 함께 들어가면 `price_ratio` 를
역산해 이상거래 레이블 규칙을 그대로 복원할 수 있습니다.

### 1.3 남은 과제
- 편향 −0.67억이 남아 있습니다. 전 구간에 고르게 퍼져 있어 시점 문제는 아니고, 로그 타깃을
  지수로 되돌릴 때 생기는 편향(Jensen)일 가능성이 있습니다. 보정 계수를 검토할 수 있습니다.
- 하이퍼파라미터는 `max_iter=400`, `learning_rate=0.1` 고정값입니다. valid 구간으로 탐색하면
  더 줄어들 여지가 있습니다.
- API 단건 입력에는 동·단지가 없어 자치구 수준 추정으로 떨어집니다. 프론트에서 단지를
  선택할 수 있게 하면 정확도가 크게 올라갑니다 (위 표의 2.63억 → 2.17억 차이).
- `contract_year` 는 그대로 둡니다. 빼는 실험에서 valid 는 나빠지고(1.50 → 1.55억) test 는
  좋아져서(1.41 → 1.38억) 방향이 엇갈립니다 — 기여가 노이즈 수준이라는 뜻입니다. 선택
  규칙(valid)대로 유지하고, `CLASSIFICATION_FEATURES` 도 건드리지 않습니다.
- 그 결과 현재 날짜로 예측하면 `out_of_range_warnings` 가 `contract_year=2026 ... (외삽)`
  경고를 **매번** 냅니다. 값이 학습 범위 밖인 건 사실이지만, 가격 수준은 시장지수가
  처리하므로 예측이 그만큼 불안하다는 뜻은 아닙니다. 프론트에서 이 경고를 어떻게 보여줄지는
  UI 쪽 판단입니다 (`scenarios/interest_rate.py: out_of_range_warnings`).

## 2. 이상거래 의심 분류 (기능 B)
| 구분 | 내용 |
|---|---|
| 정답값 | `anomaly_label` (0 = 정상, 1 = 의심) — **미확보** |
| 입력 변수 | `schemas.CLASSIFICATION_FEATURES` (회귀 입력 + `price_krw`, `price_per_m2`) |
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
- 기준이 확정되면 다음 순서로 진행합니다: 이 문서에 정의 기록 → `settings.yaml`의 `classification.training_enabled: true` → 학습 코드에서 `training_enabled=settings.classification_training_enabled`를 전달
- 클래스 불균형이 예상되므로 `class_weight`, 임계값(`threshold`) 조정, PR-AUC 같은 보조 지표를 검토합니다.
- 기준가격(`reference_price_krw`)을 분류 입력으로 쓴다면, 학습 데이터에 대해서는 시간순 out-of-fold 예측값을 사용해야 합니다.

### Voting 앙상블 적용 계획
1. 개별 모델 3종을 같은 train 데이터로 학습하고, valid 데이터에서 하이퍼파라미터와 임계값을 정합니다.
2. soft voting(확률 평균)으로 결합합니다. 가중치는 valid 성능을 기준으로 정합니다.
3. test 데이터에서는 개별 모델 3종과 앙상블을 **한 번만** 평가해 같은 지표로 비교합니다.

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
