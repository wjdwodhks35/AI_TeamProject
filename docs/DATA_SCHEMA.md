# 데이터 스키마 (공통 데이터 계약)

코드 기준은 `src/apartment_ml/schemas.py`입니다. 필드를 바꿀 때는 **코드와 이 문서를 같은 PR에서 함께** 수정합니다.

## 1. 거래 테이블 (내부 표준)

| 필드 | 타입 | 단위 | 필수 | 설명 / 유효 범위 |
|---|---|---|---|---|
| `district` | str | - | ✅ | 서울 25개 자치구명 (`SEOUL_DISTRICTS`). 그 외 값은 오류 |
| `contract_date` | datetime64[ns] (날짜) | YYYY-MM-DD | ✅ | 계약일. 시간 정보는 버리고(normalize) 저장 |
| `area_m2` | float | ㎡ | ✅ | 전용면적, 10 ~ 400 |
| `floor` | Int64 | 층 | ✅ | -5 ~ 100 (원본에서 지하층이 음수로 기록될 수 있음) |
| `price_krw` | Int64 | **원** | ✅ | 실거래가. 1 이상 1조 이하. 원본 단위인 '만원'에 10,000을 곱해 저장 |
| `built_year` | Int64 (nullable) | 년 | - | 1950 ~ 2030 |
| `base_rate` | float (nullable) | % | - | 계약일에 **적용 중이던** 기준금리 (§4) |
| `reg_speculation_overheated` | Int64 0/1 (nullable) | - | - | 투기과열지구 |
| `reg_adjustment_target` | Int64 0/1 (nullable) | - | - | 조정대상지역 |
| `reg_land_permit` | Int64 0/1 (nullable) | - | - | 토지거래허가구역 |
| `anomaly_label` | Int64 0/1 (nullable) | - | - | 이상거래 **정답 레이블**. 현재 미확보 상태이며, 전부 null이어도 됨 |
| `apartment_name` | str (nullable) | - | - | 단지명 |
| `dong` | str (nullable) | - | - | 법정동 |
| `source` | str (nullable) | - | - | 데이터 출처 (`molit_api`, `synthetic_sample` 등) |
| `complex_id` | str (nullable) | - | - | 단지 일련번호(`aptSeq`). 같은 이름의 다른 단지를 구분 |
| `is_direct_deal` | Int64 0/1 (nullable) | - | - | 직거래 1 / 중개거래 0. 2021년 자료에는 거의 없음 |
| `agent_location` | Int64 0/1/2 (nullable) | - | - | 중개사 소재지: 0 없음 / 1 같은 자치구 / 2 다른 자치구 |
| `land_leasehold` | Int64 0/1 (nullable) | - | - | 토지임대부 1. 땅값이 빠져 같은 평형이라도 훨씬 싸다 |
| `base_rate_change_12m` | float (nullable) | %p | - | 계약일과 12개월 전 기준금리의 차이 (§4) |
| `base_rate_change_3m` | float (nullable) | %p | - | 계약일과 3개월 전 기준금리의 차이 |

### 파생 컬럼 (행 단위 계산이므로 데이터 누출 없음)
| 필드 | 계산식 | 사용 가능 모델 |
|---|---|---|
| `contract_year`, `contract_month` | 계약일의 연도·월 | 회귀·분류 |
| `building_age` | `contract_year - built_year` | 회귀·분류 |
| `price_per_m2` | `price_krw / area_m2` | **분류만** (정답값에서 나온 값이므로 회귀 입력 금지) |
| `peer_median_price_krw` | 직전 기간 같은 조건의 ㎡당 시세 중앙값 × 면적 (§9) | **입력 금지** |
| `peer_tier` | 기준가를 만든 그룹 단계 (1~4, 0=못 만듦) | **입력 금지** |
| `price_ratio` | `price_krw / peer_median_price_krw` | **입력 금지** (레이블의 정의) |

## 2. 입력 변수와 정답값의 구분

| 구분 | 가격 회귀 (기능 A·C) | 이상거래 분류 (기능 B) |
|---|---|---|
| 정답값 | `price_krw` | `anomaly_label` |
| 입력 변수 | `REGRESSION_FEATURES` | `CLASSIFICATION_FEATURES` (= 회귀 입력 + `price_krw`, `price_per_m2`) |
| 입력 금지 | `price_krw`, `price_per_m2`, `anomaly_label`, `reference_price_krw`, `peer_median_price_krw`, `price_ratio`, `peer_tier` | `anomaly_label`, `peer_median_price_krw`, `price_ratio`, `peer_tier` |

이상거래 레이블은 `price_ratio = price_krw / peer_median_price_krw`로 만듭니다. 분류 모델은 `price_krw`를
입력으로 쓸 수 있으므로, 기준가가 함께 들어가면 **레이블을 그대로 역산할 수 있습니다.** `peer_tier`도 넓은
단계일수록 이상 판정 비율이 높아(§9) 부분적인 누출입니다. 세 컬럼 모두 입력에서 막혀 있습니다.

`PriceRegressionModel`과 `AnomalyClassificationModel`은 금지 컬럼이 입력 변수에 들어오면 생성 단계에서 `ValueError`를 발생시킵니다.

### '기준가격' (`reference_price_krw`)
- **의미:** 동일한 거래 조건에서 가격 회귀 모델이 산출한 *예상가격*. UI와 이상거래 판정에서 실제 거래가와 비교하는 기준입니다.
- **실제 거래가(`price_krw`)와는 다른 값**이며, 데이터 원천 컬럼이 아닙니다. 회귀 모델의 정답값으로 쓰지 않습니다.
- 분류 모델의 입력 변수로 쓰려면 학습 데이터에 대해 **out-of-fold(시간순)** 로 생성해야 합니다. 그렇게 하지 않으면 누출이 생깁니다.

## 3. 국토교통부 원본 → 내부 컬럼 매핑
`schemas.MOLIT_API_COLUMN_MAP`(OpenAPI)과 `schemas.MOLIT_CSV_COLUMN_MAP`(실거래가 공개시스템 CSV)을 사용합니다.

| OpenAPI | CSV | 내부 |
|---|---|---|
| `sggCd` | `시군구` (문자열 분리 필요) | `district` (`SEOUL_LAWD_CODES`로 변환) |
| `umdNm` | (`시군구`에 포함) | `dong` |
| `aptNm` | `단지명` | `apartment_name` |
| `excluUseAr` | `전용면적(㎡)` | `area_m2` |
| `dealYear`/`dealMonth`/`dealDay` | `계약년월` + `계약일` | `contract_date` |
| `dealAmount` (만원, 쉼표 포함) | `거래금액(만원)` | `price_krw` (×10,000) |
| `floor` | `층` | `floor` |
| `buildYear` | `건축년도` | `built_year` |
| `aptSeq` | - | `complex_id` |
| `dealingGbn` | `거래유형` | `deal_type` → `is_direct_deal` |
| `estateAgentSggNm` | - | `agent_district` → `agent_location` |
| `landLeaseholdGbn` | - | `land_leasehold_raw` → `land_leasehold` |
| `cdealType`, `cdealDay` | `해제사유발생일` | 해제된 거래는 **쌍 전체를 제거** (아래) |

**해제(취소)된 거래:** 국토교통부 자료는 계약이 해제되면 원본 행을 그대로 남기고 해제 행을 **한 줄 더**
넣습니다. 해제 행만 지우면 원본 행이 남아 깨진 계약이 정상 거래로 보입니다. 실제 수집 데이터(서울 284,111건)에서
이 쌍이 10,698건 있었습니다. `normalize_molit_api_frame`은 같은 거래(§6의 중복 키)를 모두 제거하며,
`drop_cancelled=False`로 호출하면 모두 남기고 `cancel_type`으로 해제 여부를 구분할 수 있습니다
(이상거래 레이블 검증용).

**수집하지 않는 항목:** `slerGbn`(매도자), `buyerGbn`(매수자)는 2024년부터만 제공됩니다. 값이 있다는 것
자체가 "2024년 이후 거래"라는 뜻이 되어 시점 정보가 새어 들어가므로 쓰지 않습니다. `rgstDate`(등기일자)와
`aptDong`(동)은 계약 이후에 생기는 값이고, 해제된 거래에는 **한 건도 없어** 레이블을 그대로 알려줍니다.

> 실제로 수집할 때 원본 필드명을 다시 확인해야 합니다. 필드명이 바뀌면 매핑 dict만 수정하면 됩니다.
> OpenAPI 정규화는 `collect.normalize_molit_api_frame`이 담당합니다. CSV 정규화 함수는 feature/data에서 구현합니다.

## 4. 시점 정합성 (기준금리·규제)
- **기준금리 테이블:** `effective_date`, `base_rate`
  - `effective_date`는 *변경 금리가 적용되기 시작한 날*입니다. 금통위 발표일과 다를 수 있습니다.
  - 각 거래에는 `effective_date <= contract_date`를 만족하는 가장 최근 값을 붙입니다 (`merge.attach_base_rate`, `merge_asof backward`).
  - 첫 적용일보다 이른 계약은 NaN으로 둡니다.
  - **금리는 수준으로 모델에 넣지 않습니다.** 금리가 오르는 시기는 경기가 좋은 시기이기도 해서, 수준끼리
    비교하면 "금리가 높을 때 집값도 높다"는 거꾸로 된 관계가 나옵니다. 수집 자료(68개월)를 월별 평균
    평당가로 집계하면 기준금리 수준과의 상관은 **+0.209**(부호가 뒤집힘), 전년 대비 변화는 **-0.752**입니다.
    `merge.attach_base_rate_changes`가 12개월·3개월 변화량을 붙입니다.
    월별 시계열은 추세가 강해 상관이 부풀려지므로, 이 숫자는 효과 크기가 아니라 '수준이 아니라 변화를
    써야 한다'는 근거로만 씁니다.
  - 기준금리는 금융통화위원회 결정일에 바뀌므로 **일별로 받아 값이 바뀐 날만** 남깁니다
    (`collect.EcosBaseRateCollector`). 월별로 받으면 적용일이 한 달 단위로 뭉개집니다.
    `data/raw/base_rate.csv`가 없으면 `build_dataset.py`가 월별 표에서 변경 월을 찾아 그 달 1일을
    적용일로 씁니다. 금통위는 월 중간에 열리므로 최대 3주 앞당겨집니다.
- **금리 10종:** 기준금리 외에 주담대(전체·고정·변동), 집단대출, 전세자금대출, 가계대출,
  국고채 3년·10년, CD 91일을 받습니다 (`collect.read_rate_series` → `merge.attach_rate_series`).
  각각 수준과 12개월·3개월 변화량이 붙어 **30칸**이 됩니다.
  - 대출금리는 월 가중평균이 **다음 달에 공개**되므로 한 달 밀려서 붙입니다
    (`merge.RATE_SERIES_LAG_MONTHS`). 계약 월 값을 그대로 쓰면 그 시점에 없던 정보가 들어갑니다.
  - 기본 입력 변수에는 변화량만 넣고 수준은 넣지 않습니다. 단, `base_rate` 수준은 금리 시나리오
    (`scenarios.interest_rate`)가 그 컬럼을 요구하므로 예외로 남겨 두었습니다.
- **전국 공통 규제:** DSR 단계(0~3), 스트레스 DSR(0~2), 주담대 한도(억, 99 = 한도 없음),
  양도세 중과, 정책대출, 만기연장 제한. 자치구별로 갈리지 않으므로 기준금리처럼 **변경일 기준**으로
  붙입니다 (`collect.to_national_regulation_state` → `merge.attach_national_regulation`).
  월 단위로 붙이면 10·15 대책이 10월 1~14일 계약에도 붙어 미래 정보가 섞입니다.
- **규제 점수(`reg_score`, `reg_score_change`):** 그 시점에 걸려 있는 규제의 가중합입니다.
  가중치는 `schemas.REGULATION_WEIGHTS`에 있고 **우리가 정한 값이라 팀 합의가 필요합니다.**
  누적 합계로 만들면 값이 한 방향으로만 움직여 모델이 경과 시간을 학습합니다.
- **규제 이력 테이블:** `district`, `regulation_type`, `announced_date`(선택), `effective_from`, `effective_to`(선택)
  - `effective_from <= contract_date < effective_to`이면 1, 아니면 0입니다. `effective_to`가 비어 있으면 현재까지 유효한 것으로 봅니다.
  - `announced_date`는 기록용이며, 적용 여부 판정에는 쓰지 않습니다.
- 계약일 이후에 공개된 정보는 붙이지 않습니다.

## 5. 결측값 정책
| 단계 | 정책 |
|---|---|
| 필수 컬럼 결측·변환 실패 | `invalid="raise"`(기본): 오류 / `invalid="drop"`: 해당 행 제거 후 리포트에 기록 |
| 선택 컬럼 결측 | 그대로 유지(null). 컬럼 자체가 없으면 null 컬럼을 생성 (`ensure_optional_columns`) |
| 모델 입력 결측 | Pipeline 안의 `SimpleImputer`로 대체 (수치형은 중앙값, 범주형은 최빈값). **학습 데이터에만 fit** |
| `anomaly_label` 결측 | 허용. 결측 행은 분류 학습에서 쓰지 않으며, 전부 결측이면 학습이 차단됨 |

## 6. 중복 거래 기준
`district, contract_date, area_m2, floor, price_krw` (+ 컬럼이 있으면 `dong`, `apartment_name`, `complex_id`)가
모두 같으면 중복으로 보고 첫 행만 남깁니다.

`complex_id`(단지 일련번호)를 키에 넣으면 같은 이름의 다른 단지를 구분할 수 있습니다. 실제 수집 데이터에서
단지명이 같고 일련번호가 다른 단지가 **511개** 있었습니다.

같은 단지·같은 날·같은 층에서 같은 가격으로 실제 거래가 2건 있었을 가능성은 여전히 배제하지 못합니다.
동(棟) 정보(`aptDong`)로 더 좁힐 수 있지만, 이 값은 등기가 끝난 거래에만 붙어 있어 입력으로 쓸 수 없습니다.

## 7. 전처리 규칙과 순서 (`preprocess.preprocess_transactions`)
1. 필수 컬럼 검증
2. 타입 변환: 날짜 표기 통일(`YYYY-MM-DD`, `YYYYMMDD`, `YYYY.MM.DD`, `YYYY/M/D`), 숫자에서 쉼표 제거
3. 값 유효성 검사 (위반 규칙별 건수를 리포트에 기록)
4. 중복 제거
5. 선택 컬럼 보완
6. 날짜 파생변수 생성
7. 계약일 오름차순 정렬

1~7단계는 모두 **행 단위** 변환입니다. 데이터에서 통계를 학습하는 변환(대체, 스케일링, 인코딩)은 분할 이후 모델 Pipeline 안에서만 수행합니다.

## 8. 데이터 출처와 저장
| 데이터 | 출처 | 저장 위치 |
|---|---|---|
| 아파트 매매 실거래가 | 국토교통부 공공데이터포털 OpenAPI (상세 자료) | `data/raw/<코드>_<자치구>/<YYYYMM>.json` → `data/processed/` |
| 기준금리 | 한국은행 ECOS (일별, 변경일만 추림) | `data/raw/base_rate.csv` |
| 부동산 규제 이력 | 국토교통부·서울시·금융위원회 고시 (수작업 정리) | **`configs/regulation_events.csv`** (커밋함) |
| 합성 샘플 | `scripts/generate_sample_data.py` | `data/sample/*_TEST_ONLY.csv` (**가짜 데이터**) |

`data/raw`와 `data/processed`는 Git에 커밋하지 않습니다 (원본 약 190MB, 가공 약 230MB).
원본은 팀 공유 드라이브로 전달합니다.

규제 이력만 예외적으로 `configs/`에 두고 커밋합니다. 공개 API가 없어 고시를 직접 찾아 **사람이 작성한 입력**이고,
2.5KB에 불과하며, 판정 기준이 바뀌면 변경 이력이 남아야 하기 때문입니다. 수집한 원본이 아니므로 `data/raw`에
두지 않습니다.

수집 실행:
```bash
export MOLIT_API_KEY=<공공데이터포털 디코딩 키>
export ECOS_API_KEY=<한국은행 ECOS 인증키>
python scripts/collect_molit.py        # 서울 25개 구 x 68개월, 약 1,700회 호출
python scripts/collect_rates.py        # 금리 10종 (월별)
python scripts/collect_base_rate.py    # 기준금리 (일별) — 변경 적용일을 정확히 쓰려면
```
이미 받은 달·항목은 건너뛰므로 중단 후 다시 실행하면 없는 것만 받습니다.

학습 데이터 만들기 (네트워크 불필요, `data/raw`만 읽습니다):
```bash
python scripts/build_dataset.py            # 자치구별 CSV + training_dataset.csv
python scripts/build_dataset.py --label     # 이상거래 레이블까지 (팀 합의 전 기본 꺼짐)
python scripts/build_dataset.py --keep-cancelled   # 해제 거래 유지 (레이블 검증용)
```
자치구별 CSV가 원본보다 새것이면 건너뛰므로 중단 후 다시 실행하면 남은 구만 읽습니다.
결과는 **257,655행 × 66칸** 입니다 (`--label` 기준).

| 단계 | 하는 일 |
|---|---|
| 1 | 자치구별 월별 JSON → 정규화 → `data/processed/transactions_<코드>_<자치구>.csv` |
| 2~3 | 합치기 → 타입 변환·값 검증·중복 제거 (`preprocess_transactions`) |
| 4~5 | 거래 조건(직거래·중개사 위치·토지임대부) → 기준가 4단계 |
| 6 | 기준금리(변경일 기준) + 금리 9종(계약 월, 한 달 밀림) + 변화량 |
| 7 | 자치구별 규제 + 전국 공통 규제 + 규제 점수 |
| 8 | (선택) 이상거래 레이블 |
| 9~10 | 계약 시점에 없는 원본 칸 22개 제거 → 저장 |

## 9. 기준가와 이상거래 레이블

**기준가 (`peer_median_price_krw`)** 는 §2의 `reference_price_krw`와 다른 값입니다. 회귀 모델의 예상가가 아니라
**직전 기간에 실제로 거래된 같은 조건의 시세**입니다. `preprocess.add_peer_price`가 만듭니다.

좁은 그룹부터 보고 표본이 모자라면 넓은 그룹으로 올라가며, 몇 단계에서 만들었는지를 `peer_tier`에 남깁니다.

| 단계 | 그룹 | 최소 표본 | 실제 데이터 비중 | 시세비율 사분위폭 |
|---|---|---|---|---|
| 1 | 단지 + 평형대 | 3건 | 45.1% | 0.077 |
| 2 | 단지 | 3건 | 19.4% | 0.226 |
| 3 | 법정동 + 평형대 | 5건 | 18.3% | 0.313 |
| 4 | 자치구 + 평형대 | 5건 | 9.5% | 0.454 |
| 0 | 만들 수 없음 | - | 7.6% | - |

**㎡당 가격으로 기준을 만들고 면적을 다시 곱합니다.** 가격 중앙값을 그대로 쓰면 같은 단지의 59㎡와 115㎡가
서로 기준이 되어, 작은 평형은 전부 싸게 큰 평형은 전부 비싸게 잡힙니다. 면적 보정 전 이상거래 비율이
21.0%였고 보정 후 4.6%가 되었습니다.

직전 기간만 보기 때문에 미래 정보가 들어가지 않습니다. 학습형 전처리가 아니라 시점 기준 집계이므로
분할 이전에 계산해도 누출이 생기지 않습니다.

**이상거래 레이블 (`anomaly_label`)** 은 기준가에서 많이 벗어난 거래에 붙입니다 (`preprocess.label_anomalies`).

- 기본 임계값: 1단계 ±20%, 2단계 ±30%. **3·4단계는 레이블을 달지 않습니다.**
- 넓은 단계일수록 기준 자체가 흔들립니다. 자치구 단위(4단계)로 판정하면 이상으로 뽑힌 거래의 직거래 비율이
  그 자치구 평균과 같아져(배수 1.0) 신호가 아니라 잡음이었습니다.
- **정의는 머신러닝 담당자와 합의해 확정합니다.** `label_anomalies`를 호출하지 않으면 `anomaly_label`은
  null로 남고, `configs/settings.yaml`의 `classification.training_enabled`도 `false`입니다.

레이블이 타당한지는 **밖에서 주어진 사실**로 확인합니다 (`docs/DATA_QUALITY.md`).

## 10. 미해결 사항
- **규제 적용 단위:** 토지거래허가구역은 실제로 동·단지 단위로 지정됩니다(2025-02-13 해제는 잠실·삼성·대치·청담
  네 개 동이었습니다). 현재는 자치구 단위로 근사하고 있어, 같은 구 안에서 지정된 동과 그렇지 않은 동이 섞입니다.
- **전국 공통 규제:** DSR 단계, 주담대 한도, 양도세 중과 등은 자치구별 값이 없어 현재 규제 컬럼으로 표현할 수
  없습니다. `configs/regulation_events.csv`에 기록만 해두었고 계산에서는 빠집니다. 전국 공통 규제를 시점 변수로
  넣으려면 스키마 확장이 필요합니다.
- **규제 이력이 없는 기간:** 현재는 0으로 처리합니다.
- **해제 신고 지연:** 해제는 계약 후 한참 뒤에 일어납니다. 실제 데이터에서 1개월 내 46.6%, 3개월 내 83.9%,
  6개월 내 96.5%였고 최장 1,845일이었습니다. 따라서 최근 2~3개월 거래는 해제 여부가 아직 다 드러나지
  않았습니다. 레이블 검증에 쓸 때는 그 구간을 빼야 합니다.
- **거래유형 제공 시점:** `dealingGbn`(중개/직거래)은 2021년 자료에 94%가 비어 있습니다. 이 변수를 쓰는
  모델은 2022년 이후로 기간을 한정해야 합니다.
