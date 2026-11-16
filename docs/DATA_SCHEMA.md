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

### 파생 컬럼 (행 단위 계산이므로 데이터 누출 없음)
| 필드 | 계산식 | 사용 가능 모델 |
|---|---|---|
| `contract_year`, `contract_month` | 계약일의 연도·월 | 회귀·분류 |
| `building_age` | `contract_year - built_year` | 회귀·분류 |
| `price_per_m2` | `price_krw / area_m2` | **분류만** (정답값에서 나온 값이므로 회귀 입력 금지) |

## 2. 입력 변수와 정답값의 구분

| 구분 | 가격 회귀 (기능 A·C) | 이상거래 분류 (기능 B) |
|---|---|---|
| 정답값 | `price_krw` | `anomaly_label` |
| 입력 변수 | `REGRESSION_FEATURES` | `CLASSIFICATION_FEATURES` (= 회귀 입력 + `price_krw`, `price_per_m2`) |
| 입력 금지 | `price_krw`, `price_per_m2`, `anomaly_label`, `reference_price_krw` | `anomaly_label` |

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
| `cdealType`, `cdealDay` | `해제사유발생일` | 해제된 거래는 **제거** |

> 실제로 수집할 때 원본 필드명을 다시 확인해야 합니다. 필드명이 바뀌면 매핑 dict만 수정하면 됩니다.
> OpenAPI 정규화는 `collect.normalize_molit_api_frame`이 담당합니다. CSV 정규화 함수는 feature/data에서 구현합니다.

## 4. 시점 정합성 (기준금리·규제)
- **기준금리 테이블:** `effective_date`, `base_rate`
  - `effective_date`는 *변경 금리가 적용되기 시작한 날*입니다. 금통위 발표일과 다를 수 있습니다.
  - 각 거래에는 `effective_date <= contract_date`를 만족하는 가장 최근 값을 붙입니다 (`merge.attach_base_rate`, `merge_asof backward`).
  - 첫 적용일보다 이른 계약은 NaN으로 둡니다.
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
`district, contract_date, area_m2, floor, price_krw` (+ 컬럼이 있으면 `dong`, `apartment_name`)가 모두 같으면 중복으로 보고 첫 행만 남깁니다.
같은 단지·같은 날·같은 층에서 같은 가격으로 실제 거래가 2건 있었을 가능성은 배제하지 못합니다. 원본에 동(棟)·거래 고유키가 있다면 키에 추가하는 것을 검토합니다.

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
| 아파트 매매 실거래가 | 국토교통부 (공공데이터포털 OpenAPI / rt.molit.go.kr) | `data/raw/molit/…` → `data/processed/` |
| 기준금리 | 한국은행 ECOS | `data/raw/ecos/…` |
| 부동산 규제 이력 | 국토교통부 보도자료 등 (수작업 정리 예정) | `data/raw/regulation/…` |
| 합성 샘플 | `scripts/generate_sample_data.py` | `data/sample/*_TEST_ONLY.csv` (**가짜 데이터**) |

`data/raw`와 `data/processed`는 Git에 커밋하지 않습니다.

## 9. 미해결 사항
- 규제 적용 단위: 토지거래허가구역처럼 동·단지 단위로 지정되는 규제는 현재의 자치구 단위 플래그로 정확히 표현할 수 없습니다. 이를 처리하려면 `dong` 단위의 규제 테이블로 확장해야 합니다.
- 규제 이력이 없는 기간을 0으로 볼지 null로 볼지: 현재는 0으로 처리합니다. 수집 범위가 확정되면 다시 검토합니다.
- 거래 해제(취소) 신고가 늦게 반영되는 문제: 수집 시점에 따라 해제 여부가 달라질 수 있습니다.
