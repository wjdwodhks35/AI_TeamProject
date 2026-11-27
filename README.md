# 서울 아파트 실거래가 추정

한국공학대학교 2학년 2학기 인공지능 기말 팀프로젝트.

국토교통부 실거래가로 서울 25개 자치구 아파트의 거래가격을 추정하고, 이상거래 의심
여부를 판정하며, 금리 조건을 바꿔 보고, 향후 가격을 내다봅니다. 학습부터 웹 화면까지
한 저장소에 있고 `deploy.bat` / `deploy.sh` 한 번으로 전부 뜹니다.

## 결과 요약

**가격 예측은 기준선을 크게 넘었고, 이상거래 분류는 넘지 못했습니다.** 두 결과를
나란히 두는 것이 이 프로젝트의 결론입니다.

| 기능 | 기준선 (학습 없음) | 우리 모델 |
|---|---|---|
| A. 가격 예측 | 자치구+평형 중앙값 — MAE 3.59억, R² 0.603 | **MAE 1.12억, R² 0.952, MAPE 9.3%** |
| B. 이상거래 의심 | 규칙 두 줄 — **F1 0.429** | 분류 모델 F1 0.194 |

가격 회귀가 기준선을 넘은 것은 구조를 세 번 바꾼 결과입니다 — 절대가격 대신 직전월
시세 대비 상대가격, 위치를 단지 단위까지, 부스팅으로 교체. 이상거래 분류가 넘지 못한
것은 레이블이 `가격 < 기준가 × 0.8` 인데 분류 입력에 기준가를 넣을 수 없기 때문입니다
(넣으면 레이블 공식을 되계산하게 됩니다). 자세한 측정은
[MODEL_PLAN](docs/MODEL_PLAN.md) 에 있습니다.

| 기능 | 내용 | 상태 |
|---|---|---|
| A. 가격 예측 | HistGradientBoosting, 시세 대비 상대가격 | 완료 · `POST /predict/price` |
| B. 이상거래 의심 | 규칙 3단계 + 분류 모델 4종 비교 | 완료 · `POST /predict/anomaly` |
| C. 금리 시나리오 | 금리만 바꿨을 때 예측가격 비교 | 완료 · `POST /scenario/interest-rate` |
| D. 웹 UI | 지도에서 자치구 → 동 → 단지 선택, 결과 4탭 | 완료 · `ui/` |
| E. 향후 N개월 예측 | 시장지수를 내다보고 80% 구간까지 | 완료 · `POST /forecast/price` |

학습된 모델 파일은 `models/` 에 **같이 올려 두었습니다**(압축해서 12MB). 받자마자
실행할 수 있습니다. `data/raw` 와 `data/processed` 는 올리지 않습니다.

## 기술 스택
Python 3.11+, pandas, NumPy, scikit-learn, FastAPI, Pydantic v2, Uvicorn, pytest, Ruff, GitHub Actions.
데이터는 CSV/Parquet 파일로 저장하며, 별도의 DB는 쓰지 않습니다.

## 폴더 구조
```text
.
├── configs/settings.yaml        # 공통 설정 (경로, 분할 비율, 모델 경로, 분류 학습 스위치)
├── data/
│   ├── raw/                     # 원본 수집 데이터 (Git 제외)
│   ├── processed/               # 정규화·전처리 결과 (Git 제외)
│   └── sample/                  # 합성 테스트 데이터 (*_TEST_ONLY.csv)
├── models/                      # 학습된 모델 (압축해서 저장소에 포함, 12MB)
│   ├── price_regression.joblib      #   가격 회귀 + 시장지수 + 보정계수
│   ├── complex_directory.json       #   자치구 → 동 → 단지 목록 (7,868개)
│   ├── anomaly_model.joblib         #   이상거래 분류
│   └── anomaly_model_metrics.json   #   모델별 지표와 규칙 기준선
├── ui/                          # 정적 웹 화면 (빌드 도구 없음)
├── deploy.bat / deploy.sh       # API + UI 를 한 포트(8010)로 실행
├── scripts/                     # generate_sample_data · build_dataset · train_price · train_anomaly
├── src/apartment_ml/
│   ├── schemas.py               # ★ 공통 데이터 계약 (컬럼명, 유효 범위, 입력 변수/정답값 구분)
│   ├── config.py                # 설정 및 API 키 로딩
│   ├── pipeline.py              # 최소 실행 경로 (smoke pipeline)
│   ├── data/      collect.py · merge.py · preprocess.py
│   ├── evaluation/ split.py · metrics.py
│   ├── models/    base.py · regression.py · classification.py · ensemble.py
│   │               market_index.py · location_price.py · complex_directory.py
│   │               forecast.py · anomaly_rules.py
│   ├── scenarios/ interest_rate.py
│   └── api/       main.py · schemas.py
├── tests/                       # pytest (외부 네트워크 차단)
├── docs/                        # ARCHITECTURE / DATA_SCHEMA / API_CONTRACT / MODEL_PLAN / CONTRIBUTING
└── .github/workflows/ci.yml
```

## 개발 환경 준비
```bash
git clone https://github.com/wjdwodhks35/AI_TeamProject.git
cd AI_TeamProject

python3.11 -m venv .venv          # 3.11 이상
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"           # Parquet이 필요하면 ".[dev,parquet]"

cp .env.example .env              # 실제 API 키는 .env 에만 넣습니다 (커밋 금지)
```
`.env`는 자동으로 로드되지 않습니다. 수집 스크립트를 실행하기 전에 환경변수로 내보내세요.
(macOS/Linux: `set -a; source .env; set +a`)

## 실행 방법

### 화면까지 한 번에 (권장)

```
deploy.bat          # 윈도우 — 더블클릭해도 됩니다
./deploy.sh         # 리눅스
```

API 와 웹 화면이 <http://localhost:8010> 한 포트에 뜹니다 (API 문서는 `/docs`).
처음 한 번은 가상환경을 만드느라 몇 분 걸리고, 그 다음부터는 바로 뜹니다. 화면 파일을
고치면 브라우저 새로고침만 하면 반영됩니다. 자세한 사용법은 [ui/README](ui/README.md).

### 다시 학습하기

`data/processed/training_dataset.csv` 가 있어야 합니다 (원본 데이터는 저장소에 없습니다).

```bash
python scripts/train_price.py          # 가격 회귀 + 단지 목록
python scripts/train_anomaly.py --label-source "<레이블 기준과 확정 날짜>"
```

모델 파일을 다시 커밋하면 히스토리에 그만큼 쌓입니다. 자주 올리지 마세요.

## 테스트 및 정적 검사
```bash
ruff check .
ruff format --check .
pytest -q
```
CI(GitHub Actions)는 Python 3.11/3.12/3.13에서 위 명령과 smoke pipeline을 실행합니다. API 키는 필요 없습니다.

## 문서
- [BACKEND_GUIDE](docs/BACKEND_GUIDE.md): 백엔드 실행, 모델 계약, 프론트엔드 인수인계
- [BACKEND_REPORT](docs/BACKEND_REPORT.md): 구현 범위와 검증 결과
- [ARCHITECTURE](docs/ARCHITECTURE.md): 데이터 흐름, 모듈 의존 관계
- [DATA_SCHEMA](docs/DATA_SCHEMA.md): 공통 필드, 단위, 결측·중복 처리, 시점 정합성
- [API_CONTRACT](docs/API_CONTRACT.md): 엔드포인트, 요청/응답, 오류 코드
- [MODEL_PLAN](docs/MODEL_PLAN.md): 모델 후보, 평가, 시간순 검증, 미해결 과제
- [CONTRIBUTING](docs/CONTRIBUTING.md): 브랜치 전략, PR, 리뷰 기준
