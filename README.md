# AI_TeamProject
2학년 2학기 인공지능 기말 팀프로젝트

## 집 가고 싶다

---

# 서울 아파트 가격 예측 및 이상거래 의심 탐지 시스템

머신러닝으로 서울 25개 자치구 아파트의 거래가격을 예측하고, 이상거래 의심 여부를 판정하며,
기준금리 조건에 따른 예측가격 변화를 비교하는 시스템입니다.

> **현재 단계: Phase 0 — 공통 개발 기반.**
> 데이터 계약, 전처리, 시간순 분할, 평가 지표, 모델/시나리오 인터페이스, API 계약, 테스트, CI까지 갖춰져 있습니다.
> 실제 데이터 수집, 모델 학습, UI 개발은 각 담당자가 `feature/*` 브랜치에서 진행합니다.
> 저장소에 있는 샘플 데이터는 **합성(가짜) 데이터**이므로, 이 데이터로 얻은 수치를 성능으로 해석하면 안 됩니다.

| 기능 | 내용 | Phase 0 상태 |
|---|---|---|
| A. 가격 예측 | 회귀 (LinearRegression 기준, RandomForest 비교) | 인터페이스·학습 래퍼 완료, 실제 학습은 하지 않음 |
| B. 이상거래 의심 탐지 | DecisionTree / RandomForest / Boosting / Voting | 인터페이스만 정의 (레이블 미확보로 학습 차단) |
| C. 금리 시나리오 | 금리만 바꿨을 때 예측가격 비교 | 계산 인터페이스 완료, 학습 모델이 없으면 503 |
| D. 웹 UI | 입력 → 결과 화면 | API 계약만 제공 |
| E. 12개월 예측 | 미래 가격 추이 | 설계 과제로 문서화 ([MODEL_PLAN](docs/MODEL_PLAN.md)) |

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
├── models/                      # 학습된 모델 파일 *.joblib (Git 제외)
├── scripts/generate_sample_data.py
├── src/apartment_ml/
│   ├── schemas.py               # ★ 공통 데이터 계약 (컬럼명, 유효 범위, 입력 변수/정답값 구분)
│   ├── config.py                # 설정 및 API 키 로딩
│   ├── pipeline.py              # 최소 실행 경로 (smoke pipeline)
│   ├── data/      collect.py · merge.py · preprocess.py
│   ├── evaluation/ split.py · metrics.py
│   ├── models/    base.py · regression.py · classification.py · ensemble.py
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
git checkout dev

python3.11 -m venv .venv          # 3.11 이상
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"           # Parquet이 필요하면 ".[dev,parquet]"

cp .env.example .env              # 실제 API 키는 .env 에만 넣습니다 (커밋 금지)
```
`.env`는 자동으로 로드되지 않습니다. 수집 스크립트를 실행하기 전에 환경변수로 내보내세요.
(macOS/Linux: `set -a; source .env; set +a`)

## 실행 방법
```bash
# 1) 최소 실행 경로: 샘플 로딩 → 검증/전처리 → 금리·규제 병합 → 시간순 분할 요약
python -m apartment_ml.pipeline

# 2) API 서버 → http://127.0.0.1:8000/docs
uvicorn apartment_ml.api.main:app --reload
curl http://127.0.0.1:8000/health
```
학습된 모델 파일(`models/price_model.joblib` 등)이 없으면, 예측 엔드포인트는 **503 `MODEL_NOT_READY`** 를 반환합니다.

## 테스트 및 정적 검사
```bash
ruff check .
ruff format --check .
pytest -q
```
CI(GitHub Actions)는 Python 3.11/3.12/3.13에서 위 명령과 smoke pipeline을 실행합니다. API 키는 필요 없습니다.

## 문서
- [ARCHITECTURE](docs/ARCHITECTURE.md): 데이터 흐름, 모듈 의존 관계
- [DATA_SCHEMA](docs/DATA_SCHEMA.md): 공통 필드, 단위, 결측·중복 처리, 시점 정합성
- [API_CONTRACT](docs/API_CONTRACT.md): 엔드포인트, 요청/응답, 오류 코드
- [MODEL_PLAN](docs/MODEL_PLAN.md): 모델 후보, 평가, 시간순 검증, 미해결 과제
- [CONTRIBUTING](docs/CONTRIBUTING.md): 브랜치 전략, PR, 리뷰 기준
