# 협업 가이드

## 브랜치 전략 (Git Flow 간소화)
| 브랜치 | 용도 | 규칙 |
|---|---|---|
| `main` | 검증된 최종 버전 | 직접 push 금지. `dev`에서 PR로만 반영 |
| `dev` | 개발 통합 브랜치 | 직접 push 금지. `feature/*`에서 PR로만 반영 |
| `feature/bootstrap` | 공통 개발 기반 (Phase 0) | |
| `feature/data` | 데이터 수집·전처리 | |
| `feature/ml` | 머신러닝 모델 | |
| `feature/scenario` | 금리 시나리오 | |
| `feature/ui` | 사용자 화면 및 API | |

- 기능을 더 잘게 나눌 때는 `feature/<영역>-<내용>` 형식을 씁니다. 예: `feature/data-molit-collector`
- 버그 수정은 `fix/<내용>`, 문서 수정은 `docs/<내용>` 형식을 씁니다.
- 작업 시작 전: `git checkout dev && git pull && git checkout -b feature/...`
- 작업이 길어지면 `dev` 변경을 자주 merge 또는 rebase해서 충돌을 줄입니다.

## 담당별 수정 파일
| 담당 | 주로 수정 | 수정 시 다른 담당자와 합의 |
|---|---|---|
| 데이터 | `data/collect.py`, `data/merge.py`, `data/preprocess.py`, `docs/DATA_SCHEMA.md` | `schemas.py` |
| ML | `models/*`, `evaluation/*`, `docs/MODEL_PLAN.md`, 학습 스크립트(`scripts/train_*.py`) | `schemas.py`의 feature 목록 |
| 시나리오 | `scenarios/*` | API 응답 필드 |
| UI/API | `api/*`, UI 디렉터리(신규), `docs/API_CONTRACT.md` | 요청·응답 스키마 |

## 개발 규칙
- 컬럼명은 문자열로 직접 쓰지 말고 `schemas`의 상수를 사용합니다.
- 회귀 입력 변수에 `price_krw`와 그 파생값을 넣지 않습니다. 모델 래퍼가 자동으로 차단합니다.
- 학습형 전처리(대체, 스케일링, 인코딩)는 Pipeline 안에 두고, 분할 이전에 전체 데이터로 fit하지 않습니다.
- 셔플 분할이나 KFold를 쓰지 않고, `evaluation.split`의 함수를 사용합니다.
- API 키나 실제 원본 데이터, 모델 파일(`*.joblib`)은 커밋하지 않습니다.
- 성능 수치는 실제 데이터와 정해진 분할로 얻은 결과만 기록합니다. 합성 샘플 데이터로 얻은 수치는 금지합니다.
- 테스트에서 외부 네트워크를 호출하지 않습니다. `tests/conftest.py`가 소켓 연결을 차단합니다.
- 테스트용 가짜 모델은 `tests/fakes.py`에만 둡니다. `src/`에서는 import하지 않습니다.

## 테스트 기준
- 새 공개 함수에는 정상 케이스 1개 이상과 오류 케이스 1개 이상의 테스트를 작성합니다.
- PR을 올리기 전에 로컬에서 다음 명령을 실행합니다.
  ```bash
  ruff check . && ruff format --check . && pytest -q
  ```
- CI(`.github/workflows/ci.yml`)가 통과해야 merge할 수 있습니다.

## 커밋 메시지
`<type>(<scope>): <요약>` 형식을 씁니다. type은 `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci` 중 하나입니다.
예: `feat(data): MOLIT CSV 정규화 함수 추가`

## PR 작성
- 대상 브랜치: `feature/*` → `dev`, 릴리스 시 `dev` → `main`
- `.github/pull_request_template.md` 양식을 채웁니다 (개요, 변경 사항, 영향 범위, 검증).
- 리뷰어를 1명 이상 지정합니다. `schemas.py`나 API 계약을 바꾸는 PR은 관련 담당자 전원이 리뷰합니다.
- PR은 작게 나눕니다. 400줄 이내를 권장합니다 (생성된 데이터는 제외).

## 코드 리뷰 기준
1. 데이터 누출이 없는가: 정답값이 입력에 들어가지 않았는지, 분할 전에 fit하지 않았는지, 미래 정보를 쓰지 않았는지
2. 공통 계약을 지키는가: `schemas` 상수 사용, 관련 문서 동시 갱신
3. 실패를 명확하게 처리하는가: 가짜 값으로 대신 응답하지 않는지, 오류 메시지가 이해하기 쉬운지
4. 테스트가 있고 CI가 통과하는가
5. 범위가 PR 주제에 맞는가: 다른 담당 영역을 무단으로 수정하지 않았는지
