# ⚠️ 합성(가짜) 샘플 데이터 — 테스트 전용

이 폴더의 `*_TEST_ONLY.csv` 파일은 `scripts/generate_sample_data.py` 가 난수로 만든 **가짜 데이터**입니다.

- 실제 국토교통부 실거래, 실제 한국은행 기준금리, 실제 부동산 정책과 **무관**합니다.
- 용도: 스키마 검증, 전처리·분할·API 테스트, 최소 실행 경로 확인.
- 이 데이터로 학습한 모델의 성능 수치를 보고서·발표에 사용하지 마세요.
- 모든 거래 행의 `source` 컬럼 값은 `synthetic_sample` 입니다.
- 전처리 검증을 위해 **중복 거래 1건**과 **음수 가격 1건**이 의도적으로 포함되어 있습니다.
- `anomaly_label` 은 전부 비어 있습니다 (레이블 미확보 상태를 재현).

| 파일 | 내용 |
|---|---|
| `sample_transactions_TEST_ONLY.csv` | 합성 거래 122행 (정상 120 + 중복 1 + 오류 1) |
| `sample_base_rates_TEST_ONLY.csv` | 가짜 기준금리 변경 이력 (`effective_date`, `base_rate`) |
| `sample_regulations_TEST_ONLY.csv` | 가짜 규제 이력 (`district`, `regulation_type`, `announced_date`, `effective_from`, `effective_to`) |

재생성: `python scripts/generate_sample_data.py` (고정 시드 → 동일 결과)
