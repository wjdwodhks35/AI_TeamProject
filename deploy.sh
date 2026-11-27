#!/usr/bin/env bash
#
# 리눅스에서 API 와 UI 를 한 포트로 띄운다.
#
#   ./deploy.sh
#
# 접속:  http://<서버주소>:8010      (API 문서: /docs, 상태: /health)
#
# 환경변수
#   HOST=127.0.0.1   바인딩 주소. 기본값 0.0.0.0 (외부 접속 허용)
#   SKIP_INSTALL=1   의존성 설치 건너뛰기. 두 번째 실행부터 쓰면 빠르다
#   RELOAD=1         코드 변경 시 자동 재시작 (개발용)

set -Eeuo pipefail

PORT=8010                      # 고정
HOST="${HOST:-0.0.0.0}"

cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
VENV="$ROOT/.venv"

# ---------------------------------------------------------------- Python 찾기

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
        >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------- 1. 가상환경

if [[ -x "$VENV/bin/python" ]]; then
  echo "[1/3] 가상환경 확인: $VENV"
else
  if ! PYTHON="$(find_python)"; then
    echo "오류: Python 3.11 이상을 찾을 수 없습니다." >&2
    echo "      예) sudo apt install python3.11 python3.11-venv" >&2
    exit 1
  fi
  echo "[1/3] 가상환경 생성: $VENV  ($("$PYTHON" -V))"
  "$PYTHON" -m venv "$VENV"
fi

PY="$VENV/bin/python"

# ---------------------------------------------------------------- 2. 의존성

if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
  echo "[2/3] 의존성 설치 건너뜀 (SKIP_INSTALL=1)"
else
  echo "[2/3] 의존성 설치 중"
  "$PY" -m pip install --upgrade pip --quiet
  "$PY" -m pip install -e "$ROOT" --quiet
fi

# ---------------------------------------------------------------- 3. 서버 시작

# UI 와 API 가 같은 출처이므로 CORS 설정은 필요하지 않다.
export APARTMENT_ML_UI_DIR="$ROOT/ui"
export APARTMENT_ML_CONFIG="${APARTMENT_ML_CONFIG:-$ROOT/configs/settings.yaml}"

echo "[3/3] 모델 파일 확인"
# 경로는 configs/settings.yaml 의 models.* 와 같아야 한다.
for model in models/price_regression.joblib models/complex_directory.json \
  models/anomaly_model.joblib; do
  if [[ -f "$ROOT/$model" ]]; then
    echo "      $model — 있음"
  else
    echo "      $model — 없음 (해당 기능은 503 MODEL_NOT_READY)"
  fi
done

echo
echo "      http://127.0.0.1:$PORT"
if command -v hostname >/dev/null 2>&1; then
  for ip in $(hostname -I 2>/dev/null || true); do
    echo "      http://$ip:$PORT"
  done
fi
echo "      종료: Ctrl+C"
echo

UVICORN_ARGS=(--host "$HOST" --port "$PORT" --log-level info)
if [[ "${RELOAD:-0}" == "1" ]]; then
  UVICORN_ARGS+=(--reload)
fi

exec "$PY" -m uvicorn apartment_ml.api.serve:app "${UVICORN_ARGS[@]}"
