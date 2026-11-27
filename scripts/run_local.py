"""로컬에서 API 와 UI 를 한 포트로 띄운다 (윈도우 deploy.bat 이 호출).

배치 파일에 로직을 두면 한글 출력 때문에 코드페이지를 바꿔야 하고, 그러면 cmd 가
라벨(goto)을 다시 찾을 때 깨진다. 그래서 배치는 가상환경만 만들고 나머지는 여기서 한다.

리눅스 서버 배포는 deploy.sh 를 쓴다. 이 스크립트는 로컬 확인용이라 localhost 에만
바인딩한다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8010

# (경로, 없을 때 멈추는 기능)
MODEL_FILES = (
    ("models/price_regression.joblib", "가격 예측"),
    ("models/complex_directory.json", "단지 선택 목록"),
    ("models/anomaly_model.joblib", "이상거래 판정"),
)


def run(*args: str) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="로컬에서 API+UI 를 띄웁니다.")
    parser.add_argument("--skip-install", action="store_true", help="의존성 설치 건너뛰기")
    parser.add_argument("--reload", action="store_true", help="파이썬 코드 변경 시 자동 재시작")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.skip_install or os.environ.get("SKIP_INSTALL") == "1":
        print("[1/2] 의존성 설치 건너뜀")
    else:
        print("[1/2] 의존성 설치 중 (처음 한 번은 몇 분 걸립니다)")
        if run("-m", "pip", "install", "--upgrade", "pip", "--quiet") != 0:
            print("pip 업그레이드 실패 — 계속 진행합니다.")
        if run("-m", "pip", "install", "-e", str(ROOT), "--quiet") != 0:
            print("\n의존성 설치에 실패했습니다. 위 메시지를 확인하세요.")
            return 1

    print("[2/2] 모델 파일 확인")
    for name, feature in MODEL_FILES:
        state = "있음" if (ROOT / name).exists() else f"없음 ({feature} 기능은 503)"
        print(f"      {name} - {state}")

    # UI 와 API 가 같은 출처라 CORS 설정이 필요 없다.
    env = dict(os.environ)
    env["APARTMENT_ML_UI_DIR"] = str(ROOT / "ui")
    env.setdefault("APARTMENT_ML_CONFIG", str(ROOT / "configs" / "settings.yaml"))

    url = f"http://localhost:{args.port}"
    print(f"\n      {url}")
    print(f"      API 문서: {url}/docs")
    print("      UI 를 고치면 브라우저 새로고침만 하면 반영됩니다.")
    print("      종료: Ctrl+C\n")

    cmd = [
        sys.executable, "-m", "uvicorn", "apartment_ml.api.serve:app",
        "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "info",
    ]
    if args.reload or os.environ.get("RELOAD") == "1":
        cmd.append("--reload")

    try:
        return subprocess.call(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
