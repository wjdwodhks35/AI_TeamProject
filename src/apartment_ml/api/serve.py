"""API 와 정적 UI 를 한 포트에서 함께 제공하는 실행 전용 앱.

``deploy.sh`` 가 이 모듈을 띄운다::

    uvicorn apartment_ml.api.serve:app --host 0.0.0.0 --port 8010

개발 중에는 ``apartment_ml.api.main:app`` 과 ``python -m http.server`` 를 따로 띄워도 된다.

``main.app`` 에 라우트를 추가하지 않고 정적 파일 mount 만 뒤에 붙인다.
Starlette 는 등록 순서대로 경로를 찾으므로, 나중에 등록한 ``/`` mount 보다
``/health``, ``/predict/*``, ``/scenario/*``, ``/docs``, ``/openapi.json`` 이 먼저 매칭된다.
UI 와 API 가 같은 출처가 되므로 이 구성에서는 CORS 설정이 필요하지 않다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from apartment_ml.api.main import app

__all__ = ["app", "UI_DIR"]

logger = logging.getLogger(__name__)


def _resolve_ui_dir() -> Path:
    """UI 디렉터리 경로. ``APARTMENT_ML_UI_DIR`` 로 덮어쓸 수 있다."""
    override = os.environ.get("APARTMENT_ML_UI_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # 이 파일: <저장소>/src/apartment_ml/api/serve.py  →  <저장소>/ui
    # editable 설치(`pip install -e .`)가 아니면 경로가 달라지므로 환경변수를 쓴다.
    return Path(__file__).resolve().parents[3] / "ui"


UI_DIR = _resolve_ui_dir()

if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    logger.info("정적 UI 제공: %s", UI_DIR)
else:
    logger.warning(
        "UI 디렉터리가 없어 정적 파일을 제공하지 않습니다 (API 는 그대로 동작): %s", UI_DIR
    )
