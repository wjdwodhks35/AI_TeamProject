"""설정 로딩.

- 비밀값(API 키)은 환경변수에서만 읽는다. (``.env.example`` 참고, ``.env`` 는 커밋 금지)
- 그 외 설정은 ``configs/settings.yaml`` 에서 읽는다.
  다른 파일을 쓰려면 환경변수 ``APARTMENT_ML_CONFIG`` 로 경로를 지정한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
CONFIG_ENV_VAR = "APARTMENT_ML_CONFIG"


def get_cors_origins() -> list[str]:
    """쉼표로 구분한 개발 UI 출처 설정."""
    value = os.environ.get(
        "APARTMENT_ML_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    return [origin.strip() for origin in value.split(",") if origin.strip()]


class ConfigError(RuntimeError):
    """설정 파일 또는 환경변수가 올바르지 않을 때."""


@dataclass(frozen=True)
class SplitSettings:
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    min_rows_per_split: int = 1


@dataclass(frozen=True)
class Settings:
    project_name: str
    version: str
    analysis_start: date
    analysis_end: date
    paths: dict[str, Path]
    split: SplitSettings
    random_seed: int
    price_model_path: Path
    anomaly_model_path: Path
    classification_training_enabled: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def path(self, key: str) -> Path:
        try:
            return self.paths[key]
        except KeyError as exc:
            raise ConfigError(f"settings.paths 에 '{key}' 항목이 없습니다.") from exc


def _resolve(p: str | os.PathLike[str]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    path = _resolve(config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        split = SplitSettings(**raw.get("split", {}))
        total = split.train_ratio + split.valid_ratio + split.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"split 비율의 합이 1 이어야 합니다 (현재 {total}).")
        period = raw["analysis_period"]
        models = raw.get("models", {})
        return Settings(
            project_name=raw["project"]["name"],
            version=str(raw["project"].get("version", "0.0.0")),
            analysis_start=date.fromisoformat(str(period["start"])),
            analysis_end=date.fromisoformat(str(period["end"])),
            paths={k: _resolve(v) for k, v in raw.get("paths", {}).items()},
            split=split,
            random_seed=int(raw.get("random_seed", 42)),
            price_model_path=_resolve(models.get("price_model_path", "models/price_model.joblib")),
            anomaly_model_path=_resolve(
                models.get("anomaly_model_path", "models/anomaly_model.joblib")
            ),
            classification_training_enabled=bool(
                raw.get("classification", {}).get("training_enabled", False)
            ),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"설정 파일 형식 오류 ({path}): {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 단위로 캐시된 기본 설정."""
    return load_settings()


def get_api_key(env_name: str) -> str:
    """외부 API 키를 환경변수에서 읽는다. 없으면 명확한 오류를 낸다.

    테스트/CI 에서는 호출되지 않아야 한다 (외부 네트워크 미사용 원칙).
    """
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ConfigError(
            f"환경변수 {env_name} 가 설정되지 않았습니다. .env.example 을 참고해 .env 를 만드세요."
        )
    return value
