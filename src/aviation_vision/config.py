from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@lru_cache(maxsize=8)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是映射：{path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_config(model_id: str) -> dict[str, Any]:
    models = load_yaml("models.yaml").get("models", {})
    if model_id not in models:
        raise KeyError(f"未知模型：{model_id}")
    return dict(models[model_id])


def mode_config(mode_id: str) -> dict[str, Any]:
    modes = load_yaml("models.yaml").get("modes", {})
    if mode_id not in modes:
        raise KeyError(f"未知推理模式：{mode_id}")
    return dict(modes[mode_id])

