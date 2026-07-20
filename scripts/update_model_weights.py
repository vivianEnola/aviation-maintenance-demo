from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
EXPECTED = {
    "scene_classifier": (
        "classify",
        ["airport_overview", "cloud_scene", "other", "runway_detail"],
    ),
    "cloud_detector": (
        "detect",
        ["cumulonimbus", "stratocumulus", "typhoon_vortex"],
    ),
    "airport_detector": (
        "detect",
        ["airport", "large_building_cluster", "construction_area"],
    ),
    "runway_segmenter": ("segment", ["runway", "snow", "standing_water"]),
}


def ordered_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    raise ValueError(f"无法读取模型类别：{names!r}")


def validate_bundle(source: Path) -> dict[str, Path]:
    validated: dict[str, Path] = {}
    for model_id, (expected_task, expected_names) in EXPECTED.items():
        weight = source / model_id / "best.pt"
        if not weight.is_file():
            raise FileNotFoundError(f"缺少权重：{weight}")
        model = YOLO(str(weight))
        names = ordered_names(model.names)
        if model.task != expected_task:
            raise ValueError(
                f"{model_id} 任务应为 {expected_task}，实际为 {model.task}。"
            )
        if names != expected_names:
            raise ValueError(
                f"{model_id} 类别不匹配：应为 {expected_names}，实际为 {names}。"
            )
        validated[model_id] = weight
        print(f"验证通过：{model_id} task={model.task} classes={names}")
    return validated


def install(source: Path, weights: dict[str, Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = PROJECT_ROOT / ".model_backups" / timestamp
    for model_id, source_weight in weights.items():
        destination = MODEL_ROOT / model_id / "best.pt"
        backup = backup_root / model_id / "best.pt"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_weight, destination)

    metrics = source / "metrics.json"
    if metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        (MODEL_ROOT / "training_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return backup_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证四个训练权重的任务和类别后，将其安装到应用模型目录。"
    )
    parser.add_argument("source", type=Path, help="包含四个模型子目录的训练结果目录。")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"训练结果目录不存在：{source}")
    weights = validate_bundle(source)
    backup = install(source, weights)
    print(f"模型更新完成。旧权重备份于：{backup}")


if __name__ == "__main__":
    main()
