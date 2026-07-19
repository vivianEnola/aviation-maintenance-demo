from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "ultralytics"
(RUNTIME_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将已发布的项目模型导出为部署格式。")
    parser.add_argument(
        "model_id",
        choices=["scene_classifier", "cloud_detector", "airport_detector", "runway_segmenter"],
    )
    parser.add_argument("--format", default="onnx", choices=["onnx", "openvino", "engine"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (PROJECT_ROOT / "configs" / "models.yaml").open("r", encoding="utf-8") as handle:
        config = (yaml.safe_load(handle) or {})["models"][args.model_id]
    weights = PROJECT_ROOT / str(config["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"模型尚未发布：{weights}")
    model = YOLO(str(weights))
    output = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.format == "onnx",
    )
    print(f"导出完成：{output}")


if __name__ == "__main__":
    main()
