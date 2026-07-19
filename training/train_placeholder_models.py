from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "ultralytics"
(RUNTIME_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


MODEL_SPECS = {
    "cloud_detector": (
        "yolo11n.pt",
        "datasets/placeholder/cloud_detection/data.yaml",
    ),
    "airport_detector": (
        "yolo11n.pt",
        "datasets/placeholder/airport_detection/data.yaml",
    ),
    "runway_segmenter": (
        "yolo11n-seg.pt",
        "datasets/placeholder/runway_segmentation/data.yaml",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在同一进程中训练并发布三个最小检测/分割占位模型。"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_id, (pretrained, data_relative) in MODEL_SPECS.items():
        print(f"\n=== 训练 {model_id} ===", flush=True)
        model = YOLO(pretrained)
        results = model.train(
            data=str(PROJECT_ROOT / data_relative),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            workers=0,
            patience=0,
            seed=42,
            deterministic=True,
            project=str(PROJECT_ROOT / "runs" / "placeholder"),
            name=model_id,
            exist_ok=True,
            plots=False,
            verbose=True,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"未生成检查点：{best}")
        destination = PROJECT_ROOT / "models" / model_id / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        print(f"已发布：{destination}", flush=True)


if __name__ == "__main__":
    main()
