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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将占位分类器快速过拟合到合成样例，以演示 Auto 路由。"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO("yolo11n-cls.pt")
    results = model.train(
        data=str(PROJECT_ROOT / "datasets" / "placeholder" / "classification"),
        epochs=args.epochs,
        imgsz=64,
        batch=16,
        device=args.device,
        workers=0,
        patience=0,
        seed=42,
        deterministic=True,
        project=str(PROJECT_ROOT / "runs" / "placeholder"),
        name="scene_classifier_overfit",
        exist_ok=True,
        plots=False,
        freeze=10,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        erasing=0.0,
        fliplr=0.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    destination = PROJECT_ROOT / "models" / "scene_classifier" / "best.pt"
    shutil.copy2(best, destination)
    print(f"已发布：{destination}")


if __name__ == "__main__":
    main()
