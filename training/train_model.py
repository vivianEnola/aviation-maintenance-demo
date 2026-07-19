from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "ultralytics"
(RUNTIME_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


def load_models() -> dict[str, dict[str, Any]]:
    with (PROJECT_ROOT / "configs" / "models.yaml").open("r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {})["models"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练项目中的分类、检测或分割模型。")
    parser.add_argument(
        "model_id",
        choices=["scene_classifier", "cloud_detector", "airport_detector", "runway_segmenter"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=Path, help="覆盖 models.yaml 中的数据集路径。")
    parser.add_argument("--pretrained", help="覆盖 models.yaml 中的预训练权重。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="验证训练输出存在后，将 best.pt 复制到 models/<model>/best.pt。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_models()[args.model_id]
    data = args.data.resolve() if args.data else PROJECT_ROOT / str(config["data"])
    if not data.exists():
        raise FileNotFoundError(f"数据集不存在：{data}")
    run_name = args.model_id
    project = PROJECT_ROOT / "runs" / "train"
    model = YOLO(str(args.pretrained or config["pretrained"]))
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=args.resume,
        resume=args.resume,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"训练输出：{save_dir}")
    if not best.is_file():
        raise FileNotFoundError(f"训练结束但未找到 best.pt：{best}")
    if args.promote:
        destination = PROJECT_ROOT / str(config["weights"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        print(f"已发布模型：{destination}")
    else:
        print("未发布权重；确认验证结果后使用 --promote。")


if __name__ == "__main__":
    main()
