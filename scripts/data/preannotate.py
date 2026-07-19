from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "ultralytics"
(RUNTIME_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已有 YOLO 权重生成待人工复核的预标注。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    image_dir = args.images.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    model = YOLO(str(model_path))
    review_rows: list[dict[str, str]] = []
    written = skipped = 0

    for image_path in images:
        label_path = output_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            continue
        results = model.predict(
            source=str(image_path),
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        if not results:
            label_path.write_text("", encoding="utf-8")
            count = 0
        else:
            results[0].save_txt(str(label_path), save_conf=False)
            count = len(results[0].boxes) if results[0].boxes is not None else 0
        review_rows.append(
            {
                "image": str(image_path),
                "prelabel": str(label_path),
                "predicted_objects": str(count),
                "review_status": "needs_review",
                "review_notes": "",
            }
        )
        written += 1

    manifest = output_dir / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "prelabel",
                "predicted_objects",
                "review_status",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"生成 {written} 个预标注，跳过 {skipped} 个；审核清单：{manifest}")


if __name__ == "__main__":
    main()
