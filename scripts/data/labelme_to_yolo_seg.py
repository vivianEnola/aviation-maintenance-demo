from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def class_map(data_yaml: Path) -> dict[str, int]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {str(name): index for index, name in enumerate(names)}
    return {str(name): int(index) for index, name in names.items()}


def image_size(payload: dict[str, Any], json_path: Path) -> tuple[int, int]:
    width = payload.get("imageWidth")
    height = payload.get("imageHeight")
    if width and height:
        return int(width), int(height)
    image_path = payload.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_path} 缺少 imageWidth/imageHeight/imagePath。")
    candidate = (json_path.parent / str(image_path)).resolve()
    with Image.open(candidate) as image:
        return image.size


def convert_file(json_path: Path, output_path: Path, classes: dict[str, int]) -> int:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    width, height = image_size(payload, json_path)
    lines: list[str] = []
    for shape in payload.get("shapes", []):
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        label = str(shape.get("label", ""))
        if label not in classes:
            raise ValueError(f"{json_path}: 未知类别 {label!r}。")
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError(f"{json_path}: {label} 多边形少于 3 个点。")
        normalized: list[str] = []
        for point in points:
            x = min(max(float(point[0]) / width, 0.0), 1.0)
            y = min(max(float(point[1]) / height, 0.0), 1.0)
            normalized.extend((f"{x:.6f}", f"{y:.6f}"))
        lines.append(" ".join([str(classes[label]), *normalized]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Labelme Polygon JSON 转为 YOLO segmentation 标签。")
    parser.add_argument("input", type=Path, help="包含 Labelme JSON 的目录。")
    parser.add_argument("output", type=Path, help="YOLO 标签输出目录。")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/runway_segmentation/data.yaml"),
        help="包含类别 names 的 Ultralytics data.yaml。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = class_map(args.data.resolve())
    converted = shapes = skipped = 0
    json_files = [
        path
        for path in sorted(args.input.resolve().glob("*.json"))
        if not path.name.endswith(".geo.json")
    ]
    for json_path in json_files:
        output_path = args.output.resolve() / f"{json_path.stem}.txt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        shapes += convert_file(json_path, output_path, classes)
        converted += 1
    print(f"转换 {converted} 个文件、{shapes} 个多边形，跳过 {skipped} 个已有标签。")


if __name__ == "__main__":
    main()
