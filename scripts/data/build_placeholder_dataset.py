from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("cloud_scene", "airport_overview", "runway_detail", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成仅用于打通训练与部署流程的最小合成占位数据集。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "placeholder",
    )
    parser.add_argument("--size", type=int, default=320)
    return parser.parse_args()


def save_all_splits(image: Image.Image, root: Path, stem: str) -> None:
    for split in SPLITS:
        target = root / "images" / split / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG", optimize=True)


def write_all_labels(root: Path, stem: str, content: str) -> None:
    for split in SPLITS:
        target = root / "labels" / split / f"{stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def classification_image(class_name: str, size: int, variant: int = 0) -> Image.Image:
    colors = {
        "cloud_scene": (70, 125, 185),
        "airport_overview": (95, 125, 75),
        "runway_detail": (80, 85, 90),
        "other": (175, 125, 65),
    }
    base = colors[class_name]
    offset = variant * 5
    image = Image.new(
        "RGB",
        (size, size),
        tuple(min(255, channel + offset) for channel in base),
    )
    draw = ImageDraw.Draw(image)
    if class_name == "cloud_scene":
        draw.ellipse((25, 70, 210, 210), fill=(242, 245, 248))
        draw.ellipse((130, 90, 300, 225), fill=(225, 232, 240))
    elif class_name == "airport_overview":
        draw.polygon(
            [(35, 250), (58, 270), (285, 65), (265, 45)],
            fill=(205, 208, 210),
        )
        draw.rectangle((215, 210, 295, 290), fill=(155, 165, 145))
    elif class_name == "runway_detail":
        draw.rectangle((95, 0, 225, size), fill=(45, 50, 55))
        for y in range(10, size, 55):
            draw.rectangle((155, y, 165, min(size, y + 32)), fill="white")
        draw.rectangle((95, 195, 225, 250), fill=(215, 230, 240))
    else:
        draw.rectangle((55, 55, 265, 265), fill=(215, 165, 80))
        draw.ellipse((110, 110, 210, 210), fill=(80, 55, 35))
    return image


def build_classification(root: Path, size: int) -> None:
    for class_name in CLASS_NAMES:
        for variant in range(4):
            image = classification_image(class_name, size, variant)
            suffix = "" if variant == 0 else f"_{variant}"
            for split in SPLITS:
                target = (
                    root
                    / split
                    / class_name
                    / f"placeholder_{class_name}{suffix}.png"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, format="PNG", optimize=True)


def build_cloud_detection(root: Path, size: int) -> None:
    image = Image.new("RGB", (size, size), (40, 90, 145))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 25, 140, 150), fill=(250, 250, 250))
    draw.rectangle((165, 45, 305, 140), fill=(215, 225, 235))
    for radius, shade in ((70, 245), (50, 220), (28, 175)):
        draw.arc(
            (160 - radius, 245 - radius, 160 + radius, 245 + radius),
            start=20,
            end=330,
            fill=(shade, shade, shade),
            width=max(5, radius // 5),
        )
    save_all_splits(image, root, "placeholder_clouds")
    labels = (
        "0 0.257812 0.273438 0.359375 0.390625\n"
        "1 0.734375 0.289062 0.437500 0.296875\n"
        "2 0.500000 0.765625 0.500000 0.437500\n"
    )
    write_all_labels(root, "placeholder_clouds", labels)


def build_airport_detection(root: Path, size: int) -> None:
    image = Image.new("RGB", (size, size), (95, 125, 75))
    draw = ImageDraw.Draw(image)
    draw.polygon([(25, 265), (48, 290), (292, 55), (268, 30)], fill=(190, 195, 200))
    draw.line((42, 278, 280, 43), fill="white", width=5)
    draw.rectangle((215, 25, 310, 115), fill=(150, 155, 145))
    for x in range(220, 305, 24):
        for y in range(30, 110, 24):
            draw.rectangle((x, y, x + 15, y + 15), fill=(215, 215, 205))
    draw.rectangle((15, 225, 85, 305), fill=(175, 125, 70))
    draw.line((20, 230, 80, 300), fill=(235, 210, 90), width=5)
    save_all_splits(image, root, "placeholder_airport")
    labels = (
        "0 0.500000 0.500000 0.937500 0.937500\n"
        "1 0.820312 0.218750 0.296875 0.281250\n"
        "2 0.156250 0.828125 0.218750 0.250000\n"
    )
    write_all_labels(root, "placeholder_airport", labels)


def build_runway_segmentation(root: Path, size: int) -> None:
    image = Image.new("RGB", (size, size), (90, 120, 75))
    draw = ImageDraw.Draw(image)
    runway = [(95, 5), (225, 5), (225, 315), (95, 315)]
    snow = [(95, 55), (225, 55), (225, 125), (95, 125)]
    water = [(95, 210), (225, 210), (225, 270), (95, 270)]
    draw.polygon(runway, fill=(52, 57, 62))
    draw.polygon(snow, fill=(235, 242, 248))
    draw.polygon(water, fill=(45, 95, 135))
    for y in (25, 145, 285):
        draw.rectangle((156, y, 164, y + 30), fill="white")
    save_all_splits(image, root, "placeholder_runway")
    labels = (
        "0 0.296875 0.015625 0.703125 0.015625 0.703125 0.984375 0.296875 0.984375\n"
        "1 0.296875 0.171875 0.703125 0.171875 0.703125 0.390625 0.296875 0.390625\n"
        "2 0.296875 0.656250 0.703125 0.656250 0.703125 0.843750 0.296875 0.843750\n"
    )
    write_all_labels(root, "placeholder_runway", labels)


def write_yaml(root: Path, relative_name: str, names: tuple[str, ...]) -> None:
    dataset = root / relative_name
    content = (
        f"path: datasets/placeholder/{relative_name}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(names))
    )
    (dataset / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT not in output.parents:
        raise ValueError(f"占位数据输出必须位于项目目录内：{output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    build_classification(output / "classification", args.size)
    build_cloud_detection(output / "cloud_detection", args.size)
    build_airport_detection(output / "airport_detection", args.size)
    build_runway_segmentation(output / "runway_segmentation", args.size)
    write_yaml(
        output,
        "cloud_detection",
        ("cumulonimbus", "stratocumulus", "typhoon_vortex"),
    )
    write_yaml(
        output,
        "airport_detection",
        ("airport", "large_building_cluster", "construction_area"),
    )
    write_yaml(
        output,
        "runway_segmentation",
        ("runway", "snow", "standing_water"),
    )
    (output / "README.md").write_text(
        "# 占位数据集\n\n"
        "该目录只用于打通训练、推理和部署流程。每类样本极少且为合成图，"
        "不得用于报告模型精度或形成业务结论。\n",
        encoding="utf-8",
    )
    print(f"占位数据集已生成：{output}")


if __name__ == "__main__":
    main()
