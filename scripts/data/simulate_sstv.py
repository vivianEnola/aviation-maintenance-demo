from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.65, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.15))
    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.4)))

    array = np.asarray(image).astype(np.float32)
    channel_gain = np.array(
        [rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)],
        dtype=np.float32,
    )
    array *= channel_gain
    array += rng.uniform(-12, 12)
    array += np.random.default_rng(rng.randrange(2**32)).normal(
        0, rng.uniform(2, 12), size=array.shape
    )

    line_interval = rng.randint(2, 6)
    line_strength = rng.uniform(0.72, 0.94)
    array[::line_interval] *= line_strength

    height, width = array.shape[:2]
    for start in range(0, height, rng.randint(12, 32)):
        if rng.random() < 0.45:
            end = min(height, start + rng.randint(1, 5))
            shift = rng.randint(-max(1, width // 50), max(1, width // 50))
            array[start:end] = np.roll(array[start:end], shift, axis=1)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为清晰 RGB 图片生成可复现的 SSTV 风格退化样本。")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--quality", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []

    for source in sorted(input_dir.iterdir()):
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(source) as image:
            clean = image.convert("RGB")
            for index in range(max(1, args.variants)):
                target = output_dir / f"{source.stem}_sstv_{index + 1:02d}.jpg"
                degraded = degrade(clean, rng)
                degraded.save(target, format="JPEG", quality=args.quality, optimize=True)
                rows.append(
                    {
                        "source": str(source),
                        "generated": str(target),
                        "capture_domain": "simulated_sstv",
                        "seed": str(args.seed),
                    }
                )

    manifest = output_dir / "simulation_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source", "generated", "capture_domain", "seed"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"生成 {len(rows)} 张退化图片；清单：{manifest}")


if __name__ == "__main__":
    main()

