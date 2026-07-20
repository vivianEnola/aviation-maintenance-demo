from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("cloud_scene", "airport_overview", "runway_detail", "other")


def image_files(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_resized(source: Path, destination: Path, max_side: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(destination.with_suffix(".jpg"), "JPEG", quality=90, optimize=True)


def reset_output(output: Path) -> None:
    resolved = output.resolve()
    runtime = (ROOT / ".runtime").resolve()
    if runtime not in resolved.parents:
        raise ValueError(f"Output must be inside {runtime}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def split_items(items: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    size = len(shuffled)
    train_end = max(1, round(size * 0.8))
    val_end = min(size, train_end + max(1, round(size * 0.1)))
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_classification(output: Path, exclusions: list[dict[str, str]]) -> dict[str, Any]:
    source_root = ROOT / "datasets" / "classification"
    by_hash: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            for path in image_files(source_root / split / class_name):
                by_hash[digest(path)].append((split, class_name, path))

    canonical: dict[str, list[Path]] = defaultdict(list)
    ambiguous_groups = 0
    for sha, entries in sorted(by_hash.items()):
        classes = {entry[1] for entry in entries}
        if len(classes) == 1:
            chosen_class = entries[0][1]
        elif classes == {"runway_detail", "other"}:
            # The duplicated runway-detail copies were visually verified as generic scenes.
            chosen_class = "other"
        else:
            ambiguous_groups += 1
            # Keep the first label but report it explicitly for later manual review.
            chosen_class = entries[0][1]
        chosen = next(entry[2] for entry in entries if entry[1] == chosen_class)
        canonical[chosen_class].append(chosen)
        for split, class_name, path in entries:
            if path != chosen or class_name != chosen_class:
                exclusions.append({
                    "dataset": "classification",
                    "path": str(path.relative_to(ROOT)),
                    "reason": f"exact_duplicate_of:{chosen.relative_to(ROOT)}",
                    "sha256": sha,
                })

    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for class_index, class_name in enumerate(CLASS_NAMES):
        assignments = split_items(canonical[class_name], 4200 + class_index)
        for split, paths in assignments.items():
            counts[split][class_name] = len(paths)
            for index, source in enumerate(paths):
                target = output / "classification" / split / class_name / f"{class_name}_{index:04d}.jpg"
                save_resized(source, target, 768)
    return {"counts": counts, "unique_images": sum(map(len, canonical.values())), "ambiguous_groups": ambiguous_groups}


def write_detection_yaml(directory: Path, names: dict[int, str]) -> None:
    payload = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": names}
    (directory / "data.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def prepare_detection(
    dataset_name: str,
    output: Path,
    max_side: int,
    names: dict[int, str],
    exclusions: list[dict[str, str]],
) -> dict[str, Any]:
    source = ROOT / "datasets" / dataset_name
    seen_hashes: dict[str, Path] = {}
    counts: dict[str, int] = {}
    instances: Counter[int] = Counter()
    for split in SPLITS:
        written = 0
        for image_path in image_files(source / "images" / split):
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                exclusions.append({"dataset": dataset_name, "path": str(image_path.relative_to(ROOT)), "reason": "missing_label"})
                continue
            sha = digest(image_path)
            if sha in seen_hashes:
                exclusions.append({
                    "dataset": dataset_name,
                    "path": str(image_path.relative_to(ROOT)),
                    "reason": f"exact_duplicate_of:{seen_hashes[sha].relative_to(ROOT)}",
                    "sha256": sha,
                })
                continue
            seen_hashes[sha] = image_path
            target_image = output / dataset_name / "images" / split / f"{image_path.stem}.jpg"
            save_resized(image_path, target_image, max_side)
            target_label = output / dataset_name / "labels" / split / f"{image_path.stem}.txt"
            target_label.parent.mkdir(parents=True, exist_ok=True)
            text = label_path.read_text(encoding="utf-8").strip()
            target_label.write_text(text + ("\n" if text else ""), encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    instances[int(line.split()[0])] += 1
            written += 1
        counts[split] = written
    destination = output / dataset_name
    write_detection_yaml(destination, names)
    return {"counts": counts, "instances": {names[index]: instances[index] for index in names}}


def runway_polygons(payload: dict[str, Any]) -> list[list[tuple[float, float]]]:
    return [
        [(float(x), float(y)) for x, y in shape.get("points", [])]
        for shape in payload.get("shapes", [])
        if shape.get("label") == "runway" and len(shape.get("points", [])) >= 3
    ]


def normalized_polygon(class_id: int, polygon: Iterable[tuple[float, float]], width: int, height: int) -> str:
    values: list[str] = [str(class_id)]
    for x, y in polygon:
        values.extend((f"{min(max(x / width, 0.0), 1.0):.6f}", f"{min(max(y / height, 0.0), 1.0):.6f}"))
    return " ".join(values)


def condition_mask(runway_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    ys, xs = np.where(runway_mask > 0)
    seeds = np.zeros_like(runway_mask)
    if len(xs) == 0:
        return seeds
    for _ in range(rng.randint(3, 7)):
        pos = rng.randrange(len(xs))
        cx, cy = int(xs[pos]), int(ys[pos])
        axes = (rng.randint(18, 68), rng.randint(7, 30))
        cv2.ellipse(seeds, (cx, cy), axes, rng.uniform(0, 180), 0, 360, 255, -1)
    seed_value = rng.randrange(2**32)
    noise_rng = np.random.default_rng(seed_value)
    noise = noise_rng.normal(0, 1, runway_mask.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), rng.uniform(5, 11))
    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    organic = cv2.bitwise_and(seeds, cv2.threshold(noise, rng.randint(92, 132), 255, cv2.THRESH_BINARY)[1])
    organic = cv2.morphologyEx(organic, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.bitwise_and(organic, runway_mask)


def paint_condition(image: np.ndarray, mask: np.ndarray, class_id: int, rng: random.Random) -> np.ndarray:
    result = image.copy().astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask, (0, 0), 2.8).astype(np.float32) / 255.0
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    if class_id == 1:
        grain = cv2.GaussianBlur(
            noise_rng.normal(0, 12, mask.shape).astype(np.float32), (0, 0), 1.4
        )
        texture = np.full_like(result, (218, 224, 228), dtype=np.float32)
        texture += grain[..., None]
        alpha = soft_mask[..., None] * 0.70
    else:
        blurred = cv2.GaussianBlur(image, (0, 0), 7).astype(np.float32)
        texture = blurred * 0.54 + np.array((25, 39, 48), dtype=np.float32)
        highlight = cv2.GaussianBlur(
            noise_rng.normal(0, 7, mask.shape).astype(np.float32), (0, 0), 3
        )
        texture += highlight[..., None]
        alpha = soft_mask[..., None] * 0.62
    result = result * (1 - alpha) + texture * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def mask_contours(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[tuple[float, float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append([(float(x), float(y)) for x, y in approx])
    return polygons


def prepare_runway(output: Path, exclusions: list[dict[str, str]], seed: int = 42) -> dict[str, Any]:
    source = ROOT / "datasets" / "runway_segmentation"
    destination = output / "runway_segmentation"
    originals: dict[str, int] = {}
    train_records: list[tuple[Path, list[list[tuple[float, float]]]]] = []
    for split in SPLITS:
        written = 0
        for image_path in image_files(source / "images" / split):
            json_path = source / "labels" / split / f"{image_path.stem}.json"
            if not json_path.is_file():
                exclusions.append({"dataset": "runway_segmentation", "path": str(image_path.relative_to(ROOT)), "reason": "missing_labelme_json"})
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            polygons = runway_polygons(payload)
            if not polygons:
                exclusions.append({"dataset": "runway_segmentation", "path": str(image_path.relative_to(ROOT)), "reason": "no_runway_polygon"})
                continue
            with Image.open(image_path) as raw:
                width, height = raw.size
            target_image = destination / "images" / split / f"{image_path.stem}.jpg"
            save_resized(image_path, target_image, 640)
            label_path = destination / "labels" / split / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(normalized_polygon(0, poly, width, height) for poly in polygons) + "\n", encoding="utf-8")
            if split == "train":
                train_records.append((target_image, polygons))
            written += 1
        originals[split] = written

    rng = random.Random(seed)
    rng.shuffle(train_records)
    synthetic_counts = {"snow": 0, "standing_water": 0, "mixed": 0}
    for index, (image_path, polygons) in enumerate(train_records):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        runway_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(runway_mask, [np.asarray(poly, dtype=np.int32) for poly in polygons], 255)
        condition_ids = [1] if index < 40 else ([2] if index < 80 else [1, 2])
        labels = [normalized_polygon(0, poly, width, height) for poly in polygons]
        augmented = image
        for class_id in condition_ids:
            mask = condition_mask(runway_mask, rng)
            augmented = paint_condition(augmented, mask, class_id, rng)
            labels.extend(normalized_polygon(class_id, poly, width, height) for poly in mask_contours(mask))
        category = "mixed" if len(condition_ids) == 2 else ("snow" if condition_ids[0] == 1 else "standing_water")
        synthetic_counts[category] += 1
        stem = f"synthetic_{category}_{index:04d}"
        Image.fromarray(augmented).save(destination / "images" / "train" / f"{stem}.jpg", quality=92)
        (destination / "labels" / "train" / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    write_detection_yaml(destination, {0: "runway", 1: "snow", 2: "standing_water"})
    return {"originals": originals, "synthetic": synthetic_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a cleaned, compact Kaggle training dataset without modifying source data.")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "kaggle_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    reset_output(output)
    exclusions: list[dict[str, str]] = []
    report = {
        "classification": prepare_classification(output, exclusions),
        "cloud_detection": prepare_detection(
            "cloud_detection", output, 1280,
            {0: "cumulonimbus", 1: "stratocumulus", 2: "typhoon_vortex"}, exclusions,
        ),
        "airport_detection": prepare_detection(
            "airport_detection", output, 1536,
            {0: "airport", 1: "large_building_cluster", 2: "construction_area"}, exclusions,
        ),
        "runway_segmentation": prepare_runway(output, exclusions),
    }
    report["excluded_files"] = len(exclusions)
    (output / "exclusions.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
