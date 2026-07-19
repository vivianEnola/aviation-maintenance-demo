from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: Path) -> tuple[Path, dict[int, str], dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    names_raw = config.get("names", {})
    if isinstance(names_raw, list):
        names = {index: str(name) for index, name in enumerate(names_raw)}
    else:
        names = {int(index): str(name) for index, name in names_raw.items()}
    splits = {
        split: str(config[split])
        for split in ("train", "val", "test")
        if config.get(split)
    }
    return root, names, splits


def validate_line(line: str, task: str, class_ids: set[int]) -> tuple[int | None, str | None]:
    parts = line.split()
    if not parts:
        return None, None
    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None, "标签包含非数字内容"
    if class_id not in class_ids:
        return None, f"类别 ID {class_id} 不在 data.yaml 中"
    if task == "detect" and len(values) != 4:
        return None, "检测标签必须包含 4 个坐标"
    if task == "segment" and (len(values) < 6 or len(values) % 2 != 0):
        return None, "分割标签必须包含至少 3 个点"
    if any(value < 0 or value > 1 for value in values):
        return None, "归一化坐标超出 [0, 1]"
    return class_id, None


def audit(data_yaml: Path, task: str) -> dict[str, Any]:
    root, names, splits = load_data(data_yaml)
    result: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "task": task,
        "classes": names,
        "splits": {},
        "errors": [],
        "duplicates_across_splits": [],
    }
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split, relative in splits.items():
        image_dir = root / relative
        label_dir = Path(str(image_dir).replace("images", "labels"))
        images = sorted(
            path for path in image_dir.glob("**/*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_counts: Counter[int] = Counter()
        missing = empty = 0
        for image in images:
            hashes[sha256(image)].append((split, str(image)))
            relative_image = image.relative_to(image_dir)
            label = (label_dir / relative_image).with_suffix(".txt")
            if not label.is_file():
                missing += 1
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line_number, line in enumerate(lines, start=1):
                class_id, error = validate_line(line, task, set(names))
                if error:
                    result["errors"].append(
                        {"file": str(label), "line": line_number, "error": error}
                    )
                elif class_id is not None:
                    class_counts[class_id] += 1
        result["splits"][split] = {
            "images": len(images),
            "missing_labels": missing,
            "empty_negative_labels": empty,
            "class_instances": {
                names[class_id]: class_counts[class_id] for class_id in sorted(names)
            },
        }

    for digest, occurrences in hashes.items():
        if len({split for split, _ in occurrences}) > 1:
            result["duplicates_across_splits"].append(
                {"sha256": digest, "files": occurrences}
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 YOLO 检测/分割数据集结构与标签。")
    parser.add_argument("data", type=Path)
    parser.add_argument("--task", choices=["detect", "segment"], required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.data.resolve(), args.task)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.resolve().write_text(serialized, encoding="utf-8")
    if report["errors"] or report["duplicates_across_splits"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
