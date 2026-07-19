from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path | None
    class_name: str
    group_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按来源组切分分类或 YOLO 检测/分割数据集，避免同源样本泄漏。"
    )
    parser.add_argument("task", choices=["classification", "yolo"])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="YOLO 标签目录；task=yolo 时必填。")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--status-column", default="review_status")
    parser.add_argument("--approved-value", default="approved")
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="只纳入清单中审核状态为 approved 的图片。",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="将缺少 YOLO 标签的图片作为空标签负样本纳入。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中同名文件；不会删除其它文件。",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": train, "val": val, "test": test}
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("train、val、test 比例都必须大于 0。")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("train、val、test 比例之和必须等于 1。")
    return ratios


def load_manifest(
    path: Path | None,
    *,
    image_column: str,
    group_column: str,
    status_column: str,
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {image_column, group_column}
        if not required.issubset(fields):
            missing = ", ".join(sorted(required - fields))
            raise ValueError(f"清单缺少列：{missing}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            image_value = (row.get(image_column) or "").strip()
            if not image_value:
                continue
            name = Path(image_value).name
            if name in rows and rows[name].get(group_column) != row.get(group_column):
                raise ValueError(f"清单中同名图片分组冲突：{name}")
            rows[name] = {
                group_column: (row.get(group_column) or "").strip(),
                status_column: (row.get(status_column) or "").strip(),
            }
        return rows


def discover_records(args: argparse.Namespace) -> list[Record]:
    image_root = args.images.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_root}")
    if args.task == "yolo" and args.labels is None:
        raise ValueError("task=yolo 时必须提供 --labels。")
    label_root = args.labels.resolve() if args.labels else None
    manifest = load_manifest(
        args.manifest,
        image_column=args.image_column,
        group_column=args.group_column,
        status_column=args.status_column,
    )

    records: list[Record] = []
    for image in sorted(image_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        manifest_row = manifest.get(image.name)
        if args.require_approved:
            if manifest_row is None:
                continue
            if manifest_row.get(args.status_column) != args.approved_value:
                continue
        group_id = (
            manifest_row.get(args.group_column, "") if manifest_row is not None else ""
        ) or image.stem

        if args.task == "classification":
            relative = image.relative_to(image_root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"分类图片必须位于类别子目录中：{image}（例如 images/cloud_scene/a.jpg）"
                )
            class_name = relative.parts[0]
            records.append(Record(image, None, class_name, group_id))
            continue

        assert label_root is not None
        relative = image.relative_to(image_root)
        label = (label_root / relative).with_suffix(".txt")
        if not label.is_file() and not args.allow_missing_labels:
            raise FileNotFoundError(f"图片缺少同名 YOLO 标签：{image} -> {label}")
        records.append(Record(image, label if label.is_file() else None, "__yolo__", group_id))

    if not records:
        qualifier = "且已审核" if args.require_approved else ""
        raise ValueError(f"没有发现可切分{qualifier}的图片。")
    return records


def assign_groups(
    records: list[Record], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)

    class_totals = Counter(record.class_name for record in records)
    assigned_total = Counter()
    assigned_classes: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    rng = random.Random(seed)
    tie_breakers = {group: rng.random() for group in grouped}
    ordered_groups = sorted(
        grouped,
        key=lambda group: (-len(grouped[group]), tie_breakers[group], group),
    )

    assignments: dict[str, str] = {}
    for group in ordered_groups:
        group_records = grouped[group]
        group_classes = Counter(record.class_name for record in group_records)

        def score(split: str) -> tuple[float, float, int]:
            target_total = len(records) * ratios[split]
            total_fill = (assigned_total[split] + len(group_records)) / target_total
            class_fills = []
            for class_name, count in group_classes.items():
                target_class = class_totals[class_name] * ratios[split]
                class_fills.append(
                    (assigned_classes[split][class_name] + count) / target_class
                )
            worst_fill = max([total_fill, *class_fills])
            return (worst_fill, total_fill, SPLITS.index(split))

        chosen = min(SPLITS, key=score)
        assignments[group] = chosen
        assigned_total[chosen] += len(group_records)
        assigned_classes[chosen].update(group_classes)
    return assignments


def destination_paths(record: Record, task: str, output: Path, split: str) -> tuple[Path, Path | None]:
    if task == "classification":
        return output / split / record.class_name / record.image.name, None
    return (
        output / "images" / split / record.image.name,
        output / "labels" / split / f"{record.image.stem}.txt",
    )


def copy_records(
    records: list[Record], assignments: dict[str, str], args: argparse.Namespace
) -> Path:
    output = args.output.resolve()
    rows: list[dict[str, str]] = []
    destinations: set[Path] = set()
    for record in records:
        split = assignments[record.group_id]
        image_destination, label_destination = destination_paths(record, args.task, output, split)
        if image_destination in destinations:
            raise ValueError(f"输出文件名冲突：{image_destination.name}")
        destinations.add(image_destination)
        if image_destination.exists() and not args.overwrite:
            raise FileExistsError(f"输出已存在；使用 --overwrite 才能覆盖：{image_destination}")
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, image_destination)

        if label_destination is not None:
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            if label_destination.exists() and not args.overwrite:
                raise FileExistsError(f"输出已存在；使用 --overwrite 才能覆盖：{label_destination}")
            if record.label is None:
                label_destination.write_text("", encoding="utf-8")
            else:
                shutil.copy2(record.label, label_destination)

        rows.append(
            {
                "image": str(record.image),
                "class_name": record.class_name,
                "group_id": record.group_id,
                "split": split,
                "output_image": str(image_destination),
            }
        )

    manifest_path = output / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args.train, args.val, args.test)
    records = discover_records(args)
    assignments = assign_groups(records, ratios, args.seed)
    manifest_path = copy_records(records, assignments, args)
    counts = Counter(assignments[record.group_id] for record in records)
    groups = Counter(assignments.values())
    print(
        "切分完成："
        + ", ".join(
            f"{split}={counts[split]} 张/{groups[split]} 组" for split in SPLITS
        )
    )
    print(f"切分清单：{manifest_path}")


if __name__ == "__main__":
    main()
