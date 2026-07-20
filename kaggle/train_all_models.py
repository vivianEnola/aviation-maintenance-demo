from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("YOLO_CONFIG_DIR", "/kaggle/working/ultralytics-config")

offline_roots = sorted(
    {
        candidate.parent
        for candidate in Path("/kaggle/input").rglob("yolo11n.pt")
        if all(
            (candidate.parent / filename).is_file()
            for filename in ("yolo11n-cls.pt", "yolo11n-seg.pt")
        )
    }
)
if len(offline_roots) != 1:
    raise FileNotFoundError(
        f"Cannot uniquely locate offline YOLO runtime: {offline_roots}"
    )
OFFLINE_ROOT = offline_roots[0]

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(OFFLINE_ROOT),
            "polars-runtime-32==1.42.1",
            "polars==1.42.1",
            "ultralytics-thop==2.0.20",
            "ultralytics==8.4.101",
        ]
    )

from ultralytics import YOLO


INPUT_ROOT = Path("/kaggle/input/aviation-maintenance-training-data")
WORK_ROOT = Path("/kaggle/working/aviation-training")
OUTPUT_ROOT = Path("/kaggle/working/trained_models")

JOBS = (
    {
        "id": "scene_classifier",
        "pretrained": "yolo11n-cls.pt",
        "data": "classification",
        "epochs": 70,
        "imgsz": 320,
        "batch": 64,
        "patience": 15,
    },
    {
        "id": "cloud_detector",
        "pretrained": "yolo11n.pt",
        "data": "cloud_detection/data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 20,
    },
    {
        "id": "airport_detector",
        "pretrained": "yolo11n.pt",
        "data": "airport_detection/data.yaml",
        "epochs": 100,
        "imgsz": 1024,
        "batch": 8,
        "patience": 20,
    },
    {
        "id": "runway_segmenter",
        "pretrained": "yolo11n-seg.pt",
        "data": "runway_segmentation/data.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 12,
        "patience": 20,
    },
)


def find_input_root() -> Path:
    if (INPUT_ROOT / "classification").is_dir():
        return INPUT_ROOT
    candidates: list[Path] = []
    for classification_dir in Path("/kaggle/input").rglob("classification"):
        root = classification_dir.parent
        if all((root / name).exists() for name in ("cloud_detection", "airport_detection", "runway_segmentation")):
            candidates.append(root)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        mounted = sorted(str(path) for path in Path("/kaggle/input").iterdir())
        raise FileNotFoundError(f"Cannot uniquely locate prepared dataset: candidates={unique}, mounted={mounted}")
    return unique[0]


def materialize_dataset(input_root: Path) -> Path:
    if (input_root / "classification").is_dir():
        return input_root
    archive_paths = sorted(input_root.glob("*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"No dataset directories or ZIP archives found under {input_root}")
    extracted = WORK_ROOT / "dataset"
    extracted.mkdir(parents=True, exist_ok=True)
    for archive in archive_paths:
        target = extracted / archive.stem
        target.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(target))
    return extracted


def absolute_data(source: Path, job_id: str) -> Path:
    if source.is_dir():
        return source
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    payload["path"] = str(source.parent.resolve())
    target = WORK_ROOT / "data" / f"{job_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def serializable_metrics(metrics: Any) -> dict[str, float]:
    values = getattr(metrics, "results_dict", {}) or {}
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def train_job(dataset_root: Path, job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job["id"])
    data = absolute_data(dataset_root / str(job["data"]), job_id)
    pretrained = OFFLINE_ROOT / str(job["pretrained"])
    if not pretrained.is_file():
        raise FileNotFoundError(f"Missing offline pretrained weights: {pretrained}")
    model = YOLO(str(pretrained))
    result = model.train(
        data=str(data),
        epochs=int(job["epochs"]),
        imgsz=int(job["imgsz"]),
        batch=int(job["batch"]),
        device=0,
        workers=4,
        patience=int(job["patience"]),
        seed=42,
        deterministic=True,
        amp=True,
        plots=True,
        project=str(WORK_ROOT / "runs"),
        name=job_id,
        exist_ok=True,
        verbose=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without best.pt: {best}")
    destination = OUTPUT_ROOT / job_id / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    validator = YOLO(str(destination)).val(data=str(data), split="test", device=0, plots=True)
    return {
        "model": job_id,
        "pretrained": job["pretrained"],
        "epochs_requested": job["epochs"],
        "image_size": job["imgsz"],
        "best_weights": str(destination),
        "test_metrics": serializable_metrics(validator),
    }


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_root = materialize_dataset(find_input_root())
    reports: list[dict[str, Any]] = []
    for job in JOBS:
        print(f"\n===== Training {job['id']} =====", flush=True)
        report = train_job(dataset_root, job)
        reports.append(report)
        (OUTPUT_ROOT / "metrics.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    shutil.make_archive("/kaggle/working/aviation_trained_models", "zip", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
