from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aviation_vision.config import load_yaml
from src.aviation_vision.inference import load_yolo_model, run_inference
from src.aviation_vision.runtime import configure_runtime


def main() -> None:
    configure_runtime()
    thresholds = load_yaml("thresholds.yaml")
    rules = load_yaml("advisory_rules.yaml")
    models: dict[tuple[str, str], Any] = {}

    def cached_loader(model_id: str, source: str) -> Any:
        key = (model_id, source)
        if key not in models:
            models[key] = load_yolo_model(model_id, source)
        return models[key]

    cases = [
        (
            "general",
            PROJECT_ROOT
            / "datasets/placeholder/airport_detection/images/test/placeholder_airport.png",
        ),
        (
            "cloud",
            PROJECT_ROOT
            / "datasets/placeholder/cloud_detection/images/test/placeholder_clouds.png",
        ),
        (
            "airport",
            PROJECT_ROOT
            / "datasets/placeholder/airport_detection/images/test/placeholder_airport.png",
        ),
        (
            "runway",
            PROJECT_ROOT
            / "datasets/placeholder/runway_segmentation/images/test/placeholder_runway.png",
        ),
        (
            "auto",
            PROJECT_ROOT
            / "datasets/placeholder/cloud_detection/images/test/placeholder_clouds.png",
        ),
    ]
    output_dir = PROJECT_ROOT / "outputs" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode_id, image_path in cases:
        image = Image.open(image_path).convert("RGB")
        result = run_inference(
            mode_id=mode_id,
            image=image,
            confidence=0.01,
            iou=0.45,
            image_size=320,
            thresholds=thresholds,
            rules=rules,
            model_loader=cached_loader,
        )
        result.annotated_image.save(output_dir / f"{mode_id}.png")
        classification = result.report.classification
        route = (
            f", class={classification.label}:{classification.confidence:.3f}"
            if classification
            else ""
        )
        print(
            f"{mode_id}: model={result.report.executed_model}, "
            f"task={result.report.task}, objects={len(result.report.objects)}{route}"
        )
    print(f"端到端推理烟雾测试完成：{output_dir}")


if __name__ == "__main__":
    main()
