from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from .advisors import apply_advice
from .config import model_config, mode_config, resolve_project_path
from .schemas import AnalysisReport, ClassificationResult, VisionObject


class PredictModel(Protocol):
    def predict(self, **kwargs: Any) -> list[Any]: ...


ModelLoader = Callable[[str, str], PredictModel]


class ModelUnavailableError(FileNotFoundError):
    pass


@dataclass(slots=True)
class InferenceOutput:
    report: AnalysisReport
    annotated_image: Image.Image


def load_yolo_model(_model_id: str, source: str) -> PredictModel:
    from ultralytics import YOLO

    return YOLO(source)


def resolve_model_source(model_id: str) -> str:
    config = model_config(model_id)
    weights = str(config["weights"])
    if config.get("builtin", False):
        return weights
    path = resolve_project_path(weights)
    if not path.is_file():
        raise ModelUnavailableError(
            f"模型权重尚未就绪：{path}。请先训练模型或把 best.pt 放到该位置。"
        )
    return str(path)


def _name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _classification(
    result: Any, thresholds: dict[str, Any], *, placeholder: bool = False
) -> ClassificationResult:
    if result.probs is None:
        raise ValueError("分类模型没有返回类别概率。")
    probabilities = result.probs.data.detach().cpu().numpy().astype(float)
    order = np.argsort(probabilities)[::-1]
    top_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else None
    top_confidence = float(probabilities[top_index])
    second_confidence = (
        float(probabilities[second_index]) if second_index is not None else None
    )
    margin = (
        top_confidence - second_confidence if second_confidence is not None else 1.0
    )
    auto_cfg = thresholds.get("auto", {})
    confidence_key = "placeholder_min_confidence" if placeholder else "min_confidence"
    margin_key = "placeholder_min_margin" if placeholder else "min_margin"
    min_confidence = float(auto_cfg.get(confidence_key, 0.60))
    min_margin = float(auto_cfg.get(margin_key, 0.15))
    return ClassificationResult(
        label=_name(result.names, top_index),
        confidence=top_confidence,
        second_label=(
            _name(result.names, second_index) if second_index is not None else None
        ),
        second_confidence=second_confidence,
        accepted=top_confidence >= min_confidence and margin >= min_margin,
        ambiguous=margin < min_margin,
    )


def _objects(result: Any) -> list[VisionObject]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    coordinates = result.boxes.xyxy.detach().cpu().numpy().astype(float)
    mask_ratios: list[float | None] = [None] * len(classes)

    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()
        for index in range(min(len(mask_ratios), len(masks))):
            mask_ratios[index] = float(np.count_nonzero(masks[index]) / masks[index].size)

    return [
        VisionObject(
            label=_name(result.names, int(class_id)),
            confidence=float(confidence),
            xyxy=tuple(float(value) for value in box),
            mask_area_ratio=mask_ratios[index],
        )
        for index, (class_id, confidence, box) in enumerate(
            zip(classes, confidences, coordinates, strict=True)
        )
    ]


def _annotated_image(result: Any, fallback: Image.Image) -> Image.Image:
    try:
        plotted = result.plot(labels=True, conf=True)
    except (AttributeError, RuntimeError, ValueError):
        return fallback.copy()
    if plotted is None:
        return fallback.copy()
    if plotted.ndim == 3 and plotted.shape[2] >= 3:
        plotted = plotted[:, :, :3][:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(plotted)).convert("RGB")


def _predict(
    model: PredictModel,
    image: Image.Image,
    *,
    task: str,
    confidence: float,
    iou: float,
    image_size: int,
) -> Any:
    kwargs: dict[str, Any] = {
        "source": np.asarray(image),
        "imgsz": image_size,
        "verbose": False,
    }
    if task != "classify":
        kwargs.update({"conf": confidence, "iou": iou})
    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("模型没有返回推理结果。")
    return results[0]


def _run_downstream(
    *,
    requested_mode: str,
    model_id: str,
    image: Image.Image,
    confidence: float,
    iou: float,
    image_size: int,
    model_loader: ModelLoader,
    thresholds: dict[str, Any],
    rules: dict[str, Any],
    classification: ClassificationResult | None = None,
) -> InferenceOutput:
    config = model_config(model_id)
    source = resolve_model_source(model_id)
    model = model_loader(model_id, source)
    task = str(config["task"])
    result = _predict(
        model,
        image,
        task=task,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
    )
    report = AnalysisReport(
        requested_mode=requested_mode,
        executed_model=model_id,
        task=task,
        classification=classification,
        objects=_objects(result),
    )
    if config.get("placeholder", False):
        report.metadata["placeholder_model"] = True
        report.warnings.append(
            "当前使用最小合成数据训练的占位权重，只用于演示流程，不代表模型精度。"
        )
    apply_advice(report, rules, thresholds)
    return InferenceOutput(report=report, annotated_image=_annotated_image(result, image))


def run_inference(
    *,
    mode_id: str,
    image: Image.Image,
    confidence: float,
    iou: float,
    image_size: int,
    thresholds: dict[str, Any],
    rules: dict[str, Any],
    model_loader: ModelLoader = load_yolo_model,
) -> InferenceOutput:
    started = perf_counter()
    mode = mode_config(mode_id)

    if mode_id != "auto":
        output = _run_downstream(
            requested_mode=mode_id,
            model_id=str(mode["model"]),
            image=image,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            model_loader=model_loader,
            thresholds=thresholds,
            rules=rules,
        )
        output.report.processing_ms = (perf_counter() - started) * 1000
        return output

    classifier_id = str(mode["classifier"])
    classifier_config = model_config(classifier_id)
    classifier_source = resolve_model_source(classifier_id)
    classifier_model = model_loader(classifier_id, classifier_source)
    classifier_result = _predict(
        classifier_model,
        image,
        task="classify",
        confidence=confidence,
        iou=iou,
        image_size=image_size,
    )
    classifier_is_placeholder = bool(classifier_config.get("placeholder", False))
    classification = _classification(
        classifier_result,
        thresholds,
        placeholder=classifier_is_placeholder,
    )

    if not classification.accepted:
        report = AnalysisReport(
            requested_mode="auto",
            executed_model=None,
            task="classify",
            classification=classification,
            summary="分类结果置信度不足或类别差距过小，Auto 模式未调用下游模型。",
            recommendations=["请改用手动模型选择，或将图片加入待复核数据集。"],
            warnings=["低置信度图片不会被强制路由，以避免错误模型产生误导性建议。"],
            processing_ms=(perf_counter() - started) * 1000,
        )
        if classifier_is_placeholder:
            report.metadata["placeholder_model"] = True
            report.warnings.append(
                "当前 Auto 分类器是占位权重，分类结果只用于演示路由流程。"
            )
        return InferenceOutput(report=report, annotated_image=image.copy())

    route = mode.get("routes", {}).get(classification.label)
    if route is None:
        report = AnalysisReport(
            requested_mode="auto",
            executed_model=None,
            task="classify",
            classification=classification,
            summary="图片被分类为 other，不调用下游业务模型。",
            recommendations=["可使用 YOLOv8n 通用检测，或人工确认图片是否属于业务场景。"],
            processing_ms=(perf_counter() - started) * 1000,
        )
        if classifier_is_placeholder:
            report.metadata["placeholder_model"] = True
            report.warnings.append(
                "当前 Auto 分类器是占位权重，分类结果只用于演示路由流程。"
            )
        return InferenceOutput(report=report, annotated_image=image.copy())

    output = _run_downstream(
        requested_mode="auto",
        model_id=str(route),
        image=image,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        model_loader=model_loader,
        thresholds=thresholds,
        rules=rules,
        classification=classification,
    )
    output.report.processing_ms = (perf_counter() - started) * 1000
    return output
