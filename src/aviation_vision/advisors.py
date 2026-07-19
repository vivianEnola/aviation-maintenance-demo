from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import AnalysisReport, VisionObject


def _display(label: str, rules: dict[str, Any]) -> str:
    return str(rules.get("display_names", {}).get(label, label))


def _max_confidence(objects: list[VisionObject], label: str) -> float:
    values = [item.confidence for item in objects if item.label == label]
    return max(values, default=0.0)


def _format_detected(objects: list[VisionObject], rules: dict[str, Any]) -> str:
    counts = Counter(item.label for item in objects)
    if not counts:
        return "未检测到达到当前置信度阈值的目标。"
    parts = []
    for label, count in sorted(counts.items()):
        confidence = _max_confidence(objects, label)
        parts.append(f"{_display(label, rules)} {count} 个（最高置信度 {confidence:.1%}）")
    return "检测到" + "、".join(parts) + "。"


def _center_inside(candidate: VisionObject, region: tuple[float, float, float, float]) -> bool:
    if candidate.xyxy is None:
        return False
    x1, y1, x2, y2 = candidate.xyxy
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    rx1, ry1, rx2, ry2 = region
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def _expanded_box(
    box: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = max(0.0, x2 - x1) * scale / 2
    half_height = max(0.0, y2 - y1) * scale / 2
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def analyze_general(report: AnalysisReport, rules: dict[str, Any]) -> None:
    report.summary = _format_detected(report.objects, rules)
    if report.objects:
        report.recommendations.append("通用检测结果仅描述 COCO 物体类别，不代表航空专业风险判断。")


def analyze_cloud(report: AnalysisReport, rules: dict[str, Any]) -> None:
    labels = {item.label for item in report.objects}
    report.summary = _format_detected(report.objects, rules)

    if not labels:
        report.recommendations.append(
            "当前图片未检出指定云型；仍应结合官方气象资料确认航路天气。"
        )
        return

    knowledge = rules.get("weather_knowledge", {})
    for label in ("cumulonimbus", "stratocumulus", "typhoon_vortex"):
        if label in labels and label in knowledge:
            report.knowledge.append(str(knowledge[label]))

    if "typhoon_vortex" in labels:
        report.recommendations.extend(
            [
                "发现疑似涡旋状天气系统，应立即结合气象雷达、卫星云图、SIGMET 和管制信息复核。",
                "避免根据本系统单独规划穿越该云系的航路。",
            ]
        )
    if "cumulonimbus" in labels:
        report.recommendations.append(
            "积雨云可能伴随雷暴、结冰和强烈气流，建议结合正式气象资料规划绕飞。"
        )
    if "stratocumulus" in labels:
        report.recommendations.append(
            "层积云通常风险低于强对流云，但应关注低云底、能见度和轻微降水。"
        )

    report.warnings.append("云型结论来自普通 RGB 图像，只能作为课程展示中的视觉辅助判断。")


def analyze_airport(
    report: AnalysisReport,
    rules: dict[str, Any],
    thresholds: dict[str, Any],
) -> None:
    report.summary = _format_detected(report.objects, rules)
    airports = [item for item in report.objects if item.label == "airport" and item.xyxy]
    surroundings = [
        item
        for item in report.objects
        if item.label in {"large_building_cluster", "construction_area"}
    ]

    if not airports:
        report.recommendations.append("未可靠定位机场主体，无法评估周边目标与机场的相对位置。")
        return

    airport_cfg = thresholds.get("airport", {})
    scale = float(airport_cfg.get("attention_scale", 2.0))
    concern_conf = float(airport_cfg.get("concern_confidence", 0.55))
    attention_regions = [_expanded_box(item.xyxy, scale) for item in airports if item.xyxy]
    nearby = [
        item
        for item in surroundings
        if item.confidence >= concern_conf
        and any(_center_inside(item, region) for region in attention_regions)
    ]

    if nearby:
        labels = sorted({_display(item.label, rules) for item in nearby})
        report.recommendations.extend(
            [
                f"机场周边关注区域内发现高置信度目标：{'、'.join(labels)}。",
                "建议人工核对机场净空保护要求、施工许可和最新现场资料。",
            ]
        )
        report.warnings.append("系统不能从单幅 RGB 图像判断建筑高度、审批状态或是否违建。")
    else:
        report.recommendations.append(
            "当前阈值下未发现进入机场周边关注区域的高置信度建筑群或施工区域。"
        )


def analyze_runway(
    report: AnalysisReport,
    rules: dict[str, Any],
    thresholds: dict[str, Any],
) -> None:
    report.summary = _format_detected(report.objects, rules)
    runway_area = sum(
        item.mask_area_ratio or 0.0 for item in report.objects if item.label == "runway"
    )
    snow_area = sum(item.mask_area_ratio or 0.0 for item in report.objects if item.label == "snow")
    water_area = sum(
        item.mask_area_ratio or 0.0
        for item in report.objects
        if item.label == "standing_water"
    )
    contamination_area = snow_area + water_area
    relative_area = contamination_area / runway_area if runway_area > 0 else None
    max_condition_conf = max(
        _max_confidence(report.objects, "snow"),
        _max_confidence(report.objects, "standing_water"),
    )

    report.metadata.update(
        {
            "runway_mask_ratio": runway_area,
            "snow_mask_ratio": snow_area,
            "water_mask_ratio": water_area,
            "condition_to_runway_ratio": relative_area,
        }
    )

    if runway_area <= 0:
        report.recommendations.append("未可靠分割出跑道，无法判断道面是否需要清理。")
        return

    if contamination_area <= 0:
        report.recommendations.append(
            "当前阈值下未分割出积雪或疑似积水；仍需按机场规程进行现场巡检。"
        )
        return

    runway_cfg = thresholds.get("runway", {})
    review_ratio = float(runway_cfg.get("review_area_ratio", 0.005))
    action_ratio = float(runway_cfg.get("action_area_ratio", 0.05))
    concern_conf = float(runway_cfg.get("concern_confidence", 0.50))
    ratio = relative_area or 0.0

    if ratio >= action_ratio and max_condition_conf >= concern_conf:
        report.recommendations.extend(
            [
                f"疑似污染区域约占跑道掩膜的 {ratio:.1%}，建议优先安排现场检查和清理。",
                "清理后应按机场运行规程复核道面状况，不可仅凭本系统恢复使用。",
            ]
        )
    elif ratio >= review_ratio or max_condition_conf >= concern_conf:
        report.recommendations.append(
            f"检测到需要复核的道面区域（相对面积约 {ratio:.1%}），建议进行现场巡检。"
        )
    else:
        report.recommendations.append(
            "检测区域较小或置信度较低，建议人工核对后再决定是否清理。"
        )
    report.warnings.append("普通 RGB 图像中的阴影和深色铺装可能与积水混淆。")


def apply_advice(
    report: AnalysisReport,
    rules: dict[str, Any],
    thresholds: dict[str, Any],
) -> AnalysisReport:
    if report.executed_model == "general":
        analyze_general(report, rules)
    elif report.executed_model == "cloud_detector":
        analyze_cloud(report, rules)
    elif report.executed_model == "airport_detector":
        analyze_airport(report, rules, thresholds)
    elif report.executed_model == "runway_segmenter":
        analyze_runway(report, rules, thresholds)
    return report
