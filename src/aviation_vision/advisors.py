from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import AnalysisReport, VisionObject


def _display(label: str, rules: dict[str, Any]) -> str:
    return str(rules.get("display_names", {}).get(label, label))


def _max_confidence(objects: list[VisionObject], label: str) -> float:
    values = [item.confidence for item in objects if item.label == label]
    return max(values, default=0.0)


def _confidence_level(value: float) -> str:
    if value >= 0.75:
        return "高"
    if value >= 0.50:
        return "中"
    return "低"


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
        report.recommendations.extend(
            [
                "建议结合目标位置、数量和最高置信度复核原图；低置信度或小尺寸目标优先人工确认。",
                "通用检测使用 COCO 类别体系，不直接代表航空运行风险或设施合规结论。",
            ]
        )
    else:
        report.recommendations.append(
            "未检出通用类别不等于画面中没有目标；可适当降低阈值或提高输入尺寸后复核。"
        )


def analyze_cloud(report: AnalysisReport, rules: dict[str, Any]) -> None:
    labels = {item.label for item in report.objects}
    report.summary = _format_detected(report.objects, rules)

    if not labels:
        report.summary += " 这不等同于无云或无危险天气，可能存在未纳入模型的云型、薄云或尺度较小的目标。"
        report.recommendations.extend(
            [
                "飞行前可结合 METAR、TAF、SIGMET、气象雷达和近期卫星云图了解云系发展趋势，重点留意航路上的低云、强降水与对流活动。",
                "图片存在明显压缩失真、黑边或云体截断时，可重新获取清晰图像后再作判断。",
            ]
        )
        return

    detected_confidences = [item.confidence for item in report.objects if item.label in labels]
    peak = max(detected_confidences, default=0.0)
    report.summary += f" 综合判定可信度为{_confidence_level(peak)}（最高 {peak:.1%}）；多类并存时按风险较高的天气系统优先处置。"

    knowledge = rules.get("weather_knowledge", {})
    for label in ("cumulonimbus", "stratocumulus", "typhoon_vortex"):
        if label in labels and label in knowledge:
            report.knowledge.append(str(knowledge[label]))

    if "typhoon_vortex" in labels:
        report.recommendations.extend(
            [
                "发现疑似白色涡旋状台风云系：立即查验热带气旋通报、最佳路径资料、SIGMET、雷达回波与管制信息，确认中心位置、移动方向和影响半径。",
                "航路与备降方案应避开强对流核心区及其外围雨带；不要依据单幅 RGB 图像估算台风等级、风速或安全间隔。",
                "评估目的地和备降场侧风、暴雨、低能见度、风切变以及地面保障中断风险，并准备延误、返航或改航方案。",
            ]
        )
    if "cumulonimbus" in labels:
        report.recommendations.append(
            "积雨云可能伴随雷暴、冰雹、积冰、强降水、风切变和强烈垂直气流；禁止仅凭图像尝试穿越，按运行手册和管制要求规划绕飞并持续监控回波演变。"
        )
    if "stratocumulus" in labels:
        report.recommendations.append(
            "层积云通常低于强对流风险，但仍需核查云底高、能见度、轻微降水及零度层；在低温含水云中同时评估机体积冰可能性。"
        )

    report.warnings.append(
        "普通 RGB 图像不能直接给出云顶高度、降水强度、风场、温度或积冰条件。"
    )


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
        report.recommendations.extend(
            [
                "未可靠定位机场主体，无法评估周边目标与机场的相对位置。请核对影像覆盖范围、方向与分辨率。",
                "如需开展净空筛查，应补充机场基准点、跑道端坐标、障碍物限制面和最新地形/建筑高度数据。",
            ]
        )
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
        nearby_peak = max(item.confidence for item in nearby)
        report.recommendations.extend(
            [
                f"机场周边关注区域内发现需复核目标：{'、'.join(labels)}，最高置信度 {nearby_peak:.1%}。该结果属于异常线索，不直接构成违建判定。",
                "建议人工核对目标位置，将其叠加到机场障碍物限制面和净空保护区图层，判断建筑高度、施工设备与跑道端及进离场方向的关系。",
                "建议安排针对性巡查，重点关注临时吊机、灯光遮蔽、鸟类吸引源、施工扬尘和夜间警示灯等运行影响。",
                "如目标可能侵入净空面或影响导航、目视助航设施，应升级安全评估，并根据影响范围考虑航行通告或临时运行限制。",
            ]
        )
        report.warnings.append(
            "单幅 RGB 图像无法测量建筑高度或获知审批状态，不能据此直接认定违建。"
        )
    else:
        report.recommendations.extend(
            [
                "当前阈值下未发现进入机场周边关注区域的高置信度建筑群或施工区域；该结果不排除小型、被遮挡或尚未纳入训练分布的目标。",
                "维持周期性影像对比和地面巡视，并将新建、扩建及临时施工信息与规划许可台账交叉核验。",
            ]
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
        report.recommendations.extend(
            [
                "未可靠分割出跑道，无法判断道面是否需要清理；请确认画面包含完整跑道并提高图像清晰度。",
                "在获得可靠分割前，不应依据本次结果降低巡检频次或恢复跑道使用。",
            ]
        )
        return

    if contamination_area <= 0:
        report.recommendations.extend(
            [
                "当前阈值下未分割出积雪或疑似积水；仍需按机场运行规程完成道面巡检，确认排水、标志标线和灯光状态。",
                "降雪、融雪或强降雨期间应提高检查频次，并结合跑道状况报告、制动效应和气象变化持续评估。",
            ]
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
                f"疑似污染区域约占跑道掩膜的 {ratio:.1%}，最高条件置信度 {max_condition_conf:.1%}，建议将该跑道列为优先现场检查与处置对象。",
                "根据污染类型组织清理，包括除雪、扫雪、除冰或排水作业；作业前后记录污染物种类、覆盖范围和深度，并检查排水口、道肩及低洼区。",
                "处置后按机场程序重新检查并发布跑道状况报告；必要时评估制动效应、更新航行通告，在授权人员确认前不得仅凭图像结果恢复正常使用。",
            ]
        )
    elif ratio >= review_ratio or max_condition_conf >= concern_conf:
        report.recommendations.append(
            f"检测到需要复核的道面区域（相对面积约 {ratio:.1%}，最高条件置信度 {max_condition_conf:.1%}）。建议定点巡检，核实污染物种类、深度、连续性及对轮迹区的影响。"
        )
    else:
        report.recommendations.append(
            "检测区域较小或置信度较低，先对标注区域进行人工复核；结合近期降水/降雪、排水状态和道面反光情况决定是否清理。"
        )
    if snow_area > 0:
        report.knowledge.append("积雪会遮挡标志标线并降低轮胎—道面摩擦，融雪后还可能形成再冻结或局部积水。")
    if water_area > 0:
        report.knowledge.append("积水可能增加滑水风险；风险与水深、速度、轮胎状态和道面纹理等因素共同相关。")
    report.warnings.append(
        "RGB 图像不能测量积雪或积水深度、摩擦系数和制动效应；阴影、反光与深色铺装可能造成混淆。"
    )


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
