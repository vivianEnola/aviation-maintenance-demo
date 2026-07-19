from __future__ import annotations

import unittest

from src.aviation_vision.advisors import apply_advice
from src.aviation_vision.schemas import AnalysisReport, VisionObject


RULES = {
    "display_names": {
        "cumulonimbus": "积雨云",
        "airport": "机场",
        "construction_area": "施工区域",
        "runway": "跑道",
        "snow": "积雪",
    },
    "weather_knowledge": {"cumulonimbus": "强对流知识"},
}

THRESHOLDS = {
    "airport": {"attention_scale": 2.0, "concern_confidence": 0.55},
    "runway": {
        "review_area_ratio": 0.005,
        "action_area_ratio": 0.05,
        "concern_confidence": 0.50,
    },
}


class AdvisorTests(unittest.TestCase):
    def test_cloud_advice_mentions_convection(self) -> None:
        report = AnalysisReport(
            requested_mode="cloud",
            executed_model="cloud_detector",
            task="detect",
            objects=[VisionObject("cumulonimbus", 0.9)],
        )
        apply_advice(report, RULES, THRESHOLDS)
        self.assertIn("积雨云", report.summary)
        self.assertTrue(any("绕飞" in item for item in report.recommendations))

    def test_airport_nearby_construction_is_flagged_without_illegal_claim(self) -> None:
        report = AnalysisReport(
            requested_mode="airport",
            executed_model="airport_detector",
            task="detect",
            objects=[
                VisionObject("airport", 0.95, (100, 100, 300, 300)),
                VisionObject("construction_area", 0.80, (320, 150, 360, 220)),
            ],
        )
        apply_advice(report, RULES, THRESHOLDS)
        combined = " ".join(report.recommendations)
        self.assertIn("人工核对", combined)
        self.assertNotIn("存在违建", combined)

    def test_runway_area_ratio_drives_action(self) -> None:
        report = AnalysisReport(
            requested_mode="runway",
            executed_model="runway_segmenter",
            task="segment",
            objects=[
                VisionObject("runway", 0.95, mask_area_ratio=0.20),
                VisionObject("snow", 0.80, mask_area_ratio=0.02),
            ],
        )
        apply_advice(report, RULES, THRESHOLDS)
        self.assertAlmostEqual(report.metadata["condition_to_runway_ratio"], 0.1)
        self.assertTrue(any("清理" in item for item in report.recommendations))


if __name__ == "__main__":
    unittest.main()

