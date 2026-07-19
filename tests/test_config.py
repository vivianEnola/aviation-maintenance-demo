from __future__ import annotations

import unittest

from src.aviation_vision.config import load_yaml, mode_config, model_config


class ConfigTests(unittest.TestCase):
    def test_all_modes_reference_known_models(self) -> None:
        models = load_yaml("models.yaml")["models"]
        modes = load_yaml("models.yaml")["modes"]
        for mode in modes.values():
            if "model" in mode:
                self.assertIn(mode["model"], models)
            if "classifier" in mode:
                self.assertIn(mode["classifier"], models)
                for target in mode.get("routes", {}).values():
                    if target is not None:
                        self.assertIn(target, models)

    def test_public_helpers_find_entries(self) -> None:
        self.assertEqual(model_config("cloud_detector")["task"], "detect")
        self.assertEqual(mode_config("runway")["model"], "runway_segmenter")


if __name__ == "__main__":
    unittest.main()
