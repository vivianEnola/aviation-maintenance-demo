from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_app_renders_manual_upload_without_exception(self) -> None:
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
        ).run(timeout=60)
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.title[0].value, "航卫智眼👀")
        self.assertEqual(app.segmented_control[0].value, "manual")


if __name__ == "__main__":
    unittest.main()
