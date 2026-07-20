from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_uploader.watch_folder import load_config


class LocalUploaderConfigTests(unittest.TestCase):
    def test_cli_values_override_config_folder_and_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = root / "configured"
            override = root / "override"
            configured.mkdir()
            override.mkdir()
            config_path = root / "uploader.toml"
            config_path.write_text(
                f'watch_folder = "{configured.as_posix()}"\n'
                'device_id = "configured-device"\n'
                'extensions = [".png"]\n',
                encoding="utf-8",
            )

            config = load_config(
                config_path,
                watch_folder_override=override,
                device_id_override="command-line-device",
            )

            self.assertEqual(config.watch_folder, override.resolve())
            self.assertEqual(config.device_id, "command-line-device")
            self.assertEqual(config.extensions, {".png"})


if __name__ == "__main__":
    unittest.main()
