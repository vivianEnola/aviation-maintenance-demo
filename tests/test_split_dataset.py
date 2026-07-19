from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.data.split_dataset import (
    assign_groups,
    copy_records,
    discover_records,
    validate_ratios,
)


class SplitDatasetTests(unittest.TestCase):
    def test_grouped_yolo_split_keeps_source_group_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            for group in range(8):
                for variant in ("original", "sstv"):
                    stem = f"source{group}_{variant}"
                    Image.new("RGB", (8, 8)).save(images / f"{stem}.png")
                    (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            manifest = root / "manifest.csv"
            manifest.write_text(
                "image,group_id,review_status\n"
                + "".join(
                    f"source{group}_{variant}.png,source{group},approved\n"
                    for group in range(8)
                    for variant in ("original", "sstv")
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                task="yolo",
                images=images,
                labels=labels,
                output=root / "output",
                manifest=manifest,
                image_column="image",
                group_column="group_id",
                status_column="review_status",
                approved_value="approved",
                require_approved=True,
                allow_missing_labels=False,
                overwrite=False,
            )

            records = discover_records(args)
            ratios = validate_ratios(0.70, 0.15, 0.15)
            assignments = assign_groups(records, ratios, seed=42)
            copy_records(records, assignments, args)

            for group in range(8):
                split = assignments[f"source{group}"]
                self.assertTrue(
                    (args.output / "images" / split / f"source{group}_original.png").is_file()
                )
                self.assertTrue(
                    (args.output / "images" / split / f"source{group}_sstv.png").is_file()
                )
            self.assertTrue((args.output / "split_manifest.csv").is_file())

    def test_ratios_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            validate_ratios(0.8, 0.2, 0.2)


if __name__ == "__main__":
    unittest.main()
