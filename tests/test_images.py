from __future__ import annotations

import unittest
from io import BytesIO

from PIL import Image

from src.aviation_vision.images import ImageValidationError, validate_image_bytes


def png_bytes(width: int = 16, height: int = 12) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_valid_png_is_normalized_to_rgb(self) -> None:
        result = validate_image_bytes(
            png_bytes(), max_pixels=1000, allowed_formats={"PNG"}
        )
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual((result.width, result.height), (16, 12))
        self.assertEqual(len(result.sha256), 64)

    def test_corrupt_file_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(
                b"not an image", max_pixels=1000, allowed_formats={"PNG"}
            )

    def test_oversized_image_is_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(
                png_bytes(50, 50), max_pixels=1000, allowed_formats={"PNG"}
            )


if __name__ == "__main__":
    unittest.main()

