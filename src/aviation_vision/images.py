from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(slots=True)
class ValidatedImage:
    image: Image.Image
    sha256: str
    format: str
    width: int
    height: int


class ImageValidationError(ValueError):
    pass


def validate_image_bytes(
    content: bytes,
    *,
    max_pixels: int,
    allowed_formats: set[str],
) -> ValidatedImage:
    if not content:
        raise ImageValidationError("图片内容为空。")

    try:
        with Image.open(BytesIO(content)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("文件不是可读取的图片或图片已损坏。") from exc

    if image_format not in allowed_formats:
        raise ImageValidationError(f"不支持的图片格式：{image_format or '未知'}。")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ImageValidationError(
            f"图片尺寸 {width}×{height} 超出限制，最大允许 {max_pixels:,} 像素。"
        )

    with Image.open(BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()

    return ValidatedImage(
        image=image,
        sha256=sha256(content).hexdigest(),
        format=image_format,
        width=image.width,
        height=image.height,
    )
