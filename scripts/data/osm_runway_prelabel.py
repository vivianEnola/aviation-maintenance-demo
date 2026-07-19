from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests
from affine import Affine
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import LineString, Polygon, box


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 OpenStreetMap runway 几何为机场裁片生成 Labelme 弱标签。"
    )
    parser.add_argument("images", type=Path, help="包含 PNG 与 .geo.json 的目录。")
    parser.add_argument("output", type=Path, help="Labelme JSON 输出目录。")
    parser.add_argument("--default-width-m", type=float, default=45.0)
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def overpass_runways(bbox: list[float]) -> list[dict[str, Any]]:
    west, south, east, north = bbox
    query = (
        "[out:json][timeout:60];"
        f'way["aeroway"="runway"]({south},{west},{north},{east});'
        "out tags geom;"
    )
    errors: list[str] = []
    headers = {"User-Agent": "AviationMaintenanceCourseProject/0.1"}
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.get(
                endpoint,
                params={"data": query},
                headers=headers,
                timeout=(10, 90),
            )
            response.raise_for_status()
            return list(response.json().get("elements", []))
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("所有 Overpass 端点均失败：" + " | ".join(errors))


def runway_polygon(
    element: dict[str, Any],
    *,
    crs: str,
    image_transform: Affine,
    image_width: int,
    image_height: int,
    default_width_m: float,
) -> Polygon | None:
    geometry = element.get("geometry", [])
    if len(geometry) < 2:
        return None
    longitudes = [float(point["lon"]) for point in geometry]
    latitudes = [float(point["lat"]) for point in geometry]
    projected_x, projected_y = transform_coordinates(
        "EPSG:4326", crs, longitudes, latitudes
    )
    inverse = ~image_transform
    pixels = [inverse * (x, y) for x, y in zip(projected_x, projected_y, strict=True)]
    tags = element.get("tags", {})
    width_text = str(tags.get("width", default_width_m)).split()[0]
    try:
        width_m = float(width_text)
    except ValueError:
        width_m = default_width_m

    if len(pixels) >= 4 and pixels[0] == pixels[-1]:
        polygon = Polygon(pixels)
    else:
        meters_per_pixel = max(
            0.1, (abs(image_transform.a) + abs(image_transform.e)) / 2
        )
        half_width_pixels = max(1.0, width_m / meters_per_pixel / 2)
        polygon = LineString(pixels).buffer(half_width_pixels, cap_style="flat")

    clipped = polygon.intersection(box(0, 0, image_width, image_height))
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda part: part.area)
    return clipped if isinstance(clipped, Polygon) and clipped.area >= 4 else None


def labelme_payload(
    image_path: Path,
    metadata: dict[str, Any],
    runways: list[dict[str, Any]],
    default_width_m: float,
) -> dict[str, Any]:
    image_transform = Affine(*metadata["transform"])
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    shapes = []
    for runway in runways:
        polygon = runway_polygon(
            runway,
            crs=str(metadata["crs"]),
            image_transform=image_transform,
            image_width=width,
            image_height=height,
            default_width_m=default_width_m,
        )
        if polygon is None:
            continue
        shapes.append(
            {
                "label": "runway",
                "points": [[round(x, 2), round(y, 2)] for x, y in polygon.exterior.coords[:-1]],
                "group_id": None,
                "description": "OSM weak label — needs manual review",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main() -> None:
    args = parse_args()
    images = args.images.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    converted = skipped = failed = 0
    review_rows: list[dict[str, str]] = []

    for metadata_path in sorted(images.glob("*.geo.json")):
        stem = metadata_path.name[: -len(".geo.json")]
        image_path = images / f"{stem}.png"
        target = output / f"{stem}.json"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            runways = overpass_runways(metadata["search_bbox_wgs84"])
            payload = labelme_payload(
                image_path, metadata, runways, args.default_width_m
            )
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            converted += 1
            review_rows.append(
                {
                    "image": str(image_path),
                    "labelme": str(target),
                    "group_id": str(metadata.get("airport_ident", stem)),
                    "shapes": str(len(payload["shapes"])),
                    "review_status": "needs_review",
                    "error": "",
                }
            )
            print(f"{image_path.name}: {len(payload['shapes'])} 条跑道弱标签")
        except RuntimeError as exc:
            failed += 1
            review_rows.append(
                {
                    "image": str(image_path),
                    "labelme": "",
                    "group_id": str(metadata.get("airport_ident", stem)),
                    "shapes": "0",
                    "review_status": "fetch_failed",
                    "error": str(exc),
                }
            )
            print(f"{image_path.name}: 查询失败，已记录并继续：{exc}")
        finally:
            time.sleep(max(0.0, args.request_delay))

    manifest = output / "runway_prelabel_manifest.csv"
    if review_rows:
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)
    print(
        f"生成 {converted} 个 Labelme JSON，失败 {failed} 个，"
        f"跳过 {skipped} 个已有文件；清单：{manifest}"
    )


if __name__ == "__main__":
    main()
