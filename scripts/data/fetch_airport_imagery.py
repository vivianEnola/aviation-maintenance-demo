from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
from affine import Affine
from PIL import Image
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


OURAIRPORTS_URL = (
    "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
)
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
MANIFEST_FIELDS = [
    "image",
    "airport_ident",
    "airport_name",
    "group_id",
    "stac_item",
    "cloud_cover",
    "license",
    "review_status",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按公开机场坐标从 Planetary Computer 获取 Sentinel-2 清晰 RGB 裁片。"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--countries", nargs="*", default=[])
    parser.add_argument("--airport-types", nargs="*", default=["large_airport", "medium_airport"])
    parser.add_argument("--date-range", default="2023-01-01/2026-12-31")
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="最多尝试的机场数量，避免服务异常时无限跳过候选。",
    )
    return parser.parse_args()


def fetch_airports() -> list[dict[str, str]]:
    print("正在下载 OurAirports 坐标清单…", flush=True)
    response = requests.get(OURAIRPORTS_URL, timeout=(10, 60))
    response.raise_for_status()
    lines = response.content.decode("utf-8-sig").splitlines()
    rows = list(csv.DictReader(lines))
    print(f"已读取 {len(rows)} 条机场记录。", flush=True)
    return rows


def bbox_around(latitude: float, longitude: float, radius_km: float) -> list[float]:
    lat_delta = radius_km / 111.32
    lon_scale = max(0.1, math.cos(math.radians(latitude)))
    lon_delta = radius_km / (111.32 * lon_scale)
    return [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    ]


def select_item(
    catalog: pystac_client.Client,
    bbox: list[float],
    date_range: str,
    max_cloud: float,
) -> Any | None:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=20,
    )
    items = list(search.item_collection())
    if not items:
        return None
    return min(items, key=lambda item: float(item.properties.get("eo:cloud_cover", 100)))


def to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    output = np.zeros(data.shape, dtype=np.uint8)
    for index in range(data.shape[0]):
        band = data[index].astype(np.float32)
        valid = band[np.isfinite(band)]
        if not valid.size:
            continue
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            high = low + 1
        output[index] = np.clip((band - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return output


def crop_visual(item: Any, bbox: list[float], size: int) -> tuple[Image.Image, dict[str, Any]]:
    signed = planetary_computer.sign(item)
    asset = signed.assets.get("visual")
    if asset is None:
        raise KeyError(f"STAC item {item.id} 缺少 visual 资源。")
    with rasterio.open(asset.href) as dataset:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", dataset.crs, *bbox, densify_pts=21
        )
        window = from_bounds(left, bottom, right, top, dataset.transform)
        window = window.round_offsets().round_lengths()
        data = dataset.read(
            indexes=list(range(1, min(dataset.count, 3) + 1)),
            window=window,
            out_shape=(min(dataset.count, 3), size, size),
            resampling=rasterio.enums.Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        transform = dataset.window_transform(window) * Affine.scale(
            window.width / size, window.height / size
        )
        metadata = {
            "crs": dataset.crs.to_string(),
            "transform": list(transform)[:6],
            "search_bbox_wgs84": bbox,
            "source_asset": asset.href.split("?", 1)[0],
            "stac_item": item.id,
            "collection": item.collection_id,
        }
    data = to_uint8(data)
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    image = Image.fromarray(np.moveaxis(data[:3], 0, -1), mode="RGB")
    return image, metadata


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned.strip("_") or "airport"


def load_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append({field: str(row.get(field, "")) for field in MANIFEST_FIELDS})
        normalized[-1]["group_id"] = row.get("group_id") or row.get("airport_ident", "")
        normalized[-1]["review_status"] = (
            row.get("review_status") or row.get("annotation_status") or "needs_review"
        )
    return normalized


def recover_manifest_from_sidecars(
    output: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    known_images = {Path(row["image"]).name for row in rows if row.get("image")}
    for sidecar in sorted(output.glob("*.geo.json")):
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        image_path = next(
            (
                candidate
                for suffix in IMAGE_EXTENSIONS
                if (candidate := output / f"{sidecar.name[:-9]}{suffix}").is_file()
            ),
            None,
        )
        if image_path is None or image_path.name in known_images:
            continue
        airport_ident = str(metadata.get("airport_ident", ""))
        rows.append(
            {
                "image": str(image_path),
                "airport_ident": airport_ident,
                "airport_name": str(metadata.get("airport_name", "")),
                "group_id": airport_ident or image_path.stem,
                "stac_item": str(metadata.get("stac_item", "")),
                "cloud_cover": str(metadata.get("cloud_cover", "")),
                "license": "Copernicus Sentinel data",
                "review_status": "needs_review",
                "review_notes": "recovered_from_geo_sidecar",
            }
        )
        known_images.add(image_path.name)
    return rows


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "airport_imagery_manifest.csv"
    manifest_rows = recover_manifest_from_sidecars(
        output, load_existing_manifest(manifest)
    )
    existing_airports = {row["airport_ident"] for row in manifest_rows}
    if len(manifest_rows) >= args.count:
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"已有 {len(manifest_rows)} 张，达到目标数量；清单：{manifest}")
        return
    airports = [
        row
        for row in fetch_airports()
        if row.get("type") in set(args.airport_types)
        and row.get("latitude_deg")
        and row.get("longitude_deg")
        and (not args.countries or row.get("iso_country") in set(args.countries))
    ]
    random.Random(args.seed).shuffle(airports)
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )
    attempts = 0
    for airport in airports:
        if len(manifest_rows) >= args.count:
            break
        if airport.get("ident", "") in existing_airports:
            continue
        if attempts >= max(args.count, args.max_attempts):
            break
        attempts += 1
        latitude = float(airport["latitude_deg"])
        longitude = float(airport["longitude_deg"])
        bbox = bbox_around(latitude, longitude, args.radius_km)
        print(
            f"尝试 {attempts}/{max(args.count, args.max_attempts)}："
            f"{airport.get('ident')} {airport.get('name')}",
            flush=True,
        )
        try:
            item = select_item(catalog, bbox, args.date_range, args.max_cloud)
            if item is None:
                continue
            image, metadata = crop_visual(item, bbox, args.image_size)
            stem = safe_name(f"{airport.get('ident', '')}_{airport.get('name', '')}")
            image_path = output / f"{stem}.png"
            sidecar = output / f"{stem}.geo.json"
            metadata.update(
                {
                    "airport_ident": airport.get("ident"),
                    "airport_name": airport.get("name"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "cloud_cover": item.properties.get("eo:cloud_cover"),
                    "image_width": image.width,
                    "image_height": image.height,
                }
            )
            image.save(image_path, format="PNG", optimize=True)
            sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_rows.append(
                {
                    "image": str(image_path),
                    "airport_ident": str(airport.get("ident", "")),
                    "airport_name": str(airport.get("name", "")),
                    "group_id": str(airport.get("ident", "")),
                    "stac_item": str(item.id),
                    "cloud_cover": str(item.properties.get("eo:cloud_cover", "")),
                    "license": "Copernicus Sentinel data",
                    "review_status": "needs_review",
                    "review_notes": "",
                }
            )
            existing_airports.add(str(airport.get("ident", "")))
            print(
                f"已获取 {len(manifest_rows)}/{args.count}: {airport.get('name')}",
                flush=True,
            )
        except Exception as exc:
            print(f"跳过 {airport.get('ident')}: {exc}", flush=True)

    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"完成：{len(manifest_rows)} 张；清单：{manifest}")


if __name__ == "__main__":
    main()
