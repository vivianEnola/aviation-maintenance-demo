from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AviationMaintenanceCourseProject/0.1 (educational dataset builder)"
CLASS_QUERIES = {
    "cumulonimbus": ["cumulonimbus satellite", "cumulonimbus from above"],
    "stratocumulus": ["stratocumulus satellite", "stratocumulus from above"],
    "typhoon_vortex": ["tropical cyclone satellite", "typhoon satellite image"],
}
CLASS_IDS = {"cumulonimbus": 0, "stratocumulus": 1, "typhoon_vortex": 2}
ALLOWED_LICENSE_MARKERS = ("public domain", "cc0", "cc by", "creative commons")
MANIFEST_FIELDS = [
    "image",
    "suggested_class",
    "group_id",
    "source_title",
    "source_page",
    "source_image",
    "license",
    "license_url",
    "artist",
    "sha256",
    "review_status",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Wikimedia Commons API 收集带许可元数据的云图候选集。"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--thumbnail-width", type=int, default=1024)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument(
        "--full-frame-prelabels",
        action="store_true",
        help="为主体占满画面的候选图生成全图弱框；仍须人工审核。",
    )
    return parser.parse_args()


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else str(value)


def acceptable_license(short_name: str, usage_terms: str) -> bool:
    combined = f"{short_name} {usage_terms}".lower()
    return any(marker in combined for marker in ALLOWED_LICENSE_MARKERS)


def search(session: requests.Session, query: str, width: int, limit: int) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(50, max(10, limit)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": width,
        "origin": "*",
    }
    response = session.get(API_URL, params=params, timeout=(10, 60))
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return list(pages.values())


def safe_stem(title: str) -> str:
    title = re.sub(r"^File:", "", title, flags=re.IGNORECASE)
    stem = Path(title).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:80] or "cloud"


def load_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, str]] = []
    for row in rows:
        source_title = row.get("source_title", "")
        normalized.append(
            {
                field: str(row.get(field, ""))
                for field in MANIFEST_FIELDS
            }
        )
        normalized[-1]["group_id"] = row.get("group_id") or safe_stem(source_title)
        normalized[-1]["review_status"] = (
            row.get("review_status") or row.get("annotation_status") or "needs_review"
        )
    return normalized


def download_with_retry(
    session: requests.Session, url: str, request_delay: float, attempts: int = 3
) -> bytes:
    for attempt in range(attempts):
        response = session.get(url, timeout=(10, 60))
        if response.status_code != 429:
            response.raise_for_status()
            return response.content
        retry_after = response.headers.get("Retry-After", "")
        try:
            wait_seconds = float(retry_after)
        except ValueError:
            wait_seconds = max(2.0, request_delay) * (attempt + 1)
        if attempt + 1 < attempts:
            print(f"  请求受限，{wait_seconds:.1f} 秒后重试…", flush=True)
            time.sleep(wait_seconds)
    response.raise_for_status()
    raise RuntimeError("unreachable")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    label_dir = output.parent / "prelabels"
    if args.full_frame_prelabels:
        label_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    manifest = output / "commons_cloud_manifest.csv"
    rows = load_existing_manifest(manifest)
    seen_hashes = {row["sha256"] for row in rows if row.get("sha256")}

    for label, queries in CLASS_QUERIES.items():
        accepted = sum(row.get("suggested_class") == label for row in rows)
        for query in queries:
            if accepted >= args.per_class:
                break
            print(f"查询 {label}: {query}", flush=True)
            try:
                pages = search(session, query, args.thumbnail_width, args.per_class * 3)
            except requests.RequestException as exc:
                print(f"查询失败：{exc}", flush=True)
                continue
            for page in pages:
                if accepted >= args.per_class:
                    break
                image_info = (page.get("imageinfo") or [{}])[0]
                metadata = image_info.get("extmetadata", {})
                license_name = metadata_value(metadata, "LicenseShortName")
                usage_terms = metadata_value(metadata, "UsageTerms")
                if not acceptable_license(license_name, usage_terms):
                    continue
                url = image_info.get("thumburl") or image_info.get("url")
                if not url:
                    continue
                try:
                    content = download_with_retry(session, url, args.request_delay)
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in seen_hashes:
                        continue
                    with Image.open(io.BytesIO(content)) as image:
                        if image.width < 256 or image.height < 256:
                            continue
                        rgb = image.convert("RGB")
                    filename = f"{label}_{safe_stem(str(page.get('title', 'cloud')))}_{digest[:8]}.jpg"
                    target = output / filename
                    rgb.save(target, format="JPEG", quality=92, optimize=True)
                    if args.full_frame_prelabels:
                        (label_dir / f"{target.stem}.txt").write_text(
                            f"{CLASS_IDS[label]} 0.500000 0.500000 1.000000 1.000000\n",
                            encoding="utf-8",
                        )
                    rows.append(
                        {
                            "image": str(target),
                            "suggested_class": label,
                            "group_id": safe_stem(str(page.get("title", "cloud"))),
                            "source_title": str(page.get("title", "")),
                            "source_page": str(image_info.get("descriptionurl", "")),
                            "source_image": str(image_info.get("url", "")),
                            "license": license_name or usage_terms,
                            "license_url": metadata_value(metadata, "LicenseUrl"),
                            "artist": metadata_value(metadata, "Artist"),
                            "sha256": digest,
                            "review_status": "needs_review",
                            "review_notes": "",
                        }
                    )
                    seen_hashes.add(digest)
                    accepted += 1
                    print(f"  已收集 {accepted}/{args.per_class}: {filename}", flush=True)
                    time.sleep(max(0.0, args.request_delay))
                except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
                    print(f"  跳过候选：{exc}", flush=True)
                finally:
                    time.sleep(max(0.0, args.request_delay))

    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["suggested_class"] == label for row in rows) for label in CLASS_QUERIES}
    print(f"完成：{counts}；所有图片均需人工审核。清单：{manifest}")


if __name__ == "__main__":
    main()
