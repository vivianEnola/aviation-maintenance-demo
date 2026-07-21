from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("uploader.toml")
STATE_FILE = Path(__file__).with_name(".uploader_state.json")


@dataclass(slots=True)
class UploaderConfig:
    watch_folder: Path
    device_id: str
    bucket: str
    table: str
    scan_interval_seconds: float
    stable_wait_seconds: float
    extensions: set[str]


def load_config(
    path: Path,
    *,
    watch_folder_override: Path | None = None,
    device_id_override: str | None = None,
) -> UploaderConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    watch_folder = (
        watch_folder_override
        if watch_folder_override is not None
        else Path(str(raw["watch_folder"]))
    ).expanduser().resolve()
    if not watch_folder.is_dir():
        raise FileNotFoundError(f"监听文件夹不存在：{watch_folder}")
    device_id = str(device_id_override or raw["device_id"]).strip()
    if not device_id:
        raise ValueError("device_id 不能为空。")
    return UploaderConfig(
        watch_folder=watch_folder,
        device_id=device_id,
        bucket=str(raw.get("bucket", "mmsstv-images")),
        table=str(raw.get("table", "image_queue")),
        scan_interval_seconds=max(1.0, float(raw.get("scan_interval_seconds", 5))),
        stable_wait_seconds=max(0.5, float(raw.get("stable_wait_seconds", 2))),
        extensions={str(item).lower() for item in raw.get("extensions", [])},
    )


def load_state() -> set[str]:
    if not STATE_FILE.is_file():
        return set()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(item) for item in payload.get("uploaded_sha256", [])}


def save_state(hashes: set[str]) -> None:
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"uploaded_sha256": sorted(hashes)}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(STATE_FILE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_stable(path: Path, wait_seconds: float) -> bool:
    try:
        before = path.stat()
        time.sleep(wait_seconds)
        after = path.stat()
    except OSError:
        return False
    return before.st_size > 0 and (
        before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    )


def connect() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "请设置 SUPABASE_URL 和 SUPABASE_SERVICE_ROLE_KEY 环境变量。"
        )
    return create_client(url, key)


def already_queued(client: Client, table: str, device_id: str, digest: str) -> bool:
    rows = (
        client.table(table)
        .select("id")
        .eq("device_id", device_id)
        .eq("sha256", digest)
        .limit(1)
        .execute()
        .data
        or []
    )
    return bool(rows)


def upload_file(
    client: Client,
    config: UploaderConfig,
    path: Path,
    digest: str,
) -> str:
    storage_path = f"{config.device_id}/{digest}{path.suffix.lower()}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        client.storage.from_(config.bucket).upload(
            storage_path,
            handle.read(),
            file_options={"content-type": content_type, "upsert": "false"},
        )
    client.table(config.table).insert(
        {
            "device_id": config.device_id,
            "storage_path": storage_path,
            "original_name": path.name,
            "sha256": digest,
            "status": "pending",
        }
    ).execute()
    return storage_path


def candidates(config: UploaderConfig) -> list[Path]:
    return sorted(
        (
            path
            for path in config.watch_folder.iterdir()
            if path.is_file() and path.suffix.lower() in config.extensions
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )


def run(config: UploaderConfig, *, once: bool = False, dry_run: bool = False) -> None:
    client = None if dry_run else connect()
    uploaded = load_state()
    print(f"监听目录：{config.watch_folder}")
    print(f"设备 ID：{config.device_id}")

    while True:
        for path in candidates(config):
            if not is_stable(path, config.stable_wait_seconds):
                continue
            digest = file_sha256(path)
            if digest in uploaded:
                continue
            try:
                if dry_run:
                    print(f"[DRY RUN] 将上传：{path.name} sha256={digest}")
                elif client is not None and already_queued(
                    client, config.table, config.device_id, digest
                ):
                    print(f"已在队列，跳过：{path.name}")
                elif client is not None:
                    remote = upload_file(client, config, path, digest)
                    print(f"上传成功：{path.name} -> {remote}")
                # A dry run must never mark a file as uploaded.  Otherwise a
                # later real run silently skips the file because of the local
                # de-duplication state.
                if not dry_run:
                    uploaded.add(digest)
                    save_state(uploaded)
            except Exception as exc:
                print(f"上传失败：{path.name}: {exc}", file=sys.stderr)
        if once:
            return
        time.sleep(config.scan_interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监听 MMSSTV 图片目录并上传到 Supabase。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--watch-folder", type=Path, help="覆盖配置文件中的监听目录。")
    parser.add_argument("--device-id", help="覆盖配置文件中的设备 ID。")
    parser.add_argument("--once", action="store_true", help="扫描一次后退出。")
    parser.add_argument("--dry-run", action="store_true", help="不连接云端，仅显示将上传的文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        load_config(
            args.config.resolve(),
            watch_folder_override=args.watch_folder,
            device_id_override=args.device_id,
        ),
        once=args.once,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
