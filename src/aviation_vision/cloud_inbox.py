from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supabase import Client, create_client


@dataclass(slots=True)
class QueueItem:
    id: str
    device_id: str
    storage_path: str
    original_name: str
    sha256: str
    created_at: str


class SupabaseInbox:
    def __init__(
        self,
        *,
        url: str,
        key: str,
        bucket: str = "mmsstv-images",
        table: str = "image_queue",
    ) -> None:
        self.client: Client = create_client(url, key)
        self.bucket = bucket
        self.table = table

    def pending(self, *, device_id: str | None = None, limit: int = 20) -> list[QueueItem]:
        query = (
            self.client.table(self.table)
            .select("id,device_id,storage_path,original_name,sha256,created_at")
            .eq("status", "pending")
            .order("created_at")
            .limit(limit)
        )
        if device_id:
            query = query.eq("device_id", device_id)
        rows = query.execute().data or []
        return [QueueItem(**row) for row in rows]

    def download(self, item: QueueItem) -> bytes:
        data = self.client.storage.from_(self.bucket).download(item.storage_path)
        if not isinstance(data, bytes):
            return bytes(data)
        return data

    def mark_processed(self, item_id: str, *, result: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"status": "processed"}
        if result is not None:
            payload["result"] = result
        self.client.table(self.table).update(payload).eq("id", item_id).execute()

    def mark_failed(self, item_id: str, error: str) -> None:
        self.client.table(self.table).update(
            {"status": "failed", "error_message": error[:1000]}
        ).eq("id", item_id).execute()
