"""Independent delivery queue for selected JPEG previews."""

import asyncio
import json
import logging

import httpx

LOG = logging.getLogger(__name__)


class FrameSender:
    def __init__(self, client: httpx.AsyncClient, queue_size: int = 2) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive.")
        self._client = client
        self._queue: asyncio.Queue[tuple[bytes, str]] = asyncio.Queue(maxsize=queue_size)
        self.delivered_jpeg_bytes = 0
        self.dropped_previews = 0
        self.failed_previews = 0

    def submit(self, jpeg: bytes, metadata: dict) -> None:
        """Enqueue a JPEG and its frame metadata, replacing a stale queued preview."""
        meta = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.dropped_previews += 1
        self._queue.put_nowait((jpeg, meta))

    async def run(self) -> None:
        while True:
            jpeg, meta = await self._queue.get()
            try:
                response = await self._client.post(
                    "/api/v1/snapshot",
                    data={"meta": meta},
                    files={"file": ("preview.jpg", jpeg, "image/jpeg")},
                )
                response.raise_for_status()
                self.delivered_jpeg_bytes += len(jpeg)
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                self.failed_previews += 1
                LOG.warning("Could not deliver a preview: %s", type(exc).__name__)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()
