"""Independent delivery queue for selected JPEG previews."""

import asyncio
from collections import OrderedDict
import json
import logging

import httpx

from .semantic_sender import delivery_rejection

LOG = logging.getLogger(__name__)


class FrameSender:
    def __init__(self, client: httpx.AsyncClient, queue_size: int = 6) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive.")
        self._client = client
        self._capacity = queue_size
        self._pending: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._ready = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self.rejected_previews = 0
        self.delivered_jpeg_bytes = 0
        self.dropped_previews = 0
        self.failed_previews = 0

    def submit(self, jpeg: bytes, metadata: dict) -> None:
        """Enqueue a JPEG and its frame metadata, replacing a stale queued preview."""
        meta = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        camera = metadata.get("camera_id", "")
        if camera in self._pending:
            self.dropped_previews += 1
        elif len(self._pending) == self._capacity:
            self.dropped_previews += 1
            return
        self._pending[camera] = (jpeg, meta)
        self._idle.clear()
        self._ready.set()

    async def run(self) -> None:
        while True:
            await self._ready.wait()
            _, (jpeg, meta) = self._pending.popitem(last=False)
            if not self._pending:
                self._ready.clear()
            try:
                response = await self._client.post(
                    "/api/v1/snapshot",
                    data={"meta": meta},
                    files={"file": ("preview.jpg", jpeg, "image/jpeg")},
                )
                response.raise_for_status()
                if delivery_rejection(response):
                    self.rejected_previews += 1
                    LOG.warning("TCC rejected a snapshot")
                else:
                    self.delivered_jpeg_bytes += len(jpeg)
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                self.failed_previews += 1
                LOG.warning("Could not deliver a preview: %s", type(exc).__name__)
            finally:
                if not self._pending:
                    self._idle.set()

    async def drain(self) -> None:
        await self._idle.wait()
