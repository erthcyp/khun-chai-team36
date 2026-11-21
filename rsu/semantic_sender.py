"""Bounded delivery queue for RSU semantic updates."""

import asyncio
import json
import logging
from time import perf_counter

import httpx

LOG = logging.getLogger(__name__)


class SemanticSender:
    def __init__(self, client: httpx.AsyncClient, queue_size: int = 32) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive.")
        self._client = client
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self.delivered_bytes = 0
        self.failed_messages = 0
        self.last_rtt_ms: float | None = None

    def submit(self, payload: dict) -> bool:
        """Enqueue without waiting. Returns False when the queue is full."""
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self._queue.put_nowait(body)
        except asyncio.QueueFull:
            return False
        return True

    async def run(self) -> None:
        while True:
            body = await self._queue.get()
            try:
                await self._send(body)
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                self.failed_messages += 1
                LOG.warning("Could not deliver a semantic update: %s", type(exc).__name__)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def _send(self, body: bytes) -> None:
        for attempt in range(3):
            try:
                started = perf_counter()
                response = await self._client.post(
                    "/api/v1/semantic",
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                self.last_rtt_ms = (perf_counter() - started) * 1000
                self.delivered_bytes += len(body)
                return
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if attempt == 2 or (status not in (408, 429) and status < 500):
                    raise
            except httpx.RequestError:
                if attempt == 2:
                    raise
            await asyncio.sleep(0.15 * (attempt + 1))
