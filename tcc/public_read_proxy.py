"""Public read-only relay for the TCC dashboard. Bind to loopback for tunneling."""

from contextlib import asynccontextmanager
import os
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import httpx


def create_app(*, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        origin = os.environ.get("ROADSENSE_TCC_ORIGIN", "http://127.0.0.1:8010")
        async with httpx.AsyncClient(base_url=origin, timeout=3, transport=transport) as client:
            app.state.tcc = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    async def read_tcc(request: Request, path: str, media_type: str, version: str | None = None) -> Response:
        try:
            upstream = await request.app.state.tcc.get(path, params={"v": version} if version is not None else None)
        except httpx.RequestError:
            return Response(status_code=502, headers={"Cache-Control": "no-store"})
        if upstream.status_code == 404:
            return Response(status_code=404, headers={"Cache-Control": "no-store"})
        if upstream.status_code != 200 or upstream.headers.get(
            "content-type", ""
        ).split(";", 1)[0].strip().lower() != media_type:
            return Response(status_code=502, headers={"Cache-Control": "no-store"})
        return Response(upstream.content, media_type=media_type, headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    async def read_json(request: Request) -> Response:
        return await read_tcc(request, request.url.path, "application/json")

    for path in ("/health", "/api/v1/cameras", "/api/v1/alerts", "/api/v1/metrics"):
        app.add_api_route(path, read_json, methods=["GET"])

    @app.get("/api/v1/snapshots/{camera_id}")
    async def read_snapshot(request: Request, camera_id: str, v: str | None = None) -> Response:
        if not re.fullmatch(r"CAM_0[1-6]", camera_id):
            raise HTTPException(404)
        return await read_tcc(request, f"/api/v1/snapshots/{camera_id}", "image/jpeg", v)

    return app


app = create_app()
