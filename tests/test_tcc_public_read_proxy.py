import unittest

import httpx

from tcc.public_read_proxy import create_app


class PublicReadProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_dashboard_reads_reach_the_receiver(self):
        requests = []

        async def receiver(request):
            requests.append(request)
            if request.url.path.endswith("/CAM_01"):
                return httpx.Response(200, content=b"jpeg", headers={"Content-Type": "image/jpeg"})
            return httpx.Response(200, json={"status": "ok"})

        app = create_app(transport=httpx.MockTransport(receiver))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as api,
        ):
            for path in ("/health", "/api/v1/cameras", "/api/v1/alerts", "/api/v1/metrics",
                         "/api/v1/snapshots/CAM_01"):
                response = await api.get(path, headers={"Authorization": "test", "Cookie": "x=y"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")

            for path in ("/api/v1/semantic", "/api/v1/snapshot", "/api/v1/cameras"):
                self.assertIn((await api.post(path)).status_code, (404, 405))
            for path in ("/dashboard/", "/docs", "/api/v1/snapshots/CAM_99"):
                self.assertEqual((await api.get(path)).status_code, 404)

        self.assertEqual(len(requests), 5)
        self.assertTrue(all(request.method == "GET" for request in requests))
        self.assertTrue(all("authorization" not in request.headers and "cookie" not in request.headers
                            for request in requests))

    async def test_receiver_errors_do_not_leak_internal_details(self):
        async def receiver(request):
            if request.url.path.endswith("CAM_01"):
                return httpx.Response(404, text="internal detail")
            raise httpx.ConnectError("internal address", request=request)

        app = create_app(transport=httpx.MockTransport(receiver))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as api,
        ):
            for path, status in (("/health", 502), ("/api/v1/snapshots/CAM_01", 404)):
                response = await api.get(path)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.content, b"")
