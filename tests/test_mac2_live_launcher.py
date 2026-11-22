"""Verify one interrupt stops the Mac 2 services."""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Mac2LiveLauncherTest(unittest.TestCase):
    def test_sigint_stops_receiver_proxy_and_tunnel(self):
        with socket.socket() as tcc, socket.socket() as proxy:
            for listener in (tcc, proxy):
                listener.bind(("127.0.0.1", 0))
            tcc_port = tcc.getsockname()[1]
            proxy_port = proxy.getsockname()[1]

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tunnel_pid = folder / "tunnel.pid"
            fake = folder / "cloudflared"
            fake.write_text(f'#!/bin/sh\nprintf "%s" "$$" > "{tunnel_pid}"\nexec sleep 60\n')
            fake.chmod(0o700)
            with (folder / "launcher.log").open("w+") as log:
                launcher = subprocess.Popen([
                    sys.executable, str(ROOT / "deploy/vercel/run_mac2.py"),
                    "--host", "127.0.0.1", "--port", str(tcc_port),
                    "--proxy-port", str(proxy_port), "--cloudflared", str(fake),
                ], cwd=ROOT, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline and not tunnel_pid.exists() and launcher.poll() is None:
                        time.sleep(0.1)
                    log.seek(0)
                    self.assertTrue(tunnel_pid.exists(), log.read())
                    pid = int(tunnel_pid.read_text())
                finally:
                    if launcher.poll() is None:
                        launcher.send_signal(signal.SIGINT)
                    launcher.wait(timeout=10)
                self.assertEqual(launcher.returncode, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                for port in (tcc_port, proxy_port):
                    with socket.socket() as client:
                        self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)
