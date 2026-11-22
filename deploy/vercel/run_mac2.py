"""Run the Mac 2 live dashboard services as one foreground command."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def serve(module, host, port, env):
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module, "--host", host, "--port", str(port)],
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )


def require_available(host, port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Cannot bind {host}:{port}: {exc}") from exc


def wait_for_health(url, services):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        require_running(services)
        try:
            with opener.open(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    raise RuntimeError(f"Health check timed out at {url}")


def require_running(services):
    for name, process in services:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{name} exited ({code})")


def shutdown(services):
    for _, process in reversed(services):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for _, process in reversed(services):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Mac 2 LAN IP used by the RSU")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--proxy-port", type=int, default=8012)
    parser.add_argument("--cloudflared", default=os.getenv("CLOUDFLARED", "cloudflared"))
    parser.add_argument("--tunnel-config", help="Cloudflare config for an existing named tunnel")
    parser.add_argument("--tunnel-name", help="Existing Cloudflare tunnel name or ID")
    args = parser.parse_args()

    if bool(args.tunnel_config) != bool(args.tunnel_name):
        parser.error("--tunnel-config and --tunnel-name must be used together")
    if not (cloudflared := shutil.which(args.cloudflared)):
        parser.error("cloudflared is required. Install it or pass --cloudflared /path/to/cloudflared")

    tcc_url = f"http://{args.host}:{args.port}"
    proxy_url = f"http://127.0.0.1:{args.proxy_port}"
    if args.tunnel_config:
        config = Path(args.tunnel_config).expanduser().resolve()
        if not config.is_file():
            parser.error(f"Tunnel config does not exist: {config}")
        tunnel = [cloudflared, "tunnel", "--config", str(config), "--no-autoupdate", "run", args.tunnel_name]
    else:
        tunnel = [cloudflared, "tunnel", "--no-autoupdate", "--url", proxy_url]

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    services = []
    try:
        require_available(args.host, args.port)
        require_available("127.0.0.1", args.proxy_port)
        services.append(("TCC", serve("tcc.pc2_receiver:app", args.host, args.port, os.environ)))
        wait_for_health(tcc_url, services)
        proxy_env = {**os.environ, "ROADSENSE_TCC_ORIGIN": tcc_url, "NO_PROXY": args.host, "no_proxy": args.host}
        services.append(("Read-only proxy", serve("tcc.public_read_proxy:app", "127.0.0.1", args.proxy_port, proxy_env)))
        wait_for_health(proxy_url, services)
        services.append(("Cloudflare Tunnel", subprocess.Popen(tunnel, cwd=ROOT, start_new_session=True)))
        print(f"TCC dashboard: {tcc_url}/dashboard/  |  Press Ctrl+C to stop the live feed.", flush=True)
        if not args.tunnel_name:
            print("Quick Tunnel URLs change each launch. Update Vercel rewrites to use the new URL.", flush=True)
        while True:
            require_running(services)
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        shutdown(services)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, previous_term)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
