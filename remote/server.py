"""Serve the mobile arm controller and receive its button presses over HTTP.

Run on the SSH machine that drives the hardware, then scan the QR code it
prints (or open http://<machine-ip>:8090) on a phone on the same network.
"""

import argparse
import html
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REMOTE_DIR = Path(__file__).resolve().parent
STATIC_DIR = REMOTE_DIR / "static"
sys.path.insert(0, str(REMOTE_DIR / "vendor"))

try:
    import segno
except ImportError:
    segno = None

MODES = ("deactivate", "ambient", "spotlight")

_state_lock = threading.Lock()
_state = {"mode": "deactivate", "changed_at": time.time()}
_public_url = ""


def lan_ip():
    """Return the address other devices use to reach this machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))  # no packet is sent for UDP connect
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def qr_page(url):
    svg = segno.make(url, error="m").svg_inline(scale=10, border=2, dark="#000", light="#fff")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arm Remote QR</title>
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0e1014; color: #f2f3f5;
         font: 16px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; }}
  .card {{ padding: 24px; }}
  .qr {{ background: #fff; padding: 16px; border-radius: 20px; display: inline-block; }}
  .qr svg {{ width: min(70vmin, 420px); height: auto; display: block; }}
  h1 {{ font-size: 22px; margin: 20px 0 4px; }}
  p {{ margin: 0; color: #9aa1ad; }}
</style></head>
<body><div class="card">
  <div class="qr">{svg}</div>
  <h1>Scan to open the arm remote</h1>
  <p>{html.escape(url)}</p>
</div></body></html>"""


def apply_mode(mode):
    """Hook for the hardware side. Not wired to the arm yet."""
    print(f"[remote] mode -> {mode}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "FollowMeRemote/0.1"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/qr":
            if segno is None:
                self._send_json(503, {"error": "QR support missing; see remote/requirements.txt"})
                return
            body = qr_page(_public_url).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            with _state_lock:
                self._send_json(200, dict(_state))
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/mode":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            mode = json.loads(self.rfile.read(length) or b"{}").get("mode")
        except (ValueError, AttributeError):
            self._send_json(400, {"error": "invalid JSON"})
            return
        if mode not in MODES:
            self._send_json(400, {"error": f"mode must be one of {', '.join(MODES)}"})
            return
        apply_mode(mode)
        with _state_lock:
            _state.update(mode=mode, changed_at=time.time())
            self._send_json(200, dict(_state))

    def log_message(self, format, *args):
        if not self.path.startswith("/api/status"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--url", help="address encoded in the QR code; defaults to this machine's LAN IP")
    args = parser.parse_args()
    global _public_url
    _public_url = args.url or f"http://{lan_ip()}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if segno is None:
        print("QR support missing; install it with:\n"
              f"  python3 -m pip install --no-deps --target {REMOTE_DIR / 'vendor'} -r {REMOTE_DIR / 'requirements.txt'}")
    else:
        print("\nScan with your phone camera:\n")
        segno.make(_public_url, error="m").terminal(compact=True)
        print(f"\nShow this QR on a bigger screen: {_public_url}qr")
    print(f"Arm controller at {_public_url}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
