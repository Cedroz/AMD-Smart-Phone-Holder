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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REMOTE_DIR = Path(__file__).resolve().parent
STATIC_DIR = REMOTE_DIR / "static"
sys.path.insert(0, str(REMOTE_DIR / "vendor"))

try:
    import segno
except ImportError:
    segno = None

MODES = ("ambient", "spotlight")
GRIPPER_ACTIONS = ("release", "grip")
MOVE_COMMANDS = ("up", "down", "left", "right", "resume")  # D-pad, and resume tracking afterwards

_state_lock = threading.Lock()
_state = {"mode": None, "held": True, "deactivated": False, "changed_at": time.time()}
_public_url = ""
_tracker_url = "http://127.0.0.1:8000"


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


def apply_gripper(action):
    """Forward release/grip to the tracker, which drives the gripper. Returns (HTTP status, body)."""
    status, body = tracker_request("/commands", {"command": action})
    print(f"[remote] gripper {action} -> tracker {status}", flush=True)
    return status, body


def tracker_request(path, payload=None, timeout=2.0):
    """Call the tracker's API (tracker/track.py). Returns (HTTP status, JSON body)."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(_tracker_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return 502, {"error": f"tracker not reachable at {_tracker_url}"}


def tracker_summary():
    status, body = tracker_request("/status", timeout=0.5)
    if status != 200:
        return {"reachable": False}
    return {"reachable": True, "mode": body.get("mode"), "dry_run": body.get("dry_run"),
            "held": body.get("held"), "deactivated": body.get("deactivated"), "frozen": body.get("frozen"),
            "voice": body.get("voice"), "motion_blocker": body.get("motion_blocker"),
            "last_command": body.get("last_command")}


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
            tracker = tracker_summary()
            with _state_lock:
                # The tracker is the source of truth for the gripper and the master switch.
                for key in ("held", "deactivated"):
                    if tracker["reachable"] and tracker.get(key) is not None:
                        _state[key] = tracker[key]
                self._send_json(200, {**_state, "tracker": tracker})
        else:
            self._send_json(404, {"error": "not found"})

    def _read_field(self, name):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}").get(name)

    def do_POST(self):
        if self.path == "/api/test-motion":
            status, body = tracker_request("/commands", {"command": "test_motion"})
            if status == 422:
                body = {"error": "this tracker has no test_motion command; restart tracker/track.py"}
            print(f"[remote] test_motion -> tracker {status}", flush=True)
            self._send_json(status, body)
            return
        if self.path == "/api/move":
            try:
                command = self._read_field("direction")
            except (ValueError, AttributeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if command not in MOVE_COMMANDS:
                self._send_json(400, {"error": f"direction must be one of {', '.join(MOVE_COMMANDS)}"})
                return
            status, body = tracker_request("/commands", {"command": command}, timeout=1.0)
            if status == 422:
                body = {"error": "this tracker has no D-pad; restart tracker/track.py"}
            self._send_json(status, body if status == 202 or "error" in body else {"error": str(body.get("detail", body))})
            return
        if self.path == "/api/active":
            try:
                active = self._read_field("active")
            except (ValueError, AttributeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if not isinstance(active, bool):
                self._send_json(400, {"error": "active must be true or false"})
                return
            command = "activate" if active else "deactivate"
            status, body = tracker_request("/commands", {"command": command})
            print(f"[remote] {command} -> tracker {status}", flush=True)
            if status == 422:
                body = {"error": "this tracker has no master switch; restart tracker/track.py"}
            if status != 202:
                self._send_json(status, body if "error" in body else {"error": str(body.get("detail", body))})
                return
            with _state_lock:
                _state.update(deactivated=not active, changed_at=time.time())
                self._send_json(200, dict(_state))
            return
        if self.path not in ("/api/mode", "/api/gripper"):
            self._send_json(404, {"error": "not found"})
            return
        field, allowed = ("mode", MODES) if self.path == "/api/mode" else ("action", GRIPPER_ACTIONS)
        try:
            value = self._read_field(field)
        except (ValueError, AttributeError):
            self._send_json(400, {"error": "invalid JSON"})
            return
        if value not in allowed:
            self._send_json(400, {"error": f"{field} must be one of {', '.join(allowed)}"})
            return
        if field == "mode":
            apply_mode(value)
            update = {"mode": value}
        else:
            status, body = apply_gripper(value)
            if status != 202:
                self._send_json(status, body if "error" in body else {"error": str(body.get("detail", body))})
                return
            update = {"held": value == "grip"}
        with _state_lock:
            _state.update(update, changed_at=time.time())
            self._send_json(200, dict(_state))

    def log_message(self, format, *args):
        if not self.path.startswith("/api/status"):
            super().log_message(format, *args)


def main():
    global _public_url, _tracker_url
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--url", help="address encoded in the QR code; defaults to this machine's LAN IP")
    parser.add_argument("--tracker-url", default=_tracker_url, help="tracker/track.py API (its --preview-port)")
    args = parser.parse_args()
    _public_url = args.url or f"http://{lan_ip()}:{args.port}/"
    _tracker_url = args.tracker_url.rstrip("/")
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
