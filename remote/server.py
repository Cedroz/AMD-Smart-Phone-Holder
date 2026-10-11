"""Serve the mobile arm controller and receive its button presses and microphone audio.

Run on the SSH machine that drives the hardware, then scan the QR code it
prints (or open https://<machine-ip>:8090) on a phone on the same network.
HTTPS uses a self-signed certificate made on first run: phones only allow
microphone access on secure pages, so accept the browser's warning once.
"""

import argparse
import fcntl
import html
import json
import shutil
import socket
import ssl
import subprocess
import sys
import termios
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
VOICE_INPUT = ("127.0.0.1", 8765)  # acoustic/voice_bridge.py's audio input (TCPAudioCapture)
MAX_AUDIO_BYTES = 64000            # 2 s of 16 kHz 16-bit mono per request


class PhoneMic:
    """Forward the phone's microphone (16 kHz 16-bit mono PCM chunks) into the voice listener.

    The voice listener takes one audio stream at a time on VOICE_INPUT, the same way it takes the laptop
    mic sender's. The connection is closed once the phone stops sending, so the listener drops the last
    window instead of replaying it.
    """

    IDLE_SECONDS = 2.0
    MAX_UNREAD = 64000  # bytes (2 s) the listener hasn't taken: it's busy with another microphone

    def __init__(self, address=VOICE_INPUT):
        self.address = address
        self.sock = None
        self.last = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._close_when_idle, daemon=True).start()

    def send(self, pcm):
        """Returns None on success, or a message saying why the audio couldn't be delivered."""
        with self.lock:
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(self.address, timeout=1.0)
                    self.sock.settimeout(1.0)
                self.sock.sendall(pcm)
                self.last = time.monotonic()
                # The listener serves one microphone at a time. If it isn't reading ours, the kernel just
                # queues it; notice that instead of building up stale audio.
                unread = int.from_bytes(fcntl.ioctl(self.sock, termios.TIOCOUTQ, bytes(4)), "little")
                if unread > self.MAX_UNREAD:
                    self._close()
                    return "another microphone is already connected (stop the laptop mic sender)"
                return None
            except ConnectionRefusedError:
                self._close()
                return "the voice listener isn't running (start ./robot)"
            except socket.timeout:
                self._close()
                return "another microphone is already connected (stop the laptop mic sender)"
            except OSError as error:
                self._close()
                return f"couldn't reach the voice listener ({error})"

    def active(self):
        with self.lock:
            return self.sock is not None and time.monotonic() - self.last < self.IDLE_SECONDS

    def _close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _close_when_idle(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                if self.sock is not None and time.monotonic() - self.last > self.IDLE_SECONDS:
                    self._close()


_phone_mic = PhoneMic()


def ensure_certificate(ip):
    """Self-signed certificate for this machine's address, made with openssl on first use."""
    folder = REMOTE_DIR / "certs"
    cert, key, made_for = folder / "app.crt", folder / "app.key", folder / "address.txt"
    if cert.exists() and key.exists() and made_for.exists() and made_for.read_text().strip() == ip:
        return cert, key
    openssl = shutil.which("openssl") or next((p for p in ("/usr/bin/openssl", "/home/aup/miniforge3/bin/openssl")
                                              if Path(p).exists()), None)
    if openssl is None:
        raise RuntimeError("openssl not found; run with --http (the phone mic then won't work)")
    folder.mkdir(exist_ok=True)
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
                    "-subj", "/CN=FollowMe arm remote", "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    key.chmod(0o600)
    made_for.write_text(ip)
    return cert, key


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
    """Ambient (sweep West-North-East) or Spotlight (follow people), applied by the tracker."""
    status, body = tracker_request("/commands", {"command": mode})
    print(f"[remote] mode {mode} -> tracker {status}", flush=True)
    return status, body


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
            "voice_following": body.get("voice_following"), "voice_face_missing": body.get("voice_face_missing"),
            "faces": body.get("faces") or [], "finding": body.get("finding"), "seeking": body.get("seeking"),
            "view_mode": body.get("view_mode"), "follow_mode": body.get("follow_mode"),
            "active_target": body.get("active_target"), "orientation": body.get("orientation"),
            "target_visible": body.get("target_visible"),
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

    def _relay_camera(self):
        """Pass the tracker's live camera (MJPEG at /stream) through, so the HTTPS page can show it."""
        try:
            upstream = urllib.request.urlopen(_tracker_url + "/stream", timeout=5)
        except (urllib.error.URLError, OSError) as error:
            self._send_json(503, {"error": f"camera not available from the tracker ({error})"})
            return
        with upstream:
            self.send_response(200)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
            except (OSError, ssl.SSLError, ValueError):
                pass  # the phone closed the page or hid the camera, or the tracker stopped

    def do_GET(self):
        if self.path.split("?")[0] == "/api/stream":
            self._relay_camera()
        elif self.path in ("/", "/index.html"):
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
                if tracker["reachable"] and tracker.get("view_mode") in MODES:
                    _state["mode"] = tracker["view_mode"]
                _state["phone_mic"] = _phone_mic.active()
                self._send_json(200, {**_state, "tracker": tracker})
        else:
            self._send_json(404, {"error": "not found"})

    def _read_field(self, name):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}").get(name)

    def do_POST(self):
        if self.path == "/api/audio":
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_AUDIO_BYTES or length % 2:
                self._send_json(400, {"error": f"send 16-bit PCM chunks of up to {MAX_AUDIO_BYTES} bytes"})
                return
            problem = _phone_mic.send(self.rfile.read(length))
            if problem:
                self._send_json(503, {"error": problem})
            else:
                self.send_response(204)
                self.end_headers()
            return
        if self.path == "/api/test-motion":
            status, body = tracker_request("/commands", {"command": "test_motion"})
            if status == 422:
                body = {"error": "this tracker has no test_motion command; restart tracker/track.py"}
            print(f"[remote] test_motion -> tracker {status}", flush=True)
            self._send_json(status, body)
            return
        if self.path == "/api/orientation":
            try:
                orientation = self._read_field("orientation")
            except (ValueError, AttributeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if orientation not in ("portrait", "landscape"):
                self._send_json(400, {"error": "orientation must be portrait or landscape"})
                return
            status, body = tracker_request("/commands", {"command": orientation})
            print(f"[remote] orientation {orientation} -> tracker {status}", flush=True)
            if status == 422:
                body = {"error": "this tracker can't rotate the phone; restart ./robot"}
            self._send_json(status, body if status == 202 or "error" in body else {"error": str(body.get("detail", body))})
            return
        if self.path == "/api/follow":
            try:
                mode = self._read_field("mode")
            except (ValueError, AttributeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if mode not in ("manual", "speaker"):
                self._send_json(400, {"error": "mode must be manual or speaker"})
                return
            status, body = tracker_request("/follow", {"mode": mode})
            print(f"[remote] follow {mode} -> tracker {status}", flush=True)
            if status == 200:
                self._send_json(200, {"follow_mode": body.get("follow_mode", mode)})
            else:
                self._send_json(status, body if "error" in body else {"error": str(body.get("detail", body))})
            return
        if self.path == "/api/find":
            try:
                name = self._read_field("name")
            except (ValueError, AttributeError):
                self._send_json(400, {"error": "invalid JSON"})
                return
            if name is not None and not isinstance(name, str):
                self._send_json(400, {"error": "name must be a string or null"})
                return
            status, body = tracker_request("/find", {"name": name})
            print(f"[remote] go to {name} -> tracker {status}", flush=True)
            if status == 404:
                body = {"error": "this tracker has no Go to; restart tracker/track.py"}
            self._send_json(status, body if status == 202 or "error" in body else {"error": str(body.get("detail", body))})
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
            status, body = apply_mode(value)
            if status == 422:
                body = {"error": "this tracker has no Ambient mode; restart tracker/track.py"}
            if status != 202:
                self._send_json(status, body if "error" in body else {"error": str(body.get("detail", body))})
                return
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
        if not self.path.startswith(("/api/status", "/api/audio", "/api/stream")):  # frequent or long-lived
            super().log_message(format, *args)


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Phones open spare connections and drop them, and http:// visits fail the TLS handshake:
        # both are harmless, so don't print tracebacks for them.
        error = sys.exc_info()[1]
        if isinstance(error, (ssl.SSLError, ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def main():
    global _public_url, _tracker_url
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--url", help="address encoded in the QR code; defaults to this machine's LAN IP")
    parser.add_argument("--tracker-url", default=_tracker_url, help="tracker/track.py API (its --preview-port)")
    parser.add_argument("--http", action="store_true", help="plain HTTP (phones then refuse the microphone)")
    parser.add_argument("--voice-input", type=int, default=VOICE_INPUT[1],
                        help="local port of the voice listener's audio input (acoustic config MIC_TCP_PORT)")
    args = parser.parse_args()
    _phone_mic.address = ("127.0.0.1", args.voice_input)
    ip = lan_ip()
    scheme = "http" if args.http else "https"
    _public_url = args.url or f"{scheme}://{ip}:{args.port}/"
    _tracker_url = args.tracker_url.rstrip("/")
    server = AppServer((args.host, args.port), Handler)
    if not args.http:
        cert, key = ensure_certificate(ip)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        # Handshake in each request's thread, so one slow phone can't stall the others.
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    if segno is None:
        print("QR support missing; install it with:\n"
              f"  python3 -m pip install --no-deps --target {REMOTE_DIR / 'vendor'} -r {REMOTE_DIR / 'requirements.txt'}")
    else:
        print("\nScan with your phone camera:\n")
        segno.make(_public_url, error="m").terminal(compact=True)
        print(f"\nShow this QR on a bigger screen: {_public_url}qr")
    print(f"Arm controller at {_public_url}  (Ctrl+C to stop)", flush=True)
    if not args.http:
        print("First visit: the phone warns the certificate isn't trusted - choose Advanced / Show details, "
              "then continue to the site. That's expected for this self-signed certificate.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
