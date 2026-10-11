#!/usr/bin/env python
"""Person tracker for an SO-101 follower arm holding a phone.

The camera must be mounted on the arm so it turns with the phone. Each frame we
find the person, measure how far they are from the image center, and nudge
shoulder_pan (left/right) and optionally wrist_flex (up/down) to re-center them.

Run on the arm, inside the `lerobot` conda env:

    python tracker/track.py --dry-run          # camera + detection only, motors untouched
    python tracker/track.py --test-pan         # wiggle the base once to check direction
    python tracker/track.py --test-tilt        # same for the up/down wrist joint
    python tracker/track.py                    # track (Ctrl+C to stop)

Live preview: open http://<arm-ip>:8000 in a browser on your laptop.
"""

import argparse
import glob
import http.server
import logging
import os
import queue
import signal
import socketserver
import sys
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("tracker")

FOLLOWER_ID = "follower_arm"
# Stable name for the follower's servo board (ttyACM0/1 can swap after a replug).
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_MODEL = os.path.join(MODELS_DIR, "yolov8n_320.onnx")
FACE_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
PAN = "shoulder_pan"
TILT = "wrist_flex"
GRIP = "gripper"
# Joints whose servo load jumps when someone grabs and tugs the phone.
PULL_JOINTS = ["wrist_flex", "wrist_roll", "gripper"]
# Joints that get pushed out of position when someone pulls the phone (these servos give way rather
# than pushing back hard, so displacement is often a clearer signal than load).
PUSH_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# SO-101 model shipped with the LeRobot move_to example on the arm; needs `placo` (installed in the lerobot env).
DEFAULT_URDF = os.path.expanduser("~/lerobot/examples/so101_move_to/SO101/so101_new_calib.urdf")
IK_ITERS = 20
MAX_IK_JUMP_DEG = 5.0  # per tracking step


# --------------------------------------------------------------------------- camera


class Camera:
    """Reads frames on a background thread so we always process the newest one (low lag)."""

    def __init__(self, source, width=640, height=480, exposure=None, flip=False):
        self.flip = flip  # camera mounted upside down -> rotate 180 deg
        # Resolve stable names like /dev/v4l/by-id/... to the /dev/videoN they point at.
        src = os.path.realpath(source) if str(source).startswith("/dev/") else str(source)
        # OpenCV's V4L2 backend can't always open "/dev/videoN" by name, but always accepts the index N.
        if src.startswith("/dev/video") and src[len("/dev/video"):].isdigit():
            src = src[len("/dev/video"):]
        src = int(src) if src.isdigit() else src
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        # A camera left mid-stream by a crashed run can refuse the first open or two; keep trying briefly.
        for attempt in range(5):
            self.cap = cv2.VideoCapture(src, backend)
            if self.cap.isOpened():
                break
            self.cap.release()
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Could not open camera {source}. Check: fuser -v {source}; v4l2-ctl --list-devices. "
                "If nothing is using it, unplug the camera's USB for 5 s and plug it back in."
            )
        # MJPG is compressed, so two cameras fit on one USB 2.0 hub at full frame rate (raw YUYV doesn't).
        # V4L2 only honors the format if it's set after the size.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if exposure is not None:
            # Auto exposure lengthens frames in dim rooms (30 fps -> ~16). Manual: V4L2 units of 0.1 ms.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        log.info(f"Camera {source}: {fourcc} {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                 f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps requested")
        self.frame = None
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                if self.flip:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self.cond:
                    self.frame = frame
                    self.seq += 1
                    self.cond.notify_all()
            else:
                time.sleep(0.01)

    def read(self, last_seq, timeout=1.0):
        """Block until a frame newer than `last_seq` arrives. Returns (seq, frame) or (last_seq, None)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > last_seq, timeout):
                return last_seq, None
            return self.seq, self.frame.copy()

    def close(self):
        # Let the reader thread finish its current read before releasing the device; releasing it
        # mid-read crashes OpenCV ("terminate called without an active exception") and can leave the
        # camera stuck so the next run can't open it.
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


# --------------------------------------------------------------------------- detection


class PersonDetector:
    """OpenCV-only detector: frontal face, then profile face, then full body (HOG).

    Faces are the most reliable when you look at the phone; the body detector is
    the fallback for when you turn away (e.g. facing the stove).
    """

    def __init__(self, proc_width=320):
        self.proc_width = proc_width
        root = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(root + "haarcascade_profileface.xml")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Return a list of (x, y, w, h, kind) boxes in full-frame pixel coordinates."""
        scale = self.proc_width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        boxes = [(*b, "face") for b in self.frontal.detectMultiScale(gray, 1.1, 6, minSize=(20, 20))]
        if not boxes:
            boxes = [(*b, "profile") for b in self.profile.detectMultiScale(gray, 1.1, 6, minSize=(20, 20))]
            # The profile cascade only finds faces turned one way; flip to catch the other side.
            flipped = cv2.flip(gray, 1)
            for x, y, w, h in self.profile.detectMultiScale(flipped, 1.1, 6, minSize=(20, 20)):
                boxes.append((gray.shape[1] - x - w, y, w, h, "profile"))
        if not boxes:
            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            boxes = [(*r, "body") for r, wt in zip(rects, np.ravel(weights)) if wt > 0.5]

        return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale), k) for x, y, w, h, k in boxes]


class YoloDetector:
    """YOLOv8 person detector run through OpenCV's DNN module, so it needs no packages beyond OpenCV.

    Recognizes a person's whole shape, so it copes with blur, distance, and people facing away.
    """

    def __init__(self, model_path, size=320, conf=0.4, nms=0.45):
        # 320 px input: ~20 ms/frame on a laptop CPU vs ~200 ms at 640, and still finds people across a room.
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.size = size
        self.conf, self.nms = conf, nms

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        # Output is (1, 84, N): cx, cy, w, h, then 80 class scores. COCO class 0 is "person".
        out = self.net.forward()[0].T
        scores = out[:, 4]
        keep = scores > self.conf
        out, scores = out[keep], scores[keep]
        sx, sy = w / self.size, h / self.size
        rects = [
            [int((cx - bw / 2) * sx), int((cy - bh / 2) * sy), int(bw * sx), int(bh * sy)]
            for cx, cy, bw, bh in out[:, :4]
        ]
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.conf, self.nms)
        return [(*rects[i], "person") for i in np.ravel(idx)]


class YuNetFaceDetector:
    """OpenCV's YuNet face detector: small, fast, and much more accurate than the old Haar cascades."""

    def __init__(self, model_path, conf=0.85):
        # 0.85 rather than YuNet's usual 0.7: motion-blurred hands and arms were passing as faces.
        self.net = cv2.FaceDetectorYN.create(model_path, "", (320, 320), conf)

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.net.setInputSize((w, h))
        _, faces = self.net.detect(frame)
        if faces is None:
            return []
        # Largest first, then drop any face whose center falls inside one we've kept (stacked duplicates).
        boxes = sorted(((int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces), key=lambda b: -b[2] * b[3])
        kept = []
        for x, y, bw, bh in boxes:
            cx, cy = x + bw / 2, y + bh / 2
            if not any(kx <= cx <= kx + kw and ky <= cy <= ky + kh for kx, ky, kw, kh in kept):
                kept.append((x, y, bw, bh))
        return [(*b, "face") for b in kept]


class FaceAndBodyDetector:
    """Faces when you look toward the phone, whole-body boxes when you turn away.

    Each person yields one target: their face if it's visible, otherwise their body.
    """

    def __init__(self, yolo_path, face_path):
        self.bodies = YoloDetector(yolo_path)
        self.faces = YuNetFaceDetector(face_path)

    def detect(self, frame):
        faces = self.faces.detect(frame)
        bodies = self.bodies.detect(frame)

        def contains_face(body):
            x, y, w, h, _ = body
            return any(x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h for fx, fy, fw, fh, _ in faces)

        return faces + [b for b in bodies if not contains_face(b)]


def aim_point(box):
    """Where on the box we want the phone to point: the face center, or near the head for a body."""
    x, y, w, h, kind = box
    if kind == "person":
        return x + w / 2, y + h * 0.15
    if kind == "body":
        return x + w / 2, y + h * 0.2
    return x + w / 2, y + h / 2


def pick_target(boxes, prev, max_jump):
    """Stick with the person nearest to last frame's target; otherwise take the biggest one.

    If the nearest detection is more than `max_jump` px away, our person was probably just missed this
    frame, so return None (hold still) rather than swinging over to someone else.
    """
    if not boxes:
        return None
    if prev is not None:
        best = min(boxes, key=lambda b: np.hypot(*(np.subtract(aim_point(b), prev))))
        return best if np.hypot(*(np.subtract(aim_point(best), prev))) <= max_jump else None
    return max(boxes, key=lambda b: b[2] * b[3])


# --------------------------------------------------------------------------- preview server


PREVIEW_PAGE = b"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phone tracker</title><style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
img{max-width:100%;display:block;margin:0 auto}
button{font-size:1.2em;padding:.7em 1.6em;margin:.8em .4em;border:0;border-radius:.5em;cursor:pointer}
#rel{background:#e5534b;color:#fff}#grip{background:#46954a;color:#fff}#msg{min-height:1.4em}
</style></head><body><img src="/stream">
<button id="rel" onclick="send('release')">Release phone</button>
<button id="grip" onclick="send('grip')">Grip + resume</button><div id="msg"></div>
<script>function send(c){fetch('/'+c,{method:'POST'}).then(r=>r.text()).then(t=>msg.textContent=t)
.catch(()=>msg.textContent='Could not reach the tracker')}</script></body></html>"""


class Preview:
    """Serves a page at http://<host>:<port>/ with the annotated camera feed and Release/Grip buttons."""

    def __init__(self, port, on_command):
        self.jpeg = None
        self.version = 0
        self.cond = threading.Condition()
        preview = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                cmd = self.path.strip("/")
                if cmd in ("release", "grip"):
                    on_command(cmd)
                    body = f"{cmd} requested".encode()
                    self.send_response(200)
                else:
                    body = b"unknown command"
                    self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PREVIEW_PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                sent = 0
                try:
                    while True:
                        # Only send new frames; re-sending old ones over slow Wi-Fi piles up and causes lag.
                        with preview.cond:
                            preview.cond.wait_for(lambda: preview.version != sent, timeout=1.0)
                            jpeg, sent = preview.jpeg, preview.version
                        if jpeg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        log.info(f"Preview at http://<arm-ip>:{port}/")

    def show(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.version += 1
                self.cond.notify_all()


def annotate(frame, boxes, target, aim, deadband, status):
    h, w = frame.shape[:2]
    db = int(deadband * w / 2)
    cv2.line(frame, (w // 2 - db, 0), (w // 2 - db, h), (80, 80, 80), 1)
    cv2.line(frame, (w // 2 + db, 0), (w // 2 + db, h), (80, 80, 80), 1)
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for b in boxes:
        x, y, bw, bh, kind = b
        color = (0, 255, 0) if b is target else (0, 160, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, kind, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if aim is not None:
        cv2.circle(frame, (int(aim[0]), int(aim[1])), 6, (0, 0, 255), -1)
    cv2.putText(frame, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


# --------------------------------------------------------------------------- arm


def find_follower_port():
    """Each Feetech board stores its arm's calibration; the follower's matches follower_arm.json."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    for port in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        robot = SO101Follower(SO101FollowerConfig(port=port, id=FOLLOWER_ID))
        try:
            robot.bus.connect()
            if robot.bus.is_calibrated:
                log.info(f"Follower arm found on {port}")
                return port
        except Exception as e:
            log.debug(f"{port}: {e}")
        finally:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
    raise RuntimeError("No port matched the follower calibration. Pass --port explicitly (see lerobot-find-port).")


def other_users_of(device):
    """PIDs + command lines of other processes that have `device` open (Linux /proc scan)."""
    target = os.path.realpath(device)
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc") if os.path.isdir("/proc") else []):
        if int(pid) == os.getpid():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
            if any(os.path.realpath(f"/proc/{pid}/fd/{fd}") == target for fd in fds):
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    found.append(f"{pid}: {f.read().replace(b'\0', b' ').decode(errors='replace').strip()}")
        except OSError:
            continue
    return found


def connect_arm(port):
    # Two programs driving the arm at once fight over it (garbled packets, joints yanked both ways,
    # gripper commands from the other program). Refuse to start if anyone else has the port open.
    others = other_users_of(port)
    if others:
        raise RuntimeError(
            "Another program is already using the arm, so we'd be fighting it:\n  " + "\n  ".join(others)
            + "\nStop it first (kill <PID>), after checking with whoever started it."
        )

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        port=port,
        id=FOLLOWER_ID,
        # Keep holding the phone when the script exits instead of going limp and dropping it.
        disable_torque_on_disconnect=False,
        # Never let a single command jump a joint more than this many degrees.
        max_relative_target=10.0,
    )
    robot = SO101Follower(config)
    # Feetech buses occasionally return a garbled packet ("Incorrect status packet"); a retry usually clears it.
    for attempt in range(1, 4):
        try:
            robot.connect(calibrate=False)
            return robot
        except ConnectionError as e:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
            if attempt == 3:
                raise ConnectionError(
                    f"{e}\nStill failing after 3 tries. Is another program using the arm? "
                    "Check: fuser -v /dev/ttyACM*; pgrep -fa lerobot"
                ) from e
            log.warning(f"Connect attempt {attempt} failed ({e}); retrying...")
            time.sleep(0.5)


def disconnect_arm(robot, args):
    """Disconnect, leaving every motor on so the arm holds its pose and the gripper keeps holding the phone."""
    robot.disconnect()
    log.info("Arm and gripper still holding. To take the phone out: hold it, then run --free-gripper")


def free_gripper(robot, args):
    """Turn only the gripper motor off so it can be opened by hand. The rest of the arm keeps holding."""
    input("HOLD THE PHONE with your hand, then press ENTER to loosen the gripper...")
    robot.bus.disable_torque(GRIP, num_retry=3)
    print("Gripper motor off: open it by hand. The next tracker run turns it back on.")


def read_joints(robot):
    return {k.removesuffix(".pos"): v for k, v in robot.get_observation().items() if k.endswith(".pos")}


def test_pan(robot, degrees=10.0):
    start = read_joints(robot)[PAN]
    print(f"{PAN} is at {start:.1f} deg. Moving to {start + degrees:.1f} deg...")
    robot.send_action({f"{PAN}.pos": start + degrees})
    time.sleep(1.5)
    input("Which way did the phone turn (from the phone's point of view: its left or its right)? Press ENTER to return.")
    robot.send_action({f"{PAN}.pos": start})
    time.sleep(1.0)
    print(
        "If +pan turned the camera toward ITS RIGHT, use --pan-sign 1 (the default).\n"
        "If it turned toward ITS LEFT, use --pan-sign -1."
    )


def test_tilt(robot, degrees=10.0):
    start = read_joints(robot)[TILT]
    print(f"{TILT} is at {start:.1f} deg. Moving to {start + degrees:.1f} deg...")
    robot.send_action({f"{TILT}.pos": start + degrees})
    time.sleep(1.5)
    input("Did the camera tilt UP or DOWN? Press ENTER to return.")
    robot.send_action({f"{TILT}.pos": start})
    time.sleep(1.0)
    print(
        "If +tilt pointed the camera DOWN, use --tilt-sign 1 (the default).\n"
        "If it pointed the camera UP, use --tilt-sign -1."
    )


class HeightIK:
    """Raises/lowers the phone with shoulder_lift + elbow_flex + wrist_flex, keeping its forward reach and pitch.

    Solved in the arm's vertical plane with pan held at 0 (turning the base doesn't change that plane's
    geometry), so the base keeps doing left/right on its own. wrist_roll is held where it started.
    """

    def __init__(self, urdf, joints, frame="gripper_frame_link"):
        from lerobot.model.kinematics import RobotKinematics

        self.kin = RobotKinematics(urdf, frame, ARM_JOINTS)
        self.q = np.array([joints[j] for j in ARM_JOINTS], dtype=float)
        self.q[0] = 0.0
        self.home = self.kin.forward_kinematics(self.q).copy()
        self.z0 = float(self.home[2, 3])

    def solve(self, z):
        """Joints that put the tip at height z with the start reach and orientation.

        Returns (q, position error in m, orientation error in deg). Starts from the last accepted
        solution, so small height changes converge quickly.
        """
        pose = self.home.copy()
        pose[2, 3] = z
        q = self.q.copy()
        roll = q[4]
        for _ in range(IK_ITERS):
            q = self.kin.inverse_kinematics(q, pose, position_weight=1.0, orientation_weight=0.3)
            q[0], q[4] = 0.0, roll
        reached = self.kin.forward_kinematics(q)
        pos_err = float(np.linalg.norm(reached[:3, 3] - pose[:3, 3]))
        rel = reached[:3, :3].T @ pose[:3, :3]
        ang_err = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0))))
        return q, pos_err, ang_err


def plane_action(q, tilt_offset=0.0):
    return {
        "shoulder_lift.pos": float(q[1]),
        "elbow_flex.pos": float(q[2]),
        f"{TILT}.pos": float(q[3] + tilt_offset),
    }


def move_gripper(robot, start, end, seconds=1.0, steps=20):
    for s in range(1, steps + 1):
        robot.send_action({f"{GRIP}.pos": float(start + (end - start) * s / steps)})
        time.sleep(seconds / steps)


def gentle_squeeze(robot, args, start):
    """Close the gripper a little past where it is so it actually presses on the phone, gently.

    Safety layers: the gripper's live torque limit is lowered to --grip-torque % first (and nothing
    happens if that can't be confirmed), it closes in small slow steps, and it stops at the first
    sign of the phone pushing back. Returns the position it ends up holding.
    """
    limit = int(args.grip_torque * 10)  # register is 0-1000
    try:
        robot.bus.write("Torque_Limit", GRIP, limit, num_retry=3)
        confirmed = robot.bus.read("Torque_Limit", GRIP, num_retry=3)
    except ConnectionError as e:
        log.warning(f"Couldn't set the gripper force limit ({e}); not squeezing")
        return start
    if confirmed != limit:
        log.warning(f"Gripper force limit reads {confirmed}, expected {limit}; not squeezing")
        return start

    loads = read_loads(robot)
    base = loads[GRIP] if loads else None
    steps = 30
    pos = start
    touching = 0
    for i in range(1, steps + 1):
        pos = start - args.gripper_open_sign * args.squeeze * i / steps
        robot.send_action({f"{GRIP}.pos": float(pos)})
        time.sleep(1.5 / steps)
        loads = read_loads(robot)
        # Two readings in a row, so one noisy reading doesn't stop the squeeze before it grips.
        touching = touching + 1 if base is not None and loads and abs(loads[GRIP] - base) > args.contact_load else 0
        if touching >= 2:
            log.info(f"Gripper felt the phone after {abs(pos - start):.1f}; holding there")
            break
    log.info(f"Gripper holding the phone at {pos:.1f} with force capped at {args.grip_torque:g}%")
    return pos


def read_loads(robot):
    """Servo load per joint, -1000..1000 (tenths of a percent of max torque). None on a bus glitch."""
    try:
        return robot.bus.sync_read("Present_Load", PULL_JOINTS, num_retry=2)
    except ConnectionError:
        return None


class PullDetector:
    """Feels someone gripping and pulling the phone, even while the arm is moving.

    Signals (see read_pull_signals): servo load on the wrist/gripper, and how far the shoulder, elbow
    and wrist sit away from where we last told them to be. Each has its own threshold. The phone's
    weight and the normal lag while moving are tracked as a slowly adapting baseline, so only a
    sudden change counts. `level` is the strongest signal as a fraction of its threshold (1.0 = pull).
    """

    def __init__(self, thresholds, hold_s):
        self.thresholds, self.hold_s = thresholds, hold_s
        self.reset()

    def reset(self):
        self.baseline, self.since, self.level, self.strongest = None, None, 0.0, ""
        self.peak, self.peak_key = 0.0, ""

    def take_peak(self):
        """Strongest level (and its signal) since the last call, for the once-a-second log line."""
        peak, key = self.peak, self.peak_key
        self.peak, self.peak_key = 0.0, ""
        return peak, key

    def update(self, signals, now):
        """Returns (pulling, confirmed): pulling = over threshold now, confirmed = for hold_s seconds."""
        if self.baseline is None:
            self.baseline = {k: float(v) for k, v in signals.items()}
            return False, False
        ratios = {k: abs(signals[k] - self.baseline[k]) / self.thresholds[k.split(":")[0]] for k in signals}
        self.strongest = max(ratios, key=ratios.get)
        self.level = ratios[self.strongest]
        if self.level > self.peak:
            self.peak, self.peak_key = self.level, self.strongest
        if self.level > 1.0:
            if self.since is None:
                self.since = now
            return True, now - self.since >= self.hold_s
        self.since = None
        for k in signals:
            self.baseline[k] += 0.05 * (signals[k] - self.baseline[k])
        return False, False


def read_pull_signals(robot, goals):
    """{"load:<joint>": servo load, "push:<joint>": degrees away from its last goal}, or None on a glitch."""
    try:
        loads = robot.bus.sync_read("Present_Load", PULL_JOINTS, num_retry=2)
        pos = robot.bus.sync_read("Present_Position", PUSH_JOINTS, num_retry=2)
    except ConnectionError:
        return None
    signals = {f"load:{j}": float(v) for j, v in loads.items()}
    signals.update({f"push:{j}": float(pos[j] - goals[j]) for j in PUSH_JOINTS})
    return signals


def test_gripper(robot, args, amount=10.0):
    """Find which way opens the gripper, safely: the person holds the phone the whole time."""
    start = read_joints(robot)[GRIP]
    print(f"{GRIP} is at {start:.1f}.")
    input("HOLD THE PHONE FIRMLY with your hand, then press ENTER. The gripper will move a little...")
    move_gripper(robot, start, start + amount)
    input("Did the gripper LOOSEN (open) or SQUEEZE (close)? Keep holding; press ENTER to return.")
    move_gripper(robot, start + amount, start)
    print("If it LOOSENED, use --gripper-open-sign 1. If it SQUEEZED, use --gripper-open-sign -1.\n")
    print("Now TUG THE PHONE FIRMLY a few times, like you're taking it out (8 s). Watch both columns...")
    base = read_loads(robot)
    base_pos = robot.bus.sync_read("Present_Position", PULL_JOINTS, num_retry=2)
    t_end = time.time() + 8
    while time.time() < t_end:
        loads = read_loads(robot)
        try:
            pos = robot.bus.sync_read("Present_Position", PULL_JOINTS, num_retry=2)
        except ConnectionError:
            pos = None
        if loads and base and pos:
            print("  force: " + " ".join(f"{j}={loads[j] - base[j]:+5.0f}" for j in PULL_JOINTS)
                  + "  | pushed by: " + " ".join(f"{j}={pos[j] - base_pos[j]:+5.1f}" for j in PULL_JOINTS))
        time.sleep(0.25)
    print("Send these numbers: they show whether force or position works better for detecting a tug.")


def check_ik(robot, args):
    """No movement: show where the model thinks the tip is, so it can be checked with a ruler."""
    joints = read_joints(robot)
    hik = HeightIK(args.urdf, joints)
    tip = hik.home[:3, 3]
    print("Joints (deg):", {j: round(joints[j], 1) for j in ARM_JOINTS})
    print(f"Model says the gripper tip is {tip[0] * 100:.1f} cm FORWARD of the base axis and "
          f"{tip[2] * 100:.1f} cm ABOVE the base mounting surface (measured along the arm's own direction).")
    print("Measure it with a ruler. If either number is off by more than ~2 cm, the model doesn't match this arm.\n")
    for dz in (+0.03, -0.03):
        q, pos_err, ang_err = hik.solve(hik.z0 + dz)
        change = {j: round(float(q[i] - hik.q[i]), 1) for i, j in enumerate(ARM_JOINTS) if j in ("shoulder_lift", "elbow_flex", "wrist_flex")}
        print(f"Plan {dz * 100:+.0f} cm: joint changes {change} | error {pos_err * 1000:.1f} mm, {ang_err:.1f} deg")
    print("\nNothing moved. Small joint changes and errors under ~5 mm / 3 deg mean IK is behaving.")


def plan_height_path(hik, dz, step_m=0.005):
    """Joint waypoints for a straight vertical move of dz metres, or (None, reason) if it isn't smooth."""
    n = max(1, int(np.ceil(abs(dz) / step_m)))
    path, q_prev = [], hik.q.copy()
    for i in range(1, n + 1):
        q, pos_err, ang_err = hik.solve(hik.z0 + dz * i / n)
        jump = float(np.abs(q - q_prev)[1:4].max())
        if pos_err > 0.005 or ang_err > 3 or jump > 15:
            return None, (f"at {dz * i / n * 100:+.1f} cm: error {pos_err * 1000:.1f} mm / {ang_err:.1f} deg, "
                          f"a joint would jump {jump:.0f} deg")
        hik.q = q  # next solve starts here, so the path stays on one continuous arm pose
        path.append(q)
        q_prev = q
    return path, None


def drive(robot, q_from, path, deg_per_step=1.0):
    """Send the waypoints slowly: no joint moves more than deg_per_step per 50 ms."""
    prev = q_from
    for q in path:
        n = max(1, int(np.ceil(np.abs(q - prev)[1:4].max() / deg_per_step)))
        for s in range(1, n + 1):
            robot.send_action(plane_action(prev + (q - prev) * s / n))
            time.sleep(0.05)
        prev = q


def move_height(robot, args, dz, come_back):
    joints = read_joints(robot)
    hik = HeightIK(args.urdf, joints)
    q0 = hik.q.copy()
    path, problem = plan_height_path(hik, dz)
    if problem:
        print(f"Refusing to move {dz * 100:+.0f} cm: {problem}. Send me this output.")
        return
    print(f"Plan {dz * 100:+.0f} cm: joint changes {np.round(path[-1] - q0, 1)[1:4]} deg (lift, elbow, wrist)")
    input("Keep hands clear. Press ENTER to move (Ctrl+C to cancel)...")
    drive(robot, q0, path)
    if come_back:
        input(f"Did the phone go straight {'UP' if dz > 0 else 'DOWN'} about {abs(dz) * 100:.0f} cm "
              "without tilting? Press ENTER to return.")
        drive(robot, path[-1], list(reversed(path[:-1])) + [q0])
        print("Back at the start.")
    else:
        print("Done. The arm holds this pose; start tracking from here.")


def test_height(robot, args):
    move_height(robot, args, 0.02, come_back=True)


def raise_arm(robot, args):
    move_height(robot, args, args.raise_cm / 100, come_back=False)


# --------------------------------------------------------------------------- main loop


def run(args):
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    one_shot = (args.test_pan or args.test_tilt or args.check_ik or args.test_height or args.raise_cm
                or args.test_gripper or args.free_gripper)
    cam = None if one_shot else Camera(args.camera, exposure=args.exposure, flip=args.flip)

    robot = None
    if not args.dry_run:
        port = args.port or (FOLLOWER_PORT if os.path.exists(FOLLOWER_PORT) else find_follower_port())
        robot = connect_arm(port)
        tests = {"test_pan": test_pan, "test_tilt": test_tilt, "check_ik": check_ik, "test_height": test_height,
                 "raise_cm": raise_arm, "test_gripper": test_gripper, "free_gripper": free_gripper}
        for name, fn in tests.items():
            if getattr(args, name):
                try:
                    fn(robot) if name in ("test_pan", "test_tilt") else fn(robot, args)
                finally:
                    disconnect_arm(robot, args)
                return

    have_models = os.path.exists(YOLO_MODEL) and os.path.exists(FACE_MODEL)
    if args.detector == "face+body" or (args.detector == "auto" and have_models):
        detector = FaceAndBodyDetector(YOLO_MODEL, FACE_MODEL)
        log.info("Detector: YuNet face + YOLO body")
    elif args.detector == "yolo":
        detector = YoloDetector(YOLO_MODEL)
        log.info("Detector: YOLO body only")
    else:
        detector = PersonDetector()
        log.info("Detector: OpenCV Haar/HOG (models not found in tracker/models/)")
    commands = queue.Queue()  # "release" / "grip" from the preview page buttons or the terminal
    preview = Preview(args.preview_port, commands.put) if args.preview_port else None
    if sys.stdin.isatty():
        def read_terminal():
            for line in sys.stdin:
                key = line.strip().lower()
                if key in ("r", "g"):
                    commands.put("release" if key == "r" else "grip")
        threading.Thread(target=read_terminal, daemon=True).start()

    pan_goal = tilt_goal = pan_home = tilt_home = None
    hik = None
    tilt_offset = 0.0  # height mode: extra wrist tilt on top of the IK solution
    z_goal = 0.0
    if robot:
        joints = read_joints(robot)
        pan_goal = pan_home = joints[PAN]
        tilt_goal = tilt_home = joints[TILT]
        log.info(f"Start pose: {PAN}={pan_home:.1f} deg, {TILT}={tilt_home:.1f} deg")
        if args.height:
            hik = HeightIK(args.urdf, joints)
            z_goal = hik.z0
            log.info(f"Height mode: tip at {hik.z0 * 100:.1f} cm, range "
                     f"-{args.height_down * 100:.0f}/+{args.height_up * 100:.0f} cm")

    held = True  # is the gripper holding the phone?
    goals = {}  # last position we commanded for each joint (to see when a hand pushes it away)
    grip_home = None
    pull = None
    near_seen = 0.0  # last time a person was close enough to be the one grabbing the phone
    pull_warned = False
    if robot:
        grip_home = read_joints(robot)[GRIP]
        goals = {j: v for j, v in read_joints(robot).items()}
        if args.gripper_open_sign and args.squeeze > 0:
            grip_home = gentle_squeeze(robot, args, grip_home)
        if args.gripper_open_sign:
            log.info(f"Gripper: holding at {grip_home:.1f}; release: 'r'+ENTER or the page button, grip: 'g'+ENTER")
            if args.release_on_pull:
                pull = PullDetector({"load": args.pull_threshold, "push": args.pull_deg}, args.pull_time)
                log.info(f"Release on pull: ON (pull the phone firmly for {args.pull_time:g} s to take it; "
                         "--no-release-on-pull to disable)")
        elif args.release_on_pull:
            log.warning("--release-on-pull ignored: run --test-gripper and pass --gripper-open-sign first")

    def release(reason):
        nonlocal held, pull_warned
        log.info(f"Releasing phone ({reason})")
        move_gripper(robot, grip_home, grip_home + args.gripper_open_sign * args.release_amount)
        held, pull_warned = False, False

    def grip():
        nonlocal held
        log.info("Gripping phone; tracking resumes in 1 s")
        move_gripper(robot, grip_home + args.gripper_open_sign * args.release_amount, grip_home, seconds=1.5)
        time.sleep(1.0)
        if pull:
            pull.reset()
        held = True

    smoothed = None
    last_seen = 0.0
    seq, seq_at_t_fps = 0, 0
    frames, t_fps, fps, cam_fps = 0, time.time(), 0.0, 0.0
    t_end = time.time() + args.max_seconds if args.max_seconds else None

    try:
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            seq, frame = cam.read(seq)
            if frame is None:
                log.warning("No new frame from the camera for 1 s")
                continue
            h, w = frame.shape[:2]

            boxes = detector.detect(frame)
            target = pick_target(boxes, smoothed, args.max_jump * w)
            if target is not None:
                aim = aim_point(target)
                smoothed = aim if smoothed is None else tuple(
                    args.smoothing * s + (1 - args.smoothing) * a for s, a in zip(smoothed, aim)
                )
                last_seen = time.time()
            elif time.time() - last_seen > args.lost_timeout:
                smoothed = None  # forget the old position so we re-acquire whoever shows up next

            # Normalized error in [-1, 1]: positive = person is right of / below center.
            ex = ey = 0.0
            if target is not None and smoothed is not None:
                ex = (smoothed[0] - w / 2) / (w / 2)
                ey = (smoothed[1] - h / 2) / (h / 2)

            now = time.time()
            if target is not None:
                tw, th = target[2], target[3]
                if (target[4] == "face" and tw > 0.15 * w) or (target[4] != "face" and th > 0.6 * h):
                    near_seen = now

            freeze = False
            if robot and args.gripper_open_sign:
                try:
                    cmd = commands.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd == "release" and held:
                    release("button")
                elif cmd == "grip" and not held:
                    grip()
                if held and pull:
                    signals = read_pull_signals(robot, goals)
                    if signals:
                        pulling, confirmed = pull.update(signals, now)
                        # Hold still while checking: lag from our own motion fades in a fraction of a
                        # second once we stop, a hand pulling on the phone doesn't.
                        freeze = pulling
                        if confirmed and now - near_seen < 5.0:
                            release(f"pulled by hand ({pull.strongest})")
                        elif confirmed and not pull_warned:
                            log.warning("Felt a pull but nobody is close to the phone; not releasing")
                            pull_warned = True
            elif robot and not commands.empty():
                commands.get_nowait()
                log.warning("Release/grip needs --gripper-open-sign (run --test-gripper first)")

            if robot and target is not None and held and not freeze:
                action = {}
                if abs(ex) > args.deadband:
                    step = args.pan_sign * args.gain * ex * args.hfov / 2
                    step = float(np.clip(step, -args.max_step, args.max_step))
                    pan_goal = float(np.clip(pan_goal + step, pan_home - args.pan_limit, pan_home + args.pan_limit))
                    action[f"{PAN}.pos"] = pan_goal
                vfov = args.hfov * h / w
                tilt_step = 0.0
                if (args.tilt or hik) and abs(ey) > args.deadband:
                    tilt_step = float(np.clip(args.tilt_sign * args.gain * ey * vfov / 2, -args.max_step, args.max_step))
                if hik:
                    # Wrist does the fast aiming. Whenever it's tilted, move the whole phone in that direction
                    # (keeping its angle) so the wrist can relax back toward level: phone ends up at eye level.
                    tilt_offset = float(np.clip(tilt_offset + tilt_step, -args.tilt_limit, args.tilt_limit))
                    looking_down = tilt_offset * args.tilt_sign
                    if abs(looking_down) > 2.0:
                        dz = float(np.clip(-args.height_rate * looking_down, -0.004, 0.004))
                        z_new = float(np.clip(z_goal + dz, hik.z0 - args.height_down, hik.z0 + args.height_up))
                        if z_new != z_goal:
                            q, pos_err, ang_err = hik.solve(z_new)
                            # A few mm of height should need only a few degrees. A bigger jump means IK flipped
                            # to a different arm pose (e.g. near the end of a joint's range): refuse it.
                            jump = float(np.abs(q - hik.q)[1:4].max())
                            if pos_err < 0.01 and ang_err < 5 and jump < MAX_IK_JUMP_DEG:
                                hik.q, z_goal = q, z_new
                    action.update(plane_action(hik.q, tilt_offset))
                elif tilt_step:
                    tilt_goal = float(
                        np.clip(tilt_goal + tilt_step, tilt_home - args.tilt_limit, tilt_home + args.tilt_limit)
                    )
                    action[f"{TILT}.pos"] = tilt_goal
                if action:
                    sent = robot.send_action(action)
                    goals.update({k.removesuffix(".pos"): float(v) for k, v in sent.items()})

            frames += 1
            if time.time() - t_fps >= 1.0:
                elapsed = time.time() - t_fps
                # cam_fps = frames the camera delivered; fps = frames we actually processed.
                fps, cam_fps = frames / elapsed, (seq - seq_at_t_fps) / elapsed
                frames, t_fps, seq_at_t_fps = 0, time.time(), seq
                pose = f" pan={pan_goal:.1f}" if robot else ""
                if hik:
                    pose += f" height={(z_goal - hik.z0) * 100:+.1f}cm tilt={tilt_offset:+.1f}"
                if pull and held:
                    peak, key = pull.take_peak()
                    pose += f" pull_peak={peak:.2f}" + (f"({key})" if peak > 0.3 else "")
                if robot and not held:
                    pose += " RELEASED (press g+ENTER or the Grip button)"
                log.info(f"camera {cam_fps:4.1f} fps, processed {fps:4.1f} fps | {len(boxes)} detections | err x={ex:+.2f} y={ey:+.2f}{pose}")

            if preview:
                mode = "DRY RUN" if not robot else ("TRACKING" if held else "RELEASED")
                seen = target[4] if target is not None else "searching"
                status = f"{mode} | {seen} | ex={ex:+.2f} | cam {cam_fps:.0f} / proc {fps:.0f} fps"
                preview.show(annotate(frame, boxes, target, smoothed, args.deadband, status))
    finally:
        cam.close()
        if robot:
            disconnect_arm(robot, args)  # arm joints stay on and hold the pose; gripper goes limp
        log.info("Stopped.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="/dev/video0", help="camera mounted on the arm (path or index)")
    p.add_argument("--flip", action="store_true", help="rotate the image 180 deg (camera mounted upside down)")
    p.add_argument("--exposure", type=float,
                   help="manual exposure in 0.1 ms units (e.g. 300 for 30 fps in a dim room); default = camera auto")
    p.add_argument("--detector", default="auto", choices=["auto", "face+body", "yolo", "haar"],
                   help="auto = face+body if both models are in tracker/models/, else Haar/HOG")
    p.add_argument("--port", help="follower arm serial port; auto-detected if omitted")
    p.add_argument("--dry-run", action="store_true", help="camera + detection only, never touch the motors")
    p.add_argument("--test-pan", action="store_true", help="move the base 10 deg and back to check direction")
    p.add_argument("--test-tilt", action="store_true", help="move wrist_flex 10 deg and back to check direction")
    p.add_argument("--check-ik", action="store_true", help="print where the IK model thinks the tip is (no movement)")
    p.add_argument("--test-height", action="store_true", help="raise the tip 2 cm with IK and bring it back")
    p.add_argument("--raise", dest="raise_cm", type=float,
                   help="move the phone straight up (or down, if negative) this many cm with IK, then hold")
    p.add_argument("--height", action="store_true",
                   help="follow up/down by moving the phone (shoulder+elbow via IK) as well as tilting")
    p.add_argument("--urdf", default=DEFAULT_URDF, help="SO-101 URDF used for IK")
    p.add_argument("--height-up", type=float, default=0.15, help="max metres the phone may rise from the start pose")
    p.add_argument("--height-down", type=float, default=0.10, help="max metres the phone may drop from the start pose")
    p.add_argument("--height-rate", type=float, default=0.0005,
                   help="metres per frame per degree of wrist tilt (how eagerly the phone moves to eye level)")
    p.add_argument("--free-gripper", action="store_true",
                   help="while you hold the phone: turn the gripper motor off so it can be opened by hand")
    p.add_argument("--test-gripper", action="store_true",
                   help="while you hold the phone: find which way opens the gripper, and show pull loads")
    p.add_argument("--gripper-open-sign", type=float, choices=[1.0, -1.0], default=1.0,
                   help="direction that opens the gripper (+1 measured on this arm with --test-gripper)")
    p.add_argument("--squeeze", type=float, default=8.0,
                   help="on start, close the gripper up to this much past where it is so it grips firmly (0 = off)")
    p.add_argument("--grip-torque", type=float, default=35.0,
                   help="max gripper force while tracking, %% of the motor's max (LeRobot default is 50)")
    p.add_argument("--contact-load", type=float, default=200.0,
                   help="stop squeezing once the gripper feels this much push-back (of 1000)")
    p.add_argument("--release-amount", type=float, default=30.0, help="how far to open the gripper (0-100 scale)")
    p.add_argument("--release-on-pull", action=argparse.BooleanOptionalAction, default=True,
                   help="let go of the phone when someone close by pulls it (on by default)")
    p.add_argument("--pull-threshold", type=float, default=100.0,
                   help="wrist/gripper servo load change (of 1000) that counts as a pull")
    p.add_argument("--pull-deg", type=float, default=3.5,
                   help="degrees a shoulder/elbow/wrist joint must be pushed out of position to count as a pull")
    p.add_argument("--pull-time", type=float, default=0.3, help="seconds the pull must last (arm holds still)")
    p.add_argument("--max-jump", type=float, default=0.25,
                   help="ignore detections farther than this (fraction of frame width) from the current person")
    p.add_argument("--tilt", action="store_true", help="also tilt up/down with wrist_flex")
    p.add_argument("--pan-sign", type=float, default=1.0, choices=[1.0, -1.0])
    p.add_argument("--tilt-sign", type=float, default=1.0, choices=[1.0, -1.0])
    p.add_argument("--gain", type=float, default=0.15, help="fraction of the angular error corrected per frame")
    p.add_argument("--deadband", type=float, default=0.10, help="ignore errors smaller than this (fraction of half-frame)")
    p.add_argument("--max-step", type=float, default=1.5, help="max degrees per frame (bigger = faster but jerkier)")
    p.add_argument("--hfov", type=float, default=60.0, help="camera horizontal field of view in degrees")
    p.add_argument("--pan-limit", type=float, default=90.0, help="max degrees left/right of the start pose")
    p.add_argument("--tilt-limit", type=float, default=25.0, help="max degrees up/down of the start pose")
    p.add_argument("--smoothing", type=float, default=0.5, help="0 = raw detections, closer to 1 = smoother")
    p.add_argument("--lost-timeout", type=float, default=2.0, help="seconds before forgetting a lost person")
    p.add_argument("--preview-port", type=int, default=8000, help="0 disables the browser preview")
    p.add_argument("--max-seconds", type=float, help="stop automatically after this long")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
