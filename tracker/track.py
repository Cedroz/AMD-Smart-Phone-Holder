#!/usr/bin/env python
"""Person tracker for an SO-101 follower arm holding a phone.

The camera must be mounted on the arm so it turns with the phone. Each frame we
find the person, measure how far they are from the image center, and nudge
shoulder_pan (left/right) and optionally wrist_flex (up/down) to re-center them.
The preview also serves POST /commands and GET /status (Rotate 90 and Party mode
buttons included). Optional --gestures
uses the separate vision environment for closer/farther/freeze/resume commands.
Optional --distance maintains an approximate calibrated camera-to-user distance.

Run on the arm, inside the `lerobot` conda env:

    python tracker/track.py --dry-run          # camera + detection only, motors untouched
    python tracker/track.py --test-pan         # wiggle the base once to check direction
    python tracker/track.py --test-tilt        # same for the up/down wrist joint
    python tracker/track.py                    # track (Ctrl+C to stop)

Live preview: open http://<arm-ip>:8000 in a browser on your laptop.
"""

import argparse
import glob
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
ROLL = "wrist_roll"
# Joints that feel it when someone grabs and tugs the phone.
PULL_JOINTS = ["wrist_flex", "wrist_roll", "gripper"]
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
<button onclick="send('closer')">Closer</button><button onclick="send('farther')">Farther</button>
<button onclick="send('freeze')">Freeze</button><button onclick="send('resume')">Resume</button>
<button id="rel" onclick="send('release')">Release phone</button>
<button id="grip" onclick="send('grip')">Grip phone</button>
<button onclick="send('rotate')" style="background:#2f81f7;color:#fff">Rotate phone 90&deg;</button>
<button onclick="send('party')" style="background:#d63ac9;color:#fff">Party mode &#127881;</button>
<div id="msg"></div><pre id="state"></pre>
<script>async function send(c){try{const r=await fetch('/commands',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({command:c})});
msg.textContent=JSON.stringify(await r.json())}catch(e){msg.textContent='Could not reach the tracker'}}
setInterval(async()=>{try{state.textContent=JSON.stringify(await(await fetch('/status')).json(),null,2)}
catch(e){state.textContent='Tracker unavailable'}},500)</script></body></html>"""


def create_api(control, preview=None):
    """HTTP handlers only enqueue commands or read the tracker's published snapshot."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="Phone tracker")

    @app.post("/commands", status_code=202)
    def command(command: Literal["closer", "farther", "freeze", "resume", "release", "grip", "rotate",
                                 "party"] = Body(embed=True)):
        try:
            control.submit(command)
        except queue.Full:
            raise HTTPException(503, "Command queue is full")
        return {"accepted": True, "command": command}

    @app.get("/status")
    def status():
        return control.snapshot()

    @app.get("/", response_class=HTMLResponse)
    def page():
        return HTMLResponse(PREVIEW_PAGE)

    @app.get("/stream")
    def stream():
        if preview is None:
            raise HTTPException(503, "Camera preview unavailable")
        return StreamingResponse(preview.frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # Preserve the old gripper-button endpoints.
    app.post("/release", status_code=202)(lambda: command("release"))
    app.post("/grip", status_code=202)(lambda: command("grip"))
    return app


class Preview:
    """One FastAPI server for the camera feed, commands, and status."""

    def __init__(self, port, control):
        import uvicorn
        self.jpeg = None
        self.version = 0
        self.cond = threading.Condition()
        self.closed = False
        self.server = uvicorn.Server(uvicorn.Config(create_api(control, self),
            host="0.0.0.0", port=port, log_level="warning", access_log=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"Could not start tracker API on port {port}")
            time.sleep(0.01)
        log.info(f"Preview/API at http://<arm-ip>:{port}/; API documentation at /docs")

    def frames(self):
        sent = 0
        while not self.closed:
            with self.cond:
                self.cond.wait_for(lambda: self.version != sent or self.closed, timeout=1)
                jpeg, version = self.jpeg, self.version
            if self.closed:
                return
            if jpeg is not None and version != sent:
                sent = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def show(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.version += 1
                self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.server.should_exit = True
        self.thread.join(timeout=2)


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
        label = f"{kind} {bw}px" if kind == "face" or kind.startswith("face:") else kind
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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


def connect_arm(port):
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
    """Disconnect, leaving the arm holding its pose but the gripper motor off so it can be opened by hand."""
    if not args.keep_grip:
        try:
            robot.bus.disable_torque(GRIP, num_retry=3)
            log.info("Gripper motor off: you can open it by hand (use --keep-grip to keep it clamped)")
        except ConnectionError as e:
            log.warning(f"Couldn't switch the gripper off ({e})")
    robot.disconnect()


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
    """Moves the phone vertically and optionally forward/back, keeping its pitch.

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
        self.reach0 = float(self.home[0, 3])

    def solve(self, z, reach=None):
        """Joints for height z and optional forward reach, with the start orientation.

        Returns (q, position error in m, orientation error in deg). Starts from the last accepted
        solution, so small height changes converge quickly.
        """
        pose = self.home.copy()
        pose[2, 3] = z
        if reach is not None:
            pose[0, 3] = reach
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
    for i in range(1, steps + 1):
        pos = start - args.gripper_open_sign * args.squeeze * i / steps
        robot.send_action({f"{GRIP}.pos": float(pos)})
        time.sleep(1.5 / steps)
        loads = read_loads(robot)
        if base is not None and loads and abs(loads[GRIP] - base) > args.contact_load:
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
    """Feels someone gripping and tugging the phone: a sustained jump in wrist/gripper servo load.

    The load from gravity and from the arm's own motion is tracked as a slowly adapting baseline;
    only a deviation above `threshold` that lasts `hold_s` seconds counts as a pull.
    """

    def __init__(self, threshold, hold_s):
        self.threshold, self.hold_s = threshold, hold_s
        self.reset()

    def reset(self):
        self.baseline, self.since, self.deviation = None, None, 0.0

    def update(self, loads, now):
        """Returns (pulling, confirmed): pulling = over threshold now, confirmed = for long enough."""
        if self.baseline is None:
            self.baseline = {j: float(v) for j, v in loads.items()}
            return False, False
        self.deviation = max(abs(loads[j] - self.baseline[j]) for j in loads)
        if self.deviation > self.threshold:
            if self.since is None:
                self.since = now
            return True, now - self.since >= self.hold_s
        self.since = None
        for j in loads:
            self.baseline[j] += 0.05 * (loads[j] - self.baseline[j])
        return False, False


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


class DepthFollower:
    """Approximate frontal-face range; propose bounded reach without committing IK state."""

    def __init__(self, args):
        self.args = args
        self.distance = None

    def update(self, target, frame_width):
        if (target is None or (target[4] != "face" and not target[4].startswith("face:"))
                or target[2] <= 0 or target[0] <= 0 or target[0] + target[2] >= frame_width):
            self.distance = None
            return
        distance = self.args.distance_reference * self.args.distance_face_width / target[2]
        self.distance = distance if self.distance is None else (
            self.args.smoothing * self.distance + (1 - self.args.smoothing) * distance)

    def propose(self, reach, home, dt):
        if self.distance is None:
            return reach
        error = self.distance - self.args.distance
        if abs(error) <= self.args.distance_tolerance:
            return reach
        dt = max(0.0, min(dt, 0.1))
        max_step = min(0.004, self.args.depth_rate * dt)
        step = float(np.clip(0.5 * error * dt, -max_step, max_step))
        return float(np.clip(reach + step, home - self.args.depth_back, home + self.args.depth_forward))


class TrackerControl:
    """Shared commands; only the tracking loop mutates control state or moves motors."""
    COMMANDS = ("closer", "farther", "freeze", "resume", "release", "grip", "rotate", "party")

    def __init__(self, args):
        self.args = args
        self.commands = queue.Queue(maxsize=32)
        self.frozen = False
        self.reach0 = self.reach = self.requested_reach = 0.0
        self.depth = DepthFollower(args) if args.distance is not None else None
        self.last_command = None
        self.lock = threading.Lock()
        self.status = {}
        self.publish(mode="starting", held=True, target_visible=False)

    def submit(self, command):
        if command not in self.COMMANDS:
            raise ValueError("Unknown command")
        self.commands.put_nowait(command)

    def apply(self, command, held=True):
        if command == "freeze":
            self.frozen = True
            self.requested_reach = self.reach  # cancel unfinished manual depth moves
        elif command == "resume":
            self.frozen = False
        elif command in ("closer", "farther"):
            if self.frozen or not held:
                return "ignored: frozen or phone released"
            sign = 1 if command == "closer" else -1
            if self.depth:
                new_distance = self.args.distance - sign * self.args.gesture_step
                if new_distance <= 0:
                    return "ignored: distance must stay positive"
                self.args.distance = new_distance
            else:
                self.requested_reach = float(np.clip(self.requested_reach + sign * self.args.gesture_step,
                    self.reach0 - self.args.depth_back, self.reach0 + self.args.depth_forward))
            return "target_updated"
        return "applied"

    def propose_reach(self, dt):
        if self.depth:
            return self.depth.propose(self.reach, self.reach0, dt)
        step = min(0.004, self.args.depth_rate * max(0.0, min(dt, 0.1)))
        return self.reach + float(np.clip(self.requested_reach - self.reach, -step, step))

    def publish(self, **fields):
        with self.lock:
            self.status.update(fields)
            self.status.update(frozen=self.frozen, dry_run=self.args.dry_run,
                target_distance=self.args.distance, estimated_distance=self.depth.distance if self.depth else None,
                reach=self.reach, requested_reach=None if self.depth else self.requested_reach,
                reach_min=self.reach0 - self.args.depth_back, reach_max=self.reach0 + self.args.depth_forward,
                at_reach_limit=(self.reach <= self.reach0 - self.args.depth_back + 1e-6
                                or self.reach >= self.reach0 + self.args.depth_forward - 1e-6),
                last_command=self.last_command)

    def snapshot(self):
        with self.lock:
            return self.status.copy()


# (joint, swing deg each side, seconds per swing at speed 1). Each joint is flung to one end, then the
# other: smooth sine waves are faster than the servos can follow, so they only wiggle a little.
# Driving straight at each end runs them at their real top speed and reaches the full swing.
PARTY_MOVES = [
    ("shoulder_pan", 30.0, 0.35),
    ("wrist_flex", 20.0, 0.25),
    ("wrist_roll", 45.0, 0.40),
    ("elbow_flex", 8.0, 0.50),
    ("shoulder_lift", 5.0, 0.60),
]


def party(robot, args, stop, commands):
    """Dance until Party is pressed again (or Ctrl+C), then ease back to where it started.

    Runs its own ~40 Hz loop (tracking is limited by the camera). Swings grow over the first second,
    wrist_roll stays in range, and the gripper is never touched. Other commands that arrive while
    dancing are put back in the queue for the tracking loop.
    """
    start = read_joints(robot)
    log.info(f"PARTY MODE (speed x{args.party_speed:g}) - press Party again to stop")
    deferred, last, t0 = [], {}, time.time()
    while not stop.is_set():
        try:
            cmd = commands.get_nowait()
            if cmd == "party":
                break
            deferred.append(cmd)
        except queue.Empty:
            pass
        t = time.time() - t0
        ramp = min(1.0, t / 1.0)
        action = {}
        for joint, amp, swing_s in PARTY_MOVES:
            side = 1.0 if int(t * args.party_speed / swing_s) % 2 == 0 else -1.0
            angle = start[joint] + ramp * amp * side
            if joint == ROLL:
                angle = np.clip(angle, -170.0, 170.0)
            action[f"{joint}.pos"] = float(angle)
        robot.send_action(action)
        last = action
        time.sleep(max(0.0, 1 / 40 - (time.time() - t0 - t)))
    # Ease back to the starting pose over 1 s so tracking picks up where it left off.
    for i in range(1, 21):
        robot.send_action({k: float(v + (start[k.removesuffix(".pos")] - v) * i / 20) for k, v in last.items()})
        time.sleep(0.05)
    for cmd in deferred:
        commands.put_nowait(cmd)
    log.info("Party over, back to tracking")


def start_gestures(args):
    # The parent imports only the IPC client; MediaPipe stays in .venv-vision.
    sys.path.insert(0, str(ROOT / "src"))
    from followme.vision_client import VisionClient
    client = VisionClient(ROOT, {
        "vision_python": args.gesture_python, "mediapipe_version": "1.1.0",
        "model_path": "models/gestures/gesture_recognizer.task",
        "model_manifest": "models/gestures/gesture_manifest.json",
        "worker_module": "followme.gestures.recognizer"}, inference_timeout=1)
    client.start()
    return client


# --------------------------------------------------------------------------- main loop


def run(args):
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    one_shot = (args.test_pan or args.test_tilt or args.check_ik or args.test_height or args.raise_cm
                or args.test_gripper)
    cam = robot = preview = gestures = None
    control = TrackerControl(args)
    try:
        cam = None if one_shot else Camera(args.camera, exposure=args.exposure, flip=args.flip)
        robot = None
        if not args.dry_run:
            port = args.port or (FOLLOWER_PORT if os.path.exists(FOLLOWER_PORT) else find_follower_port())
            robot = connect_arm(port)
            tests = {"test_pan": test_pan, "test_tilt": test_tilt, "check_ik": check_ik, "test_height": test_height,
                     "raise_cm": raise_arm, "test_gripper": test_gripper}
            for name, fn in tests.items():
                if getattr(args, name):
                    fn(robot) if name in ("test_pan", "test_tilt") else fn(robot, args)
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
        commands = control.commands
        gestures = start_gestures(args) if args.gestures else None
        preview = Preview(args.preview_port, control) if args.preview_port else None
        if sys.stdin.isatty():
            def read_terminal():
                for line in sys.stdin:
                    key = line.strip().lower()
                    command = {"r": "release", "g": "grip"}.get(key, key)
                    if command in control.COMMANDS:
                        try:
                            control.submit(command)
                        except queue.Full:
                            log.warning("Command queue is full")
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
            if args.height or args.distance is not None or args.gestures:
                hik = HeightIK(args.urdf, joints)
                control.reach0 = control.reach = control.requested_reach = hik.reach0
                z_goal = hik.z0
                log.info(f"IK: tip at {hik.z0 * 100:.1f} cm, height range "
                         f"-{args.height_down * 100:.0f}/+{args.height_up * 100:.0f} cm")

        held = True  # is the gripper holding the phone?
        grip_home = None
        pull = None
        near_seen = 0.0  # last time a person was close enough to be the one grabbing the phone
        pull_warned = False
        if robot:
            grip_home = read_joints(robot)[GRIP]
            if args.gripper_open_sign and args.squeeze > 0:
                grip_home = gentle_squeeze(robot, args, grip_home)
            if args.gripper_open_sign:
                log.info(f"Gripper: holding at {grip_home:.1f}; release: 'r'+ENTER or the page button, grip: 'g'+ENTER")
                if args.release_on_pull:
                    pull = PullDetector(args.pull_threshold, args.pull_time)
                    log.info(f"Release on pull: ON (threshold {args.pull_threshold:g}, {args.pull_time:g} s)")
            elif args.release_on_pull:
                log.warning("--release-on-pull ignored: run --test-gripper and pass --gripper-open-sign first")

        roll_home = read_joints(robot)[ROLL] if robot else 0.0
        # wrist_roll spans about -180..180 deg: turn whichever way stays inside that range.
        rotate_by = args.rotate_deg if abs(roll_home + args.rotate_deg) <= 170 else -args.rotate_deg
        rotated = False  # phone turned 90 deg from where it started

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

        def process_commands():
            nonlocal hik, z_goal, rotated, smoothed
            for _ in range(32):
                try:
                    cmd = commands.get_nowait()
                except queue.Empty:
                    break
                if cmd in ("rotate", "party") and not robot:
                    result = "ignored: dry run"
                elif cmd == "rotate":
                    start = roll_home + (rotate_by if rotated else 0.0)
                    end = roll_home + (0.0 if rotated else rotate_by)
                    log.info(f"Rotating the phone {'back' if rotated else 'by'} {end - start:+.0f} deg")
                    for i in range(1, 31):  # slowly, over about 1.5 s
                        robot.send_action({f"{ROLL}.pos": float(start + (end - start) * i / 30)})
                        time.sleep(0.05)
                    rotated = not rotated
                    smoothed = None  # the view just turned; re-find the person
                    result = "applied"
                elif cmd == "party":
                    control.publish(mode="party")
                    party(robot, args, stop, commands)
                    smoothed = None  # re-find the person after dancing
                    result = "applied"
                elif cmd in ("release", "grip"):
                    if not robot:
                        result = "ignored: dry run"
                    elif not args.gripper_open_sign:
                        result = "ignored: gripper direction is not configured"
                    elif cmd == "release" and held:
                        release("command")
                        control.requested_reach = control.reach
                        result = "applied"
                    elif cmd == "grip" and not held:
                        grip()
                        result = "applied"
                    else:
                        result = "already applied"
                else:
                    if (cmd in ("closer", "farther") and robot and hik is None
                            and not control.frozen and held):
                        hik = HeightIK(args.urdf, read_joints(robot))
                        z_goal = hik.z0
                        control.reach0 = control.reach = control.requested_reach = hik.reach0
                    result = control.apply(cmd, held)
                control.last_command = {"command": cmd, "result": result}
                log.info(f"Command {cmd}: {result}")
            control.publish(held=held)

        smoothed = None
        last_seen = 0.0
        seq, seq_at_t_fps = 0, 0
        frames, t_fps, fps, cam_fps = 0, time.time(), 0.0, 0.0
        t_end = time.time() + args.max_seconds if args.max_seconds else None

        last_control = time.monotonic()
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            process_commands()
            seq, frame = cam.read(seq)
            now_control = time.monotonic()
            dt, last_control = now_control - last_control, now_control
            if frame is None:
                control.publish(mode="frozen" if control.frozen else "hold", target_visible=False, motion_blocker="camera unavailable")
                log.warning("No new frame from the camera for 1 s")
                continue
            if rotated and args.camera_on_roll != "no":
                # The camera turned with the phone: turn the picture back so people are upright again.
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if args.camera_on_roll == "cw"
                                   else cv2.ROTATE_90_COUNTERCLOCKWISE)
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

            if control.depth:
                control.depth.update(target, w)
            if gestures:
                face_center = (aim_point(target)[0] / w, aim_point(target)[1] / h) if target is not None and target[4] == "face" else None
                try:
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise RuntimeError("Could not encode gesture frame")
                    reply = gestures.infer("external", seq, time.monotonic_ns(), jpeg.tobytes(), face_center=face_center)
                    if reply.get("error"):
                        raise RuntimeError(reply["error"])
                    if reply.get("command"):
                        control.submit(reply["command"])
                except Exception:
                    # Do not continue moving if an enabled gesture control stops working.
                    control.frozen = True
                    control.requested_reach = control.reach
                    control.publish(mode="frozen", motion_blocker="gesture worker failed")
                    raise
            # Include HTTP/terminal commands received while inference was running.
            process_commands()

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
            if robot and args.gripper_open_sign and held and pull:
                loads = read_loads(robot)
                if loads:
                    pulling, confirmed = pull.update(loads, now)
                    freeze = pulling
                    if confirmed and now - near_seen < 3.0:
                        release("pulled by hand")
                        control.requested_reach = control.reach
                    elif confirmed and not pull_warned:
                        log.warning("Felt a pull but nobody is close to the phone; not releasing")
                        pull_warned = True

            motion_blocker = None
            if control.frozen:
                motion_blocker = "frozen"
            elif not held:
                motion_blocker = "phone released"
            elif target is None:
                motion_blocker = "target unavailable"
            elif freeze:
                motion_blocker = "pull detected"
            if robot and motion_blocker is None:
                action = {}
                if abs(ex) > args.deadband:
                    step = args.pan_sign * args.gain * ex * args.hfov / 2
                    step = float(np.clip(step, -args.max_step, args.max_step))
                    pan_goal = float(np.clip(pan_goal + step, pan_home - args.pan_limit, pan_home + args.pan_limit))
                    action[f"{PAN}.pos"] = pan_goal
                vfov = args.hfov * h / w
                tilt_step = 0.0
                if (args.tilt or args.height) and abs(ey) > args.deadband:
                    tilt_step = float(np.clip(args.tilt_sign * args.gain * ey * vfov / 2, -args.max_step, args.max_step))
                if hik:
                    # Wrist does the fast aiming. Whenever it's tilted, move the whole phone in that direction
                    # (keeping its angle) so the wrist can relax back toward level: phone ends up at eye level.
                    tilt_offset = float(np.clip(tilt_offset + tilt_step, -args.tilt_limit, args.tilt_limit))
                    looking_down = tilt_offset * args.tilt_sign
                    z_new = z_goal
                    if args.height and abs(looking_down) > 2.0:
                        dz = float(np.clip(-args.height_rate * looking_down, -0.004, 0.004))
                        z_new = float(np.clip(z_goal + dz, hik.z0 - args.height_down, hik.z0 + args.height_up))
                    reach_new = control.propose_reach(dt)
                    if z_new != z_goal or reach_new != control.reach:
                        q, pos_err, ang_err = hik.solve(z_new, reach_new)
                        jump = float(np.abs(q - hik.q)[1:4].max())
                        if np.isfinite(q).all() and pos_err < 0.01 and ang_err < 5 and jump < MAX_IK_JUMP_DEG:
                            hik.q, z_goal, control.reach = q, z_new, reach_new
                        else:
                            motion_blocker = "IK rejected"
                            control.requested_reach = control.reach
                    action.update(plane_action(hik.q, tilt_offset))
                elif tilt_step:
                    tilt_goal = float(
                        np.clip(tilt_goal + tilt_step, tilt_home - args.tilt_limit, tilt_home + args.tilt_limit)
                    )
                    action[f"{TILT}.pos"] = tilt_goal
                if action:
                    sent = robot.send_action(action)
                    # Anti-windup: LeRobot caps each move at 10 deg from where the joint really is, and a joint at
                    # its physical end can't move at all. Continue from what was actually sent, so our target never
                    # runs ahead of the arm (otherwise it must "unwind" before the arm turns back the other way).
                    if f"{PAN}.pos" in sent:
                        pan_goal = float(sent[f"{PAN}.pos"])
                    if f"{TILT}.pos" in sent:
                        if hik:
                            tilt_offset = float(np.clip(sent[f"{TILT}.pos"] - hik.q[3], -args.tilt_limit, args.tilt_limit))
                        else:
                            tilt_goal = float(sent[f"{TILT}.pos"])

            control.publish(mode="frozen" if control.frozen else ("tracking" if held else "released"),
                held=held, target_visible=target is not None, motion_blocker=motion_blocker)
            frames += 1
            if time.time() - t_fps >= 1.0:
                elapsed = time.time() - t_fps
                # cam_fps = frames the camera delivered; fps = frames we actually processed.
                fps, cam_fps = frames / elapsed, (seq - seq_at_t_fps) / elapsed
                frames, t_fps, seq_at_t_fps = 0, time.time(), seq
                pose = f" pan={pan_goal:.1f}" if robot else ""
                if hik:
                    pose += f" height={(z_goal - hik.z0) * 100:+.1f}cm tilt={tilt_offset:+.1f}"
                if pull:
                    pose += f" load_change={pull.deviation:.0f}"
                if robot and not held:
                    pose += " RELEASED (press g+ENTER or the Grip button)"
                if rotated:
                    pose += " ROTATED"
                log.info(f"camera {cam_fps:4.1f} fps, processed {fps:4.1f} fps | {len(boxes)} detections | err x={ex:+.2f} y={ey:+.2f}{pose}")

            if preview:
                mode = "FROZEN" if control.frozen else ("DRY RUN" if not robot else ("TRACKING" if held else "RELEASED"))
                seen = target[4] if target is not None else "searching"
                status = f"{mode} | {seen} | ex={ex:+.2f} | cam {cam_fps:.0f} / proc {fps:.0f} fps"
                if control.depth:
                    status += f" | range {control.depth.distance:.2f}m" if control.depth.distance is not None else " | depth HOLD"
                preview.show(annotate(frame, boxes, target, smoothed, args.deadband, status))
    finally:
        control.publish(mode="stopped")
        if gestures:
            gestures.close()
        if preview:
            preview.close()
        if cam:
            cam.close()
        if robot:
            disconnect_arm(robot, args)
        log.info("Stopped.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="/dev/video0", help="camera mounted on the arm (path or index)")
    p.add_argument("--party-speed", type=float, default=1.0,
                   help="Party mode tempo (bigger = quicker direction changes, smaller swings once the servos "
                        "can't reach the ends in time)")
    p.add_argument("--rotate-deg", type=float, default=90.0,
                   help="how far the Rotate button turns the phone (wrist_roll); it picks the direction with room")
    p.add_argument("--camera-on-roll", choices=["no", "cw", "ccw"], default="no",
                   help="if the camera turns with the phone, which way to turn the picture back after rotating")
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
    p.add_argument("--gestures", action="store_true", help="enable closer/farther/freeze/resume via the vision worker")
    p.add_argument("--gesture-python", default=str(ROOT / ".venv-vision/bin/python"), help="isolated vision Python interpreter")
    p.add_argument("--gesture-step", type=float, default=0.02, help="metres per closer/farther command")
    p.add_argument("--distance", type=float, help="follow at this approximate calibrated distance in metres")
    p.add_argument("--distance-reference", type=float, help="known camera-to-face calibration distance in metres")
    p.add_argument("--distance-face-width", type=float, help="face-box width in pixels at the reference distance")
    p.add_argument("--distance-tolerance", type=float, default=0.10, help="distance deadband in metres")
    p.add_argument("--depth-forward", type=float, default=0.05, help="max metres to extend from starting reach")
    p.add_argument("--depth-back", type=float, default=0.05, help="max metres to retract from starting reach")
    p.add_argument("--depth-rate", type=float, default=0.02, help="max reach speed in metres per second")
    p.add_argument("--urdf", default=DEFAULT_URDF, help="SO-101 URDF used for IK")
    p.add_argument("--height-up", type=float, default=0.15, help="max metres the phone may rise from the start pose")
    p.add_argument("--height-down", type=float, default=0.10, help="max metres the phone may drop from the start pose")
    p.add_argument("--height-rate", type=float, default=0.0005,
                   help="metres per frame per degree of wrist tilt (how eagerly the phone moves to eye level)")
    p.add_argument("--keep-grip", action="store_true",
                   help="leave the gripper clamped when the program exits (default: gripper motor turns off)")
    p.add_argument("--test-gripper", action="store_true",
                   help="while you hold the phone: find which way opens the gripper, and show pull loads")
    p.add_argument("--gripper-open-sign", type=float, choices=[1.0, -1.0], default=1.0,
                   help="direction that opens the gripper (+1 measured on this arm with --test-gripper)")
    p.add_argument("--squeeze", type=float, default=6.0,
                   help="on start, close the gripper up to this much past where it is so it grips firmly (0 = off)")
    p.add_argument("--grip-torque", type=float, default=20.0,
                   help="max gripper force while tracking, %% of the motor's max (LeRobot default is 50)")
    p.add_argument("--contact-load", type=float, default=80.0,
                   help="stop squeezing once the gripper feels this much push-back (of 1000)")
    p.add_argument("--release-amount", type=float, default=30.0, help="how far to open the gripper (0-100 scale)")
    p.add_argument("--release-on-pull", action="store_true",
                   help="let go of the phone when someone close by grabs and tugs it")
    p.add_argument("--pull-threshold", type=float, default=250.0,
                   help="servo load change (of 1000) that counts as a tug")
    p.add_argument("--pull-time", type=float, default=0.5, help="seconds the tug must last")
    p.add_argument("--max-jump", type=float, default=0.25,
                   help="ignore detections farther than this (fraction of frame width) from the current person")
    p.add_argument("--tilt", action="store_true", help="also tilt up/down with wrist_flex")
    p.add_argument("--pan-sign", type=float, default=1.0, choices=[1.0, -1.0])
    p.add_argument("--tilt-sign", type=float, default=1.0, choices=[1.0, -1.0])
    p.add_argument("--gain", type=float, default=0.15, help="fraction of the angular error corrected per frame")
    p.add_argument("--deadband", type=float, default=0.10, help="ignore errors smaller than this (fraction of half-frame)")
    p.add_argument("--max-step", type=float, default=3.0, help="max degrees per frame")
    p.add_argument("--hfov", type=float, default=60.0, help="camera horizontal field of view in degrees")
    p.add_argument("--pan-limit", type=float, default=90.0, help="max degrees left/right of the start pose")
    p.add_argument("--tilt-limit", type=float, default=25.0, help="max degrees up/down of the start pose")
    p.add_argument("--smoothing", type=float, default=0.5, help="0 = raw detections, closer to 1 = smoother")
    p.add_argument("--lost-timeout", type=float, default=2.0, help="seconds before forgetting a lost person")
    p.add_argument("--preview-port", type=int, default=8000, help="0 disables the browser preview")
    p.add_argument("--max-seconds", type=float, help="stop automatically after this long")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    for name in ("gesture_step", "depth_rate"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("depth_forward", "depth_back", "distance_tolerance"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.distance is not None:
        for name in ("distance", "distance_reference", "distance_face_width"):
            value = getattr(args, name)
            if value is None or not np.isfinite(value) or value <= 0:
                p.error(f"--{name.replace('_', '-')} must be finite and positive with --distance")
        if not np.isfinite(args.smoothing) or not 0 <= args.smoothing < 1:
            p.error("--smoothing must be between 0 (inclusive) and 1 (exclusive) for depth following")
        if args.detector == "yolo":
            p.error("--distance requires frontal-face detection")
    if not 0 <= args.preview_port <= 65535:
        p.error("--preview-port must be between 0 and 65535")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    # LeRobot warns each time a move is capped at 10 deg per command; Party mode does that on purpose
    # dozens of times a second (it's how it runs the servos at full speed), so keep the log readable.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
    run(args)


if __name__ == "__main__":
    main()
