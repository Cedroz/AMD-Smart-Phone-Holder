#!/usr/bin/env python
"""Person tracker for an SO-101 follower arm holding a phone.

The camera must be mounted on the arm so it turns with the phone. Each frame we
find the person, measure how far they are from the image center, and nudge
shoulder_pan (left/right) and optionally wrist_flex (up/down) to re-center them.
The preview also serves POST /commands and GET /status (Rotate 90 and Party mode
buttons included). Optional --gestures
uses the separate vision environment for closer/farther/freeze/resume commands (pointing the index
finger up/down also moves the phone closer/farther); an L sign with the left hand rotates the phone
to landscape, with the right hand back to portrait.
Optional --distance maintains an approximate calibrated camera-to-user distance.
Cover the camera with your hand for 2 s to release the phone (Grip phone puts it back).

Run on the arm, inside the `lerobot` conda env:

    python tracker/track.py --dry-run          # camera + detection only, motors untouched
    python tracker/track.py --test-pan         # wiggle the base once to check direction
    python tracker/track.py --test-tilt        # same for the up/down wrist joint
    python tracker/track.py                    # track (Ctrl+C to stop)

Live preview: open http://<arm-ip>:8000 in a browser on your laptop.
"""

import argparse
import atexit
import glob
import logging
import os
import queue
import signal
import subprocess
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
        # However the program ends (even a crash during startup), stop the reader thread first: if
        # Python exits while it's inside OpenCV, it aborts with "FATAL: exception not rethrown".
        atexit.register(self.close)

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
        if not self.running:
            return
        self.running = False
        self.thread.join(timeout=3.0)
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
#face-enrollment{max-width:640px;margin:1em auto;padding:1em;background:#222;border-radius:.6em}
input{font:inherit;padding:.6em;border-radius:.4em;border:1px solid #666;max-width:55%}
progress{width:90%;margin:.7em auto;display:block}button:disabled{opacity:.5;cursor:wait}
#face-message{min-height:2.5em}
</style></head><body><img src="/stream">
<button onclick="send('closer')">Closer</button><button onclick="send('farther')">Farther</button>
<button onclick="send('freeze')">Freeze</button><button onclick="send('resume')">Resume</button>
<button id="rel" onclick="send('release')">Release phone</button>
<button id="grip" onclick="send('grip')">Grip phone</button><button onclick="send('rotate')" style="background:#2f81f7;color:#fff">Rotate phone 90&deg;</button>
<button onclick="send('party')" style="background:#d63ac9;color:#fff">Party mode &#127881;</button>
<button onclick="send('home')" style="background:#8b949e;color:#fff">Reset pose</button>
<div id="msg" role="status"></div>
<section aria-label="Following mode" style="max-width:640px;margin:1em auto;padding:1em;background:#222;border-radius:.6em">
<h2>Following mode</h2>
<button id="follow-manual" onclick="setFollow('manual')" disabled>Manual / Go to</button>
<button id="follow-speaker" onclick="setFollow('speaker')" disabled>Follow speaker</button>
<p>Manual uses the mobile Go to selection. Speaker mode switches to recognized speakers.</p>
<div id="follow-status" role="status" aria-live="polite">Connecting to tracker...</div>
<h3>Go to a person</h3><div id="follow-people" aria-label="Enrolled people"></div>
<button id="clear-person" onclick="choosePerson(null)" disabled>Clear selection</button>
<p id="people-help">Loading enrolled names...</p>
<p id="voice-help" role="status" aria-live="polite"></p>
<details id="mic-setup"><summary>Laptop microphone setup</summary>
<p>In your laptop's Terminal, start the SSH tunnel and the existing microphone sender in separate terminals.
Then click Follow speaker above. The web page itself does not capture microphone audio.</p>
<pre id="mic-commands" style="text-align:left;white-space:pre-wrap"></pre>
<p>Replace INDEX with your microphone's device number from <code>python3 mac_mic_sender.py --list-devices</code>.
Keep both terminals running. If the listener fails, click Follow speaker again to retry.</p></details></section>
<section id="face-enrollment" aria-label="Face enrollment">
<h2>Enroll a face</h2><p>Keep only this person in view and slowly vary their head angle.
Tracking pauses while 20 samples are collected. Enrolling an existing name replaces its samples.</p>
<form id="face-form"><label for="face-name">Name </label>
<input id="face-name" maxlength="80" placeholder="e.g. Alice" required autocomplete="off">
<button id="face-toggle" type="submit" disabled>Enroll face</button></form>
<progress id="face-progress" value="0" max="20" aria-label="Enrollment progress"></progress>
<div id="face-message" role="status" aria-live="polite">Connecting to tracker...</div></section>
<pre id="state"></pre>
<script>async function send(c){try{const r=await fetch('/commands',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({command:c})});
msg.textContent=JSON.stringify(await r.json())}catch(e){msg.textContent='Could not reach the tracker'}}
const faceName=document.getElementById('face-name'),faceToggle=document.getElementById('face-toggle'),
faceMessage=document.getElementById('face-message'),faceProgress=document.getElementById('face-progress');
let enrolling=false,facePending=false,pollPending=false;
function showEnrollment(e){if(!e)return;enrolling=['starting','running','cancelling'].includes(e.state);
faceName.disabled=enrolling;faceName.required=!enrolling;faceToggle.disabled=facePending||e.state==='cancelling';
faceToggle.textContent=enrolling?'Cancel enrollment':'Enroll face';
faceProgress.max=e.total;faceProgress.value=e.samples;
faceMessage.textContent=e.message+(enrolling?' ('+e.samples+'/'+e.total+')':'');}
document.getElementById('face-form').addEventListener('submit',async(event)=>{event.preventDefault();
if(facePending)return;facePending=true;faceToggle.disabled=true;
try{const r=await fetch(enrolling?'/faces/enrollment/cancel':'/faces/enrollment',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({name:faceName.value.trim()})});
const result=await r.json();if(!r.ok)throw new Error(typeof result.detail==='string'?result.detail:'Enter a valid name.');
showEnrollment(result);}catch(e){document.getElementById('msg').textContent=e.message;}
finally{facePending=false;faceToggle.disabled=false;}});
let followPending=false,personPending=false,shownPeople='';
const people=document.getElementById('follow-people'),clearPerson=document.getElementById('clear-person');
document.getElementById('mic-commands').textContent=
`# Terminal 1: SSH tunnel
ssh -N -L 18765:127.0.0.1:8765 aup@${location.hostname}

# Terminal 2: microphone sender
cd ~/speakerid-mic
source .venv/bin/activate
python3 mac_mic_sender.py --device INDEX`;
async function choosePerson(name){if(personPending)return;personPending=true;
for(const button of people.children)button.disabled=true;clearPerson.disabled=true;
try{const r=await fetch('/find',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
const result=await r.json();if(!r.ok)throw new Error(result.detail||'Could not select person');
}catch(e){msg.textContent=e.message;}finally{personPending=false;poll();}}
function showPeople(s){const names=s.faces||[],signature=JSON.stringify(names);
if(signature!==shownPeople){shownPeople=signature;people.replaceChildren(...names.map(name=>{
const button=document.createElement('button');button.textContent=name;button.dataset.name=name;
button.addEventListener('click',()=>choosePerson(name));return button;}));}
for(const button of people.children){button.disabled=personPending||s.deactivated;
button.setAttribute('aria-pressed',String(button.dataset.name===s.finding));
button.style.background=button.dataset.name===s.finding?'#46954a':'';}
clearPerson.disabled=personPending||!s.finding;
document.getElementById('people-help').textContent=names.length?
'Choose a name to follow. In speaker mode, the next recognized speaker can override your selection.':
'No faces enrolled yet. Use Enroll a face below.';}

const manualButton=document.getElementById('follow-manual'),speakerButton=document.getElementById('follow-speaker');
function showFollow(s){if(!s.follow_mode)return;
manualButton.disabled=followPending;speakerButton.disabled=followPending;
manualButton.setAttribute('aria-pressed',String(s.follow_mode==='manual'));
speakerButton.setAttribute('aria-pressed',String(s.follow_mode==='speaker'));
manualButton.style.background=s.follow_mode==='manual'?'#46954a':'';
speakerButton.style.background=s.follow_mode==='speaker'?'#46954a':'';
showPeople(s);const v=s.voice||{};document.getElementById('voice-help').textContent=s.voice_feedback||'';
document.getElementById('follow-status').textContent=
'Mode: '+(s.follow_mode==='speaker'?'Follow speaker':'Manual / Go to')+
' | Selected: '+(s.finding||'none')+' | Following: '+(s.active_target||'none')+
' | Voice listener: '+(s.voice_listener||'off')+' | Latest speaker: '+(v.speaker||'none')+
(v.speaker?' (similarity '+Number(v.confidence||0).toFixed(2)+')':'');}
async function setFollow(mode){if(followPending)return;followPending=true;manualButton.disabled=speakerButton.disabled=true;
try{const r=await fetch('/follow',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
const result=await r.json();if(!r.ok)throw new Error(result.detail||'Could not change following mode');showFollow(result);
}catch(e){msg.textContent=e.message;}finally{followPending=false;poll();}}
async function poll(){if(pollPending)return;pollPending=true;try{const r=await fetch('/status',{cache:'no-store'});
if(!r.ok)throw new Error();const snapshot=await r.json();state.textContent=JSON.stringify(snapshot,null,2);
showEnrollment(snapshot.enrollment);showFollow(snapshot);}catch(e){state.textContent='Tracker unavailable';
faceMessage.textContent='Tracker unavailable; enrollment status cannot be confirmed.';faceToggle.disabled=true;
manualButton.disabled=speakerButton.disabled=true;clearPerson.disabled=true;
for(const button of people.children)button.disabled=true;document.getElementById('follow-status').textContent='Tracker unavailable';}
finally{pollPending=false;}}setInterval(poll,500);poll();</script></body></html>"""


def create_api(control, preview=None):
    """HTTP handlers only enqueue commands or read the tracker's published snapshot."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="Phone tracker")

    @app.post("/commands", status_code=202)
    def command(command: Literal["closer", "farther", "freeze", "resume", "release", "grip", "test_motion",
                                 "deactivate", "activate", "up", "down", "left", "right", "rotate", "party", "home", "landscape", "portrait",
                                 "ambient", "spotlight"] = Body(embed=True)):
        try:
            control.submit(command)
        except queue.Full:
            raise HTTPException(503, "Command queue is full")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"accepted": True, "command": command}

    @app.post("/find", status_code=202)
    def find(name: str | None = Body(None, embed=True)):
        """Go to a person: follow their enrolled face, sweeping to find them. name=null stops."""
        try:
            control.find(name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"accepted": True, "finding": name}

    @app.post("/follow")
    def follow(mode: Literal["manual", "speaker"] = Body(embed=True)):
        try:
            control.set_follow_mode(mode)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return control.snapshot()

    @app.post("/voice", status_code=202)
    def voice(speaker: str = Body(embed=True), confidence: float = Body(0.0, embed=True),
              level: float | None = Body(None, embed=True)):
        """Speaker-ID results from acoustic/voice_bridge.py ("silence", "unknown" or a name), plus loudness in dBFS."""
        if not np.isfinite(confidence) or not -1 <= confidence <= 1:
            raise HTTPException(422, "confidence must be finite and between -1 and 1")
        if level is not None and (not np.isfinite(level) or not -120 <= level <= 0):
            raise HTTPException(422, "level must be dBFS between -120 and 0")
        control.hear(speaker, confidence, level)
        return {"accepted": True}

    @app.post("/faces/enrollment", status_code=202)
    def enroll_face(name: str = Body(embed=True)):
        try:
            return control.enrollment.start(name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/faces/enrollment/cancel", status_code=202)
    def cancel_enrollment():
        return control.enrollment.cancel()

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


def picture_detail(frame):
    """How much detail the picture has (grey-level spread). A hand over the lens gives a few; a room 40+."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60))
    return float(small.std())


def annotate(frame, boxes, target, aim, deadband, status, face_labels=None):
    h, w = frame.shape[:2]
    db = int(deadband * w / 2)
    cv2.line(frame, (w // 2 - db, 0), (w // 2 - db, h), (80, 80, 80), 1)
    cv2.line(frame, (w // 2 + db, 0), (w // 2 + db, h), (80, 80, 80), 1)
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for index, b in enumerate(boxes):
        x, y, bw, bh, kind = b
        color = (0, 255, 0) if b is target else (0, 160, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = f"{kind} {bw}px" if kind == "face" or kind.startswith("face:") else kind
        label = (face_labels or {}).get(index, label)
        cv2.putText(frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
    if not isinstance(device, (str, bytes, os.PathLike)):
        return []  # not a real port (e.g. a test double)
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
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # Two programs driving the arm at once fight over it (garbled packets, joints yanked both ways).
    others = other_users_of(port)
    if others:
        raise RuntimeError(
            "Another program is already using the arm, so we'd be fighting it:\n  " + "\n  ".join(others)
            + "\nStop it first (kill <PID>), after checking with whoever started it."
        )

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
    port = robot.config.port
    if not args.keep_grip:
        # Read it back and retry: a garbled bus packet can drop the command and leave the clamp locked.
        for attempt in range(5):
            try:
                robot.bus.disable_torque(GRIP, num_retry=3)
                if robot.bus.read("Torque_Enable", GRIP, num_retry=3) == 0:
                    log.info("Gripper motor off: you can open it by hand (use --keep-grip to keep it clamped)")
                    break
            except BUS_ERRORS as e:
                log.debug(f"Gripper off attempt {attempt + 1} failed: {e}")
            time.sleep(0.1)
        else:
            log.warning("Couldn't confirm the gripper motor is off; trying again on a fresh connection")
    try:
        robot.disconnect()
    except BUS_ERRORS as e:
        log.warning(f"Disconnect hit a bus error ({e})")
    if not args.keep_grip:
        free_gripper_fresh(port)


def free_gripper_fresh(port):
    """The manual fix that always worked, as the very last step: a brand-new connection that only
    turns the gripper motor off, then disconnects without touching the other motors."""
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        r = SO101Follower(SO101FollowerConfig(port=port, id=FOLLOWER_ID))
        r.bus.connect(handshake=False)  # skip the all-motors check: only the gripper matters here
        r.bus.disable_torque(GRIP, num_retry=3)
        r.bus.disconnect(disable_torque=False)
        log.info("Gripper is free")
    except Exception as e:  # never let this hide the real reason the program stopped
        log.warning(f"Couldn't free the gripper on a fresh connection ({e})")


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

    def camera_pitch(self, tilt_offset=0.0, mounting_offset=0.0):
        """Upward camera pitch in degrees; gripper-frame +Z points forward in this URDF."""
        q = self.q.copy()
        q[3] += tilt_offset
        forward = self.kin.forward_kinematics(q)[:3, 2]
        return float(np.degrees(np.arctan2(forward[2], np.hypot(forward[0], forward[1])))) + mounting_offset

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


def valid_ik(hik, q, pos_err, ang_err):
    jump = float(np.abs(q - hik.q)[1:4].max())
    return np.isfinite(q).all() and pos_err < 0.01 and ang_err < 5 and jump < MAX_IK_JUMP_DEG


def plane_action(q, tilt_offset=0.0):
    return {
        "shoulder_lift.pos": float(q[1]),
        "elbow_flex.pos": float(q[2]),
        f"{TILT}.pos": float(q[3] + tilt_offset),
    }


def gripper_pitch(kin):
    """Degrees the gripper points above (+) or below (-) horizontal, for the pose last passed to FK."""
    d = kin.robot.get_T_world_frame(kin.target_frame_name)[:3, 3] - kin.robot.get_T_world_frame("wrist_link")[:3, 3]
    return float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))), d / np.linalg.norm(d)


def lowest_point(kin):
    """Height (m) of the lowest of the elbow, wrist and gripper tip, for the pose last passed to FK."""
    return min(kin.robot.get_T_world_frame(f)[2, 3] for f in ("lower_arm_link", "wrist_link", kin.target_frame_name))


def rotation_about(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def plan_home(urdf, joints, args):
    """Joint waypoints from the current pose to the default "ready" pose, or (None, reason).

    Ready pose: base centred, gripper tip --home-reach forward and --home-height up, gripper pointing
    level (--home-pitch). It is solved once with IK from a neutral seed, then reached by moving every
    joint together in a straight line, in <=1 deg steps, checked so no part of the arm ever goes lower
    than it already is (it never sweeps down into the table). wrist_roll and the gripper are untouched.
    """
    from lerobot.model.kinematics import RobotKinematics

    kin = RobotKinematics(urdf, "gripper_frame_link", ARM_JOINTS)
    roll = joints[ROLL]
    seed = np.array([0.0, -30.0, 30.0, 0.0, roll])  # upper arm up, forearm forward
    target = kin.forward_kinematics(seed).copy()
    pitch, d = gripper_pitch(kin)
    level = rotation_about(np.cross(d, [0, 0, 1]) / np.linalg.norm(np.cross(d, [0, 0, 1])),
                           np.radians(args.home_pitch - pitch))
    target[:3, :3] = level @ target[:3, :3]
    target[0, 3], target[2, 3] = args.home_reach, args.home_height
    q = seed.copy()
    for _ in range(150):
        q = kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=1.0)
        q[0], q[4] = 0.0, roll
    reached = kin.forward_kinematics(q)
    pos_err = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
    pitch_err = gripper_pitch(kin)[0] - args.home_pitch
    if not np.isfinite(q).all() or pos_err > 0.01 or abs(pitch_err) > 3:
        return None, f"can't reach the ready pose ({pos_err * 1000:.0f} mm / {pitch_err:+.0f} deg off)"

    q0 = np.array([joints[j] for j in ARM_JOINTS], dtype=float)
    q_goal = q.copy()
    q_goal[4] = q0[4]
    kin.forward_kinematics(q0)
    floor = min(lowest_point(kin), 0.03) - 0.005  # never lower than it already is (or 3 cm)
    n = max(1, int(np.ceil(np.abs(q_goal - q0)[:4].max() / 1.0)))
    path = [q0 + (q_goal - q0) * i / n for i in range(1, n + 1)]
    for i, qi in enumerate(path):
        kin.forward_kinematics(qi)
        if lowest_point(kin) < floor:
            return None, (f"the way there would dip to {lowest_point(kin) * 100:.0f} cm "
                          f"({100 * i // n}% of the way); lift the arm a little by hand first")
    return path, None


def drive_joints(robot, path, step_s=0.05):
    """Send pan/lift/elbow/wrist waypoints (from plan_home) one by one."""
    for q in path:
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS[:4], q[:4])})
        time.sleep(step_s)


def move_gripper(robot, start, end, seconds=1.0, steps=20):
    for s in range(1, steps + 1):
        robot.send_action({f"{GRIP}.pos": float(start + (end - start) * s / steps)})
        time.sleep(seconds / steps)


# Bus failures: ConnectionError for garbled/missing replies, RuntimeError when a servo reports an error
# such as "Overload error" (its overload protection has tripped).
BUS_ERRORS = (ConnectionError, RuntimeError)
# The gripper uses its whole calibrated travel (0 = one end stop, 100 = the other), stopping this little
# short of each end. A servo left pushing on an end stop trips its overload protection, so drive_gripper
# also eases off near the ends and stops pushing wherever the jaws end up.
GRIP_END_MARGIN = 1.0
GRIP_SLOW_ZONE = 15.0  # last part of a move done at holding force, so the end stop isn't hit at full force


def set_grip_torque(robot, percent):
    """Live gripper force limit, % of the motor's max (LeRobot caps the gripper at 50 %)."""
    try:
        robot.bus.write("Torque_Limit", GRIP, int(np.clip(percent, 0, 50) * 10), num_retry=3)
    except BUS_ERRORS as e:
        log.warning(f"Couldn't set the gripper force limit ({e})")


def prepare_fast_gripper(robot):
    """Servo settings for the quickest gripper: no speed cap and the factory position gain.

    LeRobot already sets the fastest acceleration (254) and halves the gain to 16 for smooth arm motion;
    the gripper only opens and closes, so it gets the snappier factory gain of 32.
    """
    try:
        if robot.bus.read("Goal_Velocity", GRIP, normalize=False, num_retry=3) != 0:
            robot.bus.write("Goal_Velocity", GRIP, 0, normalize=False, num_retry=3)  # 0 = full speed
        robot.bus.write("P_Coefficient", GRIP, 32, normalize=False, num_retry=3)
    except BUS_ERRORS as e:
        log.warning(f"Couldn't speed up the gripper ({e})")


def drive_gripper(robot, target, move_torque, hold_torque, timeout=4.0, stall_s=0.15):
    """Send the gripper straight to `target` with one command so the servo moves at full speed.

    This deliberately skips the per-command max_relative_target cap, which exists for arm swings. Force
    is `move_torque` % for most of the move and `hold_torque` % for the last GRIP_SLOW_ZONE and after;
    pass the holding force for any move that could close on something. When the move ends (arrived, or
    stopped short on the phone or an end stop) the goal is set to where the jaws are, so the servo is
    never left pushing. Raises BUS_ERRORS if the servo can't be commanded. Returns (final position,
    True if it stopped short of the target).
    """
    target = float(np.clip(target, GRIP_END_MARGIN, 100.0 - GRIP_END_MARGIN))
    set_grip_torque(robot, move_torque)
    robot.bus.write("Goal_Position", GRIP, target, num_retry=3)
    t0 = time.monotonic()
    pos, last, moved_at, blocked, eased = None, None, t0, False, move_torque <= hold_torque
    while time.monotonic() - t0 < timeout:
        time.sleep(0.01)
        try:
            pos = float(robot.bus.read("Present_Position", GRIP, num_retry=1))
        except BUS_ERRORS:
            continue
        if not eased and abs(pos - target) < GRIP_SLOW_ZONE:
            set_grip_torque(robot, hold_torque)  # arrive at the end gently
            eased = True
        if abs(pos - target) < 1.5:
            break
        if last is None or abs(pos - last) > 0.3:
            moved_at = time.monotonic()
        elif time.monotonic() - moved_at > stall_s:
            blocked = True
            break
        last = pos
    set_grip_torque(robot, hold_torque)
    if pos is not None:
        robot.bus.write("Goal_Position", GRIP, pos, num_retry=3)  # stop pushing where the jaws are
    return pos, blocked


def pan_bounds(args, robot):
    """Base headings the arm may face: West to East through North, never south of them.

    North is --north-pan (the calibrated forward direction, 0 deg = centre of the base's travel).
    --pan-limit is how far either side of North it may turn, capped at 90 deg (due West/East), and
    the calibrated travel of the joint is respected too.
    """
    half = float(np.clip(args.pan_limit, 0.0, 90.0))
    lo, hi = args.north_pan - half, args.north_pan + half
    limits = joint_limits(robot)
    if limits:
        lo, hi = max(lo, limits[PAN][0] + 2.0), min(hi, limits[PAN][1] - 2.0)
    return lo, hi


def joint_limits(robot):
    """Calibrated travel of each arm joint, in the same degrees as the actions; None if unavailable."""
    try:
        limits = {}
        for j in ARM_JOINTS:
            cal, motor_id = robot.bus.calibration[j], robot.bus.motors[j].id
            ends = [robot.bus._normalize({motor_id: v})[motor_id] for v in (cal.range_min, cal.range_max)]
            limits[j] = (min(ends), max(ends))
        return limits
    except Exception as e:
        log.warning(f"Couldn't read joint limits ({e})")
        return None

# Motor test (test_motion command): degrees each joint moves. Pan and wrist roll sweep both ways;
# shoulder lift, elbow and wrist flex only move in the direction that raises the phone.
TEST_MOTION_DEG = {"shoulder_pan": 30.0, "shoulder_lift": 15.0, "elbow_flex": 40.0, "wrist_flex": 25.0,
                   "wrist_roll": 40.0}
SWEEP_JOINTS = ("shoulder_pan", "wrist_roll")


def plan_test_motion(urdf, start, scale, limits=None, margin=5.0):
    """Per-joint signed amplitudes for the motor test, chosen with the arm model so the phone goes up.

    Each lifting joint gets the sign that raises the gripper tip from `start`; then the whole path is
    checked and the lifting moves are halved until the tip never dips more than 1 cm below where it
    started. Without the model only the sweeping joints move. Returns (amplitudes, predicted rise in m).
    """
    amps = {j: TEST_MOTION_DEG[j] * scale for j in ARM_JOINTS}
    if limits is None:
        log.warning("Motor test: joint limits unknown; using half-size moves")
        amps = {j: a / 2 for j, a in amps.items()}

    def within_limits(j, a):
        """Shrink amplitude `a` so the joint stays `margin` inside its travel (both ways for sweeps)."""
        if limits is None:
            return a
        lo, hi = limits[j][0] + margin, limits[j][1] - margin
        room_up, room_down = max(0.0, hi - start[j]), max(0.0, start[j] - lo)
        if j in SWEEP_JOINTS:
            return min(abs(a), room_up, room_down)
        return min(a, room_up) if a >= 0 else max(a, -room_down)
    try:
        from lerobot.model.kinematics import RobotKinematics
        kin = RobotKinematics(urdf, "gripper_frame_link", ARM_JOINTS)
    except Exception as e:
        log.warning(f"Motor test: no arm model ({e}); only sweeping {', '.join(SWEEP_JOINTS)}")
        return {j: (within_limits(j, amps[j]) if j in SWEEP_JOINTS else 0.0) for j in ARM_JOINTS}, None

    q0 = np.array([start[j] for j in ARM_JOINTS], dtype=float)
    z = lambda q: float(kin.forward_kinematics(q)[2, 3])
    z0 = z(q0)
    for i, j in enumerate(ARM_JOINTS):
        if j not in SWEEP_JOINTS:
            step = np.zeros(len(ARM_JOINTS))
            step[i] = 2.0
            amps[j] = within_limits(j, abs(amps[j]) if z(q0 + step) >= z0 else -abs(amps[j]))
        else:
            amps[j] = within_limits(j, amps[j])
    lift = np.array([0.0 if j in SWEEP_JOINTS else amps[j] for j in ARM_JOINTS])
    for _ in range(4):
        heights = [z(q0 + lift * phase) for phase in np.linspace(0, 1, 21)]
        if min(heights) >= z0 - 0.01:
            break
        lift /= 2
    else:
        lift[:] = 0.0
        heights = [z0]
    amps.update({j: float(lift[i]) for i, j in enumerate(ARM_JOINTS) if j not in SWEEP_JOINTS})
    return amps, max(heights) - z0


def exercise_joints(robot, amps, seconds=10.0, period=2.0, should_stop=lambda: False, gripper=None,
                    gripper_period=0.8):
    """Mobile-to-motor check: move the arm joints by `amps` (degrees) and back, repeatedly.

    Sweeping joints go +/- amplitude; the others go out to amplitude and back. With
    gripper=(closed, open) the gripper also snaps fully closed and open each `gripper_period` seconds
    with direct full-speed commands, ending open (the caller sets its force). Starts and ends at the
    current pose. Returns False if stopped early; it still eases back rather than jumping.
    """
    base = read_joints(robot)
    joints = list(ARM_JOINTS)
    snaps = -1

    def offsets(t):
        ramp = min(1.0, t / 0.5)  # ease the sweeps in so they don't start at full speed
        lift = 0.5 - 0.5 * np.cos(2 * np.pi * t / period)
        sweep = ramp * np.sin(2 * np.pi * t / period)
        return {j: amps[j] * (sweep if j in SWEEP_JOINTS else lift) for j in ARM_JOINTS}

    def send(off):
        robot.send_action({f"{j}.pos": float(base[j] + off[j]) for j in joints})

    t0, last, completed = time.monotonic(), {j: 0.0 for j in joints}, True
    while (t := time.monotonic() - t0) < seconds:
        if should_stop():
            completed = False
            break
        last = offsets(t)
        send(last)
        if gripper and int(t / (gripper_period / 2)) != snaps:  # closed for half a period, then open
            snaps = int(t / (gripper_period / 2))
            try:
                robot.bus.write("Goal_Position", GRIP, float(gripper[snaps % 2]), num_retry=1)
            except BUS_ERRORS as e:
                log.warning(f"Motor test: gripper stopped responding ({e}); continuing without it")
                gripper = None
        time.sleep(0.05)
    if gripper:
        try:
            robot.bus.write("Goal_Position", GRIP, float(gripper[1]), num_retry=3)
        except BUS_ERRORS as e:
            log.warning(f"Motor test: couldn't reopen the gripper ({e})")
    for s in range(1, 21):
        send({j: last[j] * (1 - s / 20) for j in joints})
        time.sleep(0.05)
    if gripper:
        try:  # it ended at the open end: stop pushing there
            robot.bus.write("Goal_Position", GRIP, float(robot.bus.read("Present_Position", GRIP, num_retry=2)), num_retry=3)
        except BUS_ERRORS as e:
            log.warning(f"Motor test: couldn't relax the gripper ({e})")
    return completed


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
    """Shared commands; only the tracking loop mutates control state or moves motors.

    deactivate/activate are the master switch: while deactivated the arm holds its pose and every
    other motion command (including gesture freeze/resume) is ignored until `activate`. Release and
    grip stay available so the phone can still be taken out.
    """
    COMMANDS = ("closer", "farther", "freeze", "resume", "release", "grip", "test_motion", "deactivate", "activate",
                "up", "down", "left", "right", "rotate", "party", "home", "landscape", "portrait", "ambient", "spotlight")
    DPAD = ("up", "down", "left", "right", "rotate", "party", "home", "landscape", "portrait")

    def __init__(self, args):
        self.args = args
        self.commands = queue.Queue(maxsize=32)
        self.frozen = False
        self.deactivated = False
        # Set as soon as deactivate is submitted (from the API thread), so a running motor test stops at once.
        self.deactivate_requested = threading.Event()
        self.voice = {"speaker": None, "confidence": 0.0, "at": 0.0}  # latest result from acoustic/voice_bridge.py
        self.find_name = None  # Existing mobile Go to selection.
        self.follow_mode = getattr(args, "follow_mode", "manual")  # "speaker": whoever talks is looked for
        self.view_mode = "spotlight"  # phone Mode: "spotlight" follows people, "ambient" sweeps West-East
        self.last_speaker = {"speaker": None, "confidence": 0.0, "at": 0.0}
        self.voice_listener_state = "off"
        self.voice_names = list(getattr(args, "voice", None) or [])
        if not self.voice_names:
            try:
                with np.load(ROOT / "acoustic" / "voiceprints.npz", allow_pickle=False) as data:
                    self.voice_names = list(map(str, data["speaker_names"]))
            except (OSError, ValueError, KeyError):
                pass
        self.face_names = []   # enrolled faces, published for the phone's buttons
        self.reach0 = self.reach = self.requested_reach = 0.0
        self.depth = DepthFollower(args) if args.distance is not None else None
        self.last_command = None
        self.lock = threading.Lock()
        self.status = {}
        try:
            from .faces import FaceEnrollment
        except ImportError:
            from faces import FaceEnrollment
        self.enrollment = FaceEnrollment(getattr(args, "face_db", Path(MODELS_DIR) / "faces.npz"),
                                         getattr(args, "face_threshold", 0.45))
        self.publish(mode="starting", held=True, target_visible=False)

    def submit(self, command):
        if command not in self.COMMANDS:
            raise ValueError("Unknown command")
        if command == "deactivate":
            self.deactivate_requested.set()
        self.commands.put_nowait(command)

    def apply(self, command, held=True):
        if command == "deactivate":
            self.deactivated = True
            self.requested_reach = self.reach  # cancel unfinished manual depth moves
            return "applied"
        if command in ("ambient", "spotlight"):
            self.view_mode = command  # a mode can be chosen while deactivated; it takes effect on activate
            return "applied"
        if command == "activate":
            self.deactivated = self.frozen = False
            self.deactivate_requested.clear()
            return "applied"
        if self.deactivated:
            return "ignored: deactivated"
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

    def hear(self, speaker, confidence, level=None):
        """Called from the API thread for every speaker-ID hop (about twice a second)."""
        with self.lock:
            self.voice = {"speaker": speaker, "confidence": float(confidence), "at": time.monotonic(),
                          "level": None if level is None else float(level)}
            if (confidence >= getattr(self.args, "voice_min_confidence", 0.45)
                    and speaker in self.voice_names and speaker in self.face_names):
                self.last_speaker = self.voice.copy()

    def find(self, name):
        """Phone "Go to": look for and follow `name`; None goes back to normal tracking."""
        with self.lock:
            if name is not None and name not in self.face_names:
                raise ValueError(f"{name!r} has no enrolled face")
            if name is not None and self.deactivated:
                raise RuntimeError("the arm is deactivated")
            self.find_name = name
            # A mobile tap selects immediately; the next accepted voice result can override it.
            self.last_speaker = {"speaker": None, "confidence": 0.0, "at": 0.0}

    def set_follow_mode(self, mode):
        with self.lock:
            if mode == "speaker" and not self.voice_names:
                raise RuntimeError("No voice enrollments available; enroll voices first.")
            if mode != self.follow_mode:
                self.follow_mode = mode
                self.last_speaker = {"speaker": None, "confidence": 0.0, "at": 0.0}
            if mode == "speaker" and self.voice_listener_state.startswith("error"):
                self.voice_listener_state = "off"  # Clicking Follow speaker retries a failed worker.

    def following(self):
        with self.lock:
            if self.follow_mode == "speaker":
                v = self.last_speaker
                if (time.monotonic() - v["at"] <= getattr(self.args, "voice_hold", 4.0)
                        and v["speaker"] in self.face_names):
                    return v["speaker"]
            return self.find_name

    def heard(self, names, within):
        """Which of `names` the voice listener identified in the last `within` seconds, else None.

        Silence/unknown results preserve the last accepted identity through short pauses.
        """
        with self.lock:
            v = self.last_speaker
            fresh = time.monotonic() - v["at"] <= within
            return v["speaker"] if fresh and v["speaker"] in names else None

    def publish(self, **fields):
        with self.lock:
            self.status.update(fields)
            heard = self.voice["speaker"]
            self.status.update(voice={"targets": self.voice_names, "speaker": heard,
                                      "confidence": self.voice["confidence"],
                                      "age": round(time.monotonic() - self.voice["at"], 1) if heard else None,
                                      "hold": getattr(self.args, "voice_hold", None),
                                      "level": self.voice.get("level")})
            v = self.last_speaker
            recent = (v["speaker"] is not None and time.monotonic() - v["at"] <= getattr(self.args, "voice_hold", 4.0)
                      and v["speaker"] in self.face_names)
            target = v["speaker"] if self.follow_mode == "speaker" and recent else self.find_name
            listener = self.voice_listener_state
            if listener == "running":
                listener = "connected" if heard and time.monotonic() - self.voice["at"] <= 3 else "waiting for audio"
            if listener.startswith("error"):
                feedback = "Voice listener failed. Click Follow speaker to retry; check acoustic/voice_bridge.log."
            elif self.follow_mode != "speaker":
                feedback = "Manual mode: select a name above, or click Follow speaker to enable voice switching."
            elif listener == "waiting for audio":
                feedback = ("No microphone audio received. Start the laptop sender and SSH tunnel (setup below)."
                            if self.args.voice_local_mic is None else "No audio received from the robot computer microphone.")
            elif heard == "silence":
                feedback = "Microphone audio connected; waiting for speech."
            elif heard == "unknown":
                feedback = "Speech detected, but no enrolled voice matched."
            elif heard and heard not in self.voice_names:
                feedback = f"Heard {heard}, but that name is not enabled for voice following."
            elif heard and heard not in self.face_names:
                feedback = f"Heard {heard}, but their face must be enrolled before following."
            elif heard and self.voice["confidence"] < 0.5:
                feedback = f"Voice match for {heard} is below the 0.5 similarity threshold."
            elif heard:
                feedback = f"Recognized {heard}; following their face or searching if it is out of view."
            else:
                feedback = "Starting the voice listener..."
            self.status.update(finding=self.find_name, faces=list(self.face_names),
                               follow_mode=self.follow_mode, active_target=target, voice_listener=listener,
                               voice_feedback=feedback)
            self.status.update(view_mode=self.view_mode)
            self.status.update(frozen=self.frozen, deactivated=self.deactivated, dry_run=self.args.dry_run,
                target_distance=self.args.distance, estimated_distance=self.depth.distance if self.depth else None,
                reach=self.reach, requested_reach=None if self.depth else self.requested_reach,
                reach_min=self.reach0 - self.args.depth_back, reach_max=self.reach0 + self.args.depth_forward,
                at_reach_limit=(self.reach <= self.reach0 - self.args.depth_back + 1e-6
                                or self.reach >= self.reach0 + self.args.depth_forward - 1e-6),
                last_command=self.last_command)

    def snapshot(self):
        self.publish()
        with self.lock:
            snapshot = self.status.copy()
        snapshot["enrollment"] = self.enrollment.snapshot()
        return snapshot


def start_voice(args, recognizer=None):
    """Face recognizer for the voice targets, plus acoustic/voice_bridge.py in the speakerid env.

    The bridge listens all the time and POSTs each speaker-ID result to this tracker's /voice. Names
    whose face isn't enrolled yet are allowed: they can be enrolled from the tracker page while it runs.
    """
    try:
        from .faces import FaceRecognizer
    except ImportError:
        from faces import FaceRecognizer
    if not args.preview_port:
        raise SystemExit("--voice needs the tracker API: don't pass --preview-port 0")
    recognizer = recognizer or FaceRecognizer(args.face_db, args.face_threshold)
    missing = [name for name in args.voice if name not in recognizer.names]
    if missing:
        log.warning(f"Voice: no enrolled face for {', '.join(missing)}; hearing them won't move the arm until "
                    f"their face is enrolled (Enroll face on the tracker page)")
    command = [args.voice_python, "-u", str(ROOT / "acoustic" / "voice_bridge.py"),
               "--tracker-url", f"http://127.0.0.1:{args.preview_port}"]
    if args.voice_local_mic is not None:
        command += ["--local-mic", "--device", str(args.voice_local_mic)]
    log_path = ROOT / "acoustic" / "voice_bridge.log"
    listener = subprocess.Popen(command, cwd=ROOT / "acoustic", stdin=subprocess.DEVNULL,
                                stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    source = f"microphone {args.voice_local_mic} on this machine" if args.voice_local_mic is not None \
        else "the laptop mic (run acoustic/mac_mic_sender.py and the SSH tunnel)"
    log.info(f"Voice: listening for {', '.join(args.voice)} via {source}; log in {log_path}")
    return recognizer, listener


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


def party(robot, args, stop, commands, should_stop=lambda: False):
    """Dance until Party is pressed again (or Ctrl+C), then ease back to where it started.

    Runs its own ~40 Hz loop (tracking is limited by the camera). Swings grow over the first second,
    wrist_roll stays in range, and the gripper is never touched. Other commands that arrive while
    dancing are put back in the queue for the tracking loop.
    """
    start = read_joints(robot)
    log.info(f"PARTY MODE (speed x{args.party_speed:g}) - press Party again to stop")
    deferred, last, t0 = [], {}, time.time()
    while not stop.is_set() and not should_stop():
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
            if joint == PAN and getattr(args, "pan_bounds", None):
                lo, hi = args.pan_bounds  # never swing past West/East (or further out than it already is)
                angle = np.clip(angle, min(lo, start[joint]), max(hi, start[joint]))
            action[f"{joint}.pos"] = float(angle)
        robot.send_action(action)
        last = action
        time.sleep(max(0.0, 1 / 40 - (time.time() - t0 - t)))
    # Ease back to the starting pose over 1 s so tracking picks up where it left off.
    for i in range(1, 21):
        if should_stop():
            break
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
    cam = robot = preview = gestures = voice_listener = None
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
        if args.follow_face:
            try:
                from .faces import FaceRecognizer
            except ImportError:
                from faces import FaceRecognizer
            detector = FaceRecognizer(args.face_db, args.face_threshold, args.follow_face)
            control.enrollment.recognizer = detector
            log.info("Detector: YuNet + SFace, following %s", args.follow_face)
        elif args.detector == "face+body" or (args.detector == "auto" and have_models):
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
        recognizer = voice_listener = None
        if args.voice:
            recognizer, voice_listener = start_voice(args)
            control.voice_listener_state = "running"
        else:
            try:
                try:
                    from .faces import FaceRecognizer
                except ImportError:
                    from faces import FaceRecognizer
                recognizer = FaceRecognizer(args.face_db, args.face_threshold)
            except Exception as e:  # models or database missing: everything else still works
                log.warning(f"Face recognition unavailable ({e}); the phone's Go to buttons are disabled")
        if recognizer is not None and control.enrollment.recognizer is None:
            control.enrollment.recognizer = recognizer  # faces enrolled from the page count at once
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
        pan_lo = pan_hi = None
        hik = None
        tilt_offset = 0.0  # height mode: extra wrist tilt on top of the IK solution
        z_goal = 0.0
        if robot:
            joints = read_joints(robot)
            pan_goal = pan_home = joints[PAN]
            tilt_goal = tilt_home = joints[TILT]
            log.info(f"Start pose: {PAN}={pan_home:.1f} deg, {TILT}={tilt_home:.1f} deg")
            pan_lo, pan_hi = args.pan_bounds = pan_bounds(args, robot)
            log.info(f"Base headings: {pan_lo:.0f} to {pan_hi:.0f} deg (North = {args.north_pan:g}); never further south")
            if not pan_lo <= pan_home <= pan_hi:
                log.warning(f"The base starts outside that range ({pan_home:.1f} deg); it will ease back toward North")
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
        person_seen = 0.0  # last time anyone was detected
        cover_start = None  # when the picture went blank; False = went blank with nobody around (ignored)
        last_share_check = 0.0
        pull_warned = False
        if robot:
            grip_home = read_joints(robot)[GRIP]
            if args.gripper_open_sign and args.squeeze > 0:
                grip_home = gentle_squeeze(robot, args, grip_home)
            if args.gripper_open_sign:
                prepare_fast_gripper(robot)
                log.info(f"Gripper: holding at {grip_home:.1f}; release: 'r'+ENTER or the page button, grip: 'g'+ENTER")
                if args.release_on_pull:
                    pull = PullDetector(args.pull_threshold, args.pull_time)
                    log.info(f"Release on pull: ON (threshold {args.pull_threshold:g}, {args.pull_time:g} s)")
            elif args.release_on_pull:
                log.warning("--release-on-pull ignored: run --test-gripper and pass --gripper-open-sign first")

        # The two ends of the gripper's calibrated travel (drive_gripper stops GRIP_END_MARGIN short of them).
        gripper_closed = 0.0 if args.gripper_open_sign > 0 else 100.0
        gripper_open = 100.0 - gripper_closed
        roll_home = read_joints(robot)[ROLL] if robot else 0.0
        # wrist_roll spans about -180..180 deg: turn whichever way stays inside that range.
        rotate_by = args.rotate_deg if abs(roll_home + args.rotate_deg) <= 170 else -args.rotate_deg
        rotated = False  # phone turned 90 deg from where it started

        def release(reason):
            nonlocal held, pull_warned
            log.info(f"Releasing phone ({reason})")
            drive_gripper(robot, gripper_open, args.grip_move_torque, args.grip_torque)  # all the way open
            held, pull_warned = False, False

        def grip():
            nonlocal held, grip_home
            # Close all the way. At holding force alone the jaws crawl (~40 units/s), so the stroke uses
            # --grip-close-torque; the last stretch, and any stop on the phone, drop straight to holding
            # force (contact is noticed within 0.08 s), then it presses lightly on the phone.
            pos, blocked = drive_gripper(robot, gripper_closed, args.grip_close_torque, args.grip_torque, stall_s=0.08)
            shut = pos is not None and abs(pos - gripper_closed) <= GRIP_END_MARGIN + 3
            if pos is not None:
                grip_home = pos
                if blocked and not shut:
                    grip_home = pos - args.gripper_open_sign * 1.5
                    robot.bus.write("Goal_Position", GRIP, float(grip_home), num_retry=3)
            if shut:
                outcome = "fully shut, nothing in the jaws"
            elif blocked:
                outcome = "on the phone"
            else:
                outcome = "stopped before closing fully - the gripper was too slow"
            log.info(f"Gripped at {grip_home:.1f} ({outcome}); tracking resumes in 1 s")
            time.sleep(1.0)
            if pull:
                pull.reset()
            held = True

        def clamp_pan(value):
            """Keep a pan goal inside West..East; if the base is already outside, never further out."""
            return float(np.clip(value, min(pan_lo, pan_goal), max(pan_hi, pan_goal)))

        def sweep_pan(speed, step_dt):
            """One step of the West -> North -> East -> North -> West sweep. True when it turned around."""
            nonlocal pan_goal, search_dir
            new = pan_goal + search_dir * speed * step_dt
            turned = not pan_lo <= new <= pan_hi
            if turned:
                search_dir = -search_dir
                new = clamp_pan(new)
            pan_goal = float(new)
            return turned

        def nudge(direction):
            """D-pad: move one step and take over from tracking, which stays paused until `resume`.

            left/right turn the base (the camera's own left/right, as in --test-pan); up/down move the
            phone with the height IK in the same 4 mm steps and checks as tracking, or tilt the wrist
            when there is no IK.
            """
            nonlocal pan_goal, tilt_goal, z_goal
            control.frozen = True  # otherwise tracking would immediately undo the move
            control.requested_reach = control.reach
            if direction in ("left", "right"):
                sign = 1 if direction == "right" else -1
                new = clamp_pan(pan_goal + sign * args.pan_sign * args.dpad_pan_step)
                if new == pan_goal:
                    return "ignored: at the pan limit"
                pan_goal = new
                robot.send_action({f"{PAN}.pos": pan_goal})
                return "applied"
            up = 1 if direction == "up" else -1
            if hik:
                start = z_goal
                target = float(np.clip(start + up * args.dpad_height_step,
                                       hik.z0 - args.height_down, hik.z0 + args.height_up))
                steps = max(1, int(np.ceil(abs(target - start) / 0.004)))
                for i in range(1, steps + 1):
                    z_new = start + (target - start) * i / steps
                    q, pos_err, ang_err = hik.solve(z_new, control.reach)
                    if not valid_ik(hik, q, pos_err, ang_err):
                        break
                    hik.q, z_goal = q, z_new
                    robot.send_action(plane_action(hik.q, tilt_offset))
                    time.sleep(0.02)
                moved = z_goal - start
                if abs(moved) < 1e-4:
                    return "ignored: can't move further that way"
                return "applied" if abs(z_goal - target) < 1e-4 else f"partly applied ({moved * 100:+.1f} cm)"
            # Positive tilt points the camera down (see --test-tilt).
            new = float(np.clip(tilt_goal - up * args.tilt_sign * args.dpad_tilt_step,
                                tilt_home - args.tilt_limit, tilt_home + args.tilt_limit))
            if new == tilt_goal:
                return "ignored: at the tilt limit"
            tilt_goal = new
            robot.send_action({f"{TILT}.pos": tilt_goal})
            return "applied"

        def process_commands():
            nonlocal hik, z_goal, rotated, smoothed, pan_goal, pan_home, tilt_goal, tilt_home, tilt_offset
            for _ in range(32):
                try:
                    cmd = commands.get_nowait()
                except queue.Empty:
                    break
                if cmd in ("rotate", "party", "home", "landscape", "portrait") and control.deactivated:
                    result = "ignored: deactivated"
                elif cmd in ("rotate", "party", "home", "landscape", "portrait") and control.enrollment.busy:
                    result = "ignored: face enrollment in progress"
                elif cmd in ("rotate", "party", "home", "landscape", "portrait") and not robot:
                    result = "ignored: dry run"
                elif (cmd == "landscape" and rotated) or (cmd == "portrait" and not rotated):
                    result = "already applied"
                elif cmd in ("rotate", "landscape", "portrait"):  # the hand-twist gesture sends landscape/portrait
                    start = roll_home + (rotate_by if rotated else 0.0)
                    end = roll_home + (0.0 if rotated else rotate_by)
                    log.info(f"Rotating the phone {'back' if rotated else 'by'} {end - start:+.0f} deg")
                    for i in range(1, 31):  # slowly, over about 1.5 s
                        robot.send_action({f"{ROLL}.pos": float(start + (end - start) * i / 30)})
                        time.sleep(0.05)
                    rotated = not rotated
                    smoothed = None  # the view just turned; re-find the person
                    result = "applied"
                elif cmd == "home":
                    path, problem = plan_home(args.urdf, read_joints(robot), args)
                    if problem:
                        result = f"refused: {problem}"
                    else:
                        log.info(f"Moving to the ready pose ({len(path) * 0.05:.0f} s)")
                        drive_joints(robot, path)
                        # Track from here, as if the program had just started in this pose.
                        joints = read_joints(robot)
                        joints.update({j: float(v) for j, v in zip(ARM_JOINTS[:4], path[-1][:4])})
                        pan_goal = pan_home = joints[PAN]
                        tilt_goal = tilt_home = joints[TILT]
                        tilt_offset = 0.0
                        if hik:
                            hik = HeightIK(args.urdf, joints)
                            z_goal = hik.z0
                            control.reach0 = control.reach = control.requested_reach = hik.reach0
                        smoothed = None
                        result = "applied"
                elif cmd == "party":
                    control.publish(mode="party")
                    party(robot, args, stop, commands, should_stop=lambda: (
                        control.deactivate_requested.is_set() or control.enrollment.busy))
                    smoothed = None  # re-find the person after dancing
                    result = "applied"
                elif cmd in ("release", "grip"):
                    if not robot:
                        result = "ignored: dry run"
                    elif not args.gripper_open_sign:
                        result = "ignored: gripper direction is not configured"
                    elif cmd == "release" and held:
                        try:
                            release("command")
                            control.requested_reach = control.reach
                            result = "applied"
                        except BUS_ERRORS as e:
                            log.error(f"Release failed: {e}")
                            result = f"failed: gripper error ({e})"
                    elif cmd == "grip" and not held:
                        try:
                            grip()
                            result = "applied"
                        except BUS_ERRORS as e:
                            log.error(f"Grip failed: {e}")
                            result = f"failed: gripper error ({e})"
                    else:
                        result = "already applied"
                elif cmd in control.DPAD:
                    if control.deactivated:
                        result = "ignored: deactivated"
                    elif control.enrollment.busy:
                        result = "ignored: face enrollment in progress"
                    elif not robot:
                        result = "ignored: dry run"
                    else:
                        try:
                            result = nudge(cmd)
                        except BUS_ERRORS as e:
                            log.error(f"D-pad {cmd} failed: {e}")
                            result = f"failed: {e}"
                elif cmd == "test_motion":
                    if control.enrollment.busy:
                        result = "ignored: face enrollment in progress"
                    elif control.deactivated:
                        result = "ignored: deactivated"
                    elif not robot:
                        result = "ignored: dry run"
                    else:
                        control.publish(mode="test_motion")
                        scale = float(np.clip(args.test_motion_scale, 0.0, 1.5))
                        limits = joint_limits(robot)
                        if limits:
                            limits[PAN] = (max(limits[PAN][0], pan_lo), min(limits[PAN][1], pan_hi))
                        amps, rise = plan_test_motion(args.urdf, read_joints(robot), scale, limits)
                        # Opening the gripper drops a held phone, so it only joins in once released.
                        gripper = None
                        if args.gripper_open_sign and grip_home is not None and not held:
                            ends = (GRIP_END_MARGIN, 100.0 - GRIP_END_MARGIN)
                            gripper = tuple(float(np.clip(v, *ends)) for v in (gripper_closed, gripper_open))  # full travel
                            set_grip_torque(robot, args.grip_torque)  # snaps close on nothing, but a finger may be there
                        log.info("Motor test for 10 s: " + ", ".join(f"{j} {a:+.0f}" for j, a in amps.items())
                                 + (f" deg; phone rises up to {rise * 100:.0f} cm" if rise is not None else " deg")
                                 + ("; gripper opening/closing" if gripper else "; gripper stays shut (phone held)"))
                        done = exercise_joints(robot, amps, gripper=gripper, should_stop=lambda: (
                            stop.is_set() or control.deactivate_requested.is_set() or control.enrollment.busy))
                        if gripper:
                            set_grip_torque(robot, args.grip_torque)
                        result = "applied" if done else "stopped early"
                        if not gripper and held:
                            result += " (gripper skipped: release the phone first)"
                else:
                    if (cmd in ("closer", "farther") and robot and hik is None
                            and not control.frozen and not control.deactivated and held):
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

        voice_was = find_was = seek_was = None
        seek_seen, search_dir, search_level = 0.0, 1.0, 0
        last_control = time.monotonic()
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            # Someone may start another program on the arm while we run; fighting it garbles the bus
            # and crashes both, so stop cleanly as soon as it shows up.
            if robot and time.time() - last_share_check > 2.0:
                last_share_check = time.time()
                others = other_users_of(robot.config.port)
                if others:
                    log.error("Another program just started using the arm, stopping so we don't fight it:\n  "
                              + "\n  ".join(others))
                    break
            process_commands()
            if voice_listener is not None and voice_listener.poll() is not None:
                control.voice_listener_state = "error: voice listener stopped; check acoustic/voice_bridge.log"
                voice_listener = None
            if (control.follow_mode == "speaker" and voice_listener is None
                    and control.voice_listener_state == "off"):
                try:
                    args.voice = control.voice_names
                    recognizer, voice_listener = start_voice(args, recognizer)
                    control.voice_listener_state = "running"
                    control.enrollment.recognizer = recognizer
                except (OSError, ValueError, SystemExit) as exc:
                    control.voice_listener_state = f"error: {exc}"
                    log.error("Could not start voice listener: %s", exc)
            seq, frame = cam.read(seq)
            now_control = time.monotonic()
            dt, last_control = now_control - last_control, now_control
            enrolling = control.enrollment.busy
            control.enrollment.update(frame)
            if enrolling:
                smoothed = None
                control.requested_reach = control.reach
            if frame is None:
                control.publish(mode="deactivated" if control.deactivated else ("frozen" if control.frozen else "hold"), target_visible=False, motion_blocker="camera unavailable")
                log.warning("No new frame from the camera for 1 s")
                continue
            if rotated and args.camera_on_roll != "no":
                # The camera turned with the phone: turn the picture back so people are upright again.
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if args.camera_on_roll == "cw"
                                   else cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = frame.shape[:2]

            # Voice: while one of the --voice people was heard recently, follow only that person's face;
            # if it isn't in view, sweep the base to look for them.
            voice_name = (control.heard(control.voice_names, args.voice_hold)
                          if recognizer and control.follow_mode == "speaker" else None)
            voice_face_known = voice_name is not None and voice_name in recognizer.names
            if recognizer is not None:
                control.face_names = sorted(set(map(str, recognizer.names)))
            # Keep mobile Go to intact; the web mode decides whether accepted speech may override it.
            find_name = control.find_name if recognizer else None
            if find_name != find_was:
                if find_name:
                    control.frozen = False  # picking someone hands control back from the D-pad
                    log.info(f"Go to {find_name}: looking for them")
                else:
                    log.info("Go to: cleared")
                find_was = find_name
            seek_name = control.following() if recognizer else None
            seek_on = seek_name is not None and seek_name in recognizer.names
            if seek_name != seek_was:
                smoothed, seek_seen = None, 0.0  # switch people immediately, not via max_jump
                seek_was = seek_name
            if voice_name != voice_was:
                if voice_name is None:
                    log.info("Voice: back to normal tracking")
                elif voice_face_known:
                    log.info(f"Voice: heard {voice_name}, looking for them")
                else:
                    log.info(f"Voice: heard {voice_name}, but their face isn't enrolled; tracking as usual")
                voice_was = voice_name
            if enrolling:
                boxes = []
            elif seek_on:
                boxes = [b for b in recognizer.detect(frame) if b[4].startswith(f"face:{seek_name} ")]
                if boxes:
                    seek_seen = time.time()
            elif control.follow_mode == "speaker":
                boxes = []  # No valid person selected: hold instead of following an unrelated face.
            else:
                boxes = detector.detect(frame)
            face_labels = control.enrollment.labels(frame, boxes) if preview and not enrolling else {}
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
                    feedback = reply.get("gesture", {})
                    control.publish(gesture=feedback)
                    for hand in feedback.get("hands", []):
                        point = (int(hand["x"] * w), int(hand["y"] * h))
                        color = (0, 255, 0) if hand["in_zone"] else (0, 160, 255)
                        cv2.circle(frame, point, 8, color, 2)
                        text = (f'L sign ({hand.get("hand")} hand)' if hand.get("l_shape")
                                else f'pointing {hand["pointing"]}' if hand.get("pointing")
                                else f'{hand["label"]} {hand["score"]:.2f}')
                        cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    command = reply.get("command")
                    if args.swap_hands and command in ("landscape", "portrait"):
                        command = "portrait" if command == "landscape" else "landscape"
                    if command:
                        control.submit(command)
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
            if boxes:
                person_seen = now

            # A hand over the lens makes the picture almost uniform. It only counts if someone was in
            # view just before (a hand reaching for the phone), so a blank wall never releases it.
            detail = picture_detail(frame)
            if held and detail < args.cover_detail:
                if cover_start is None:
                    cover_start = now if now - person_seen < 2.0 else False
                if cover_start and now - cover_start >= args.cover_seconds:
                    if robot and args.gripper_open_sign:
                        release("camera covered")
                        control.requested_reach = control.reach
                    else:
                        log.info("Camera covered: would release the phone now (dry run)")
                    cover_start = None
            else:
                cover_start = None

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
            if control.deactivated:
                motion_blocker = "deactivated"
            elif enrolling or control.enrollment.busy:
                motion_blocker = "face enrollment"
            elif control.frozen:
                motion_blocker = "frozen"
            elif not held:
                motion_blocker = "phone released"
            elif control.view_mode == "ambient" and not seek_on:
                motion_blocker = "ambient sweep"
            elif target is None and seek_on and time.time() - seek_seen > 0.7:
                motion_blocker = f"searching for {seek_name}"
            elif target is None:
                motion_blocker = "target unavailable"
            elif freeze:
                motion_blocker = "pull detected"
            if robot and motion_blocker == "ambient sweep":
                sweep_pan(args.ambient_speed, min(dt, 0.1))
                robot.send_action({f"{PAN}.pos": pan_goal})
            elif (robot and motion_blocker in (None, "target unavailable")
                  and not pan_lo <= pan_goal <= pan_hi):
                # Started (or was left) facing south of West/East: ease back toward North.
                edge = pan_lo if pan_goal < pan_lo else pan_hi
                pan_goal = float(pan_goal + np.clip(edge - pan_goal, -30 * min(dt, 0.1), 30 * min(dt, 0.1)))
                robot.send_action({f"{PAN}.pos": pan_goal})
            if robot and seek_on and motion_blocker == f"searching for {seek_name}":
                step_dt = min(dt, 0.1)
                if sweep_pan(args.search_speed, step_dt):
                    search_level = (search_level + 1) % len(args.search_heights)  # next pass at the next height
                action = {f"{PAN}.pos": pan_goal}
                if hik:
                    # Search at face height: ease the phone to this pass's height (same IK checks as
                    # tracking) and level the camera, so faces at that height are in frame.
                    z_target = float(np.clip(hik.z0 + args.search_heights[search_level],
                                             hik.z0 - args.height_down, hik.z0 + args.height_up))
                    dz = float(np.clip(z_target - z_goal, -args.search_lift_rate * step_dt, args.search_lift_rate * step_dt))
                    for scale in (1.0, 0.5, 0.25):
                        if abs(dz) < 1e-4:
                            break
                        q, pos_err, ang_err = hik.solve(z_goal + dz * scale, control.reach)
                        if valid_ik(hik, q, pos_err, ang_err):
                            hik.q, z_goal = q, z_goal + dz * scale
                            break
                    pitch = hik.camera_pitch(tilt_offset, args.camera_pitch_offset)
                    slope = hik.camera_pitch(tilt_offset + 1.0, args.camera_pitch_offset) - pitch  # deg pitch per deg tilt
                    if abs(slope) > 0.1:
                        want = tilt_offset + (args.search_pitch - pitch) / slope
                        tilt_offset = float(np.clip(tilt_offset + np.clip(want - tilt_offset, -20 * step_dt, 20 * step_dt),
                                                    -args.tilt_limit, args.tilt_limit))
                    action.update(plane_action(hik.q, tilt_offset))
                elif args.tilt:
                    tilt_goal = float(tilt_goal + np.clip(tilt_home - tilt_goal, -20 * step_dt, 20 * step_dt))
                    action[f"{TILT}.pos"] = tilt_goal
                robot.send_action(action)
            if robot and motion_blocker is None:
                action = {}
                if abs(ex) > args.deadband:
                    step = args.pan_sign * args.gain * ex * args.hfov / 2
                    step = float(np.clip(step, -args.max_step, args.max_step))
                    pan_goal = clamp_pan(pan_goal + step)
                    action[f"{PAN}.pos"] = pan_goal
                vfov = args.hfov * h / w
                tilt_step = 0.0
                if (args.tilt or args.height) and abs(ey) > args.deadband:
                    tilt_step = float(np.clip(args.tilt_sign * args.gain * ey * vfov / 2, -args.max_step, args.max_step))
                if hik:
                    # Wrist does the fast aiming. Whenever it's tilted, move the whole phone in that direction
                    # (keeping its angle) so the wrist can relax back toward level: phone ends up at eye level.
                    tilt_offset = float(np.clip(tilt_offset + tilt_step, -args.tilt_limit, args.tilt_limit))
                    # Absolute camera pitch includes the starting arm pose. A centered face
                    # while looking upward still means the phone needs to rise.
                    camera_pitch = hik.camera_pitch(tilt_offset, args.camera_pitch_offset) if args.height else 0.0
                    z_new = z_goal
                    if args.height and abs(camera_pitch) > 2.0:
                        dz = float(np.clip(args.height_rate * camera_pitch, -0.004, 0.004))
                        z_new = float(np.clip(z_goal + dz, hik.z0 - args.height_down, hik.z0 + args.height_up))
                    reach_new = control.propose_reach(dt)
                    if z_new != z_goal or reach_new != control.reach:
                        q, pos_err, ang_err = hik.solve(z_new, reach_new)
                        if not valid_ik(hik, q, pos_err, ang_err) and z_new != z_goal:
                            # Depth must not prevent a feasible height correction. First hold
                            # reach; if needed, retract slightly within the configured bound.
                            reach_new = control.reach
                            q, pos_err, ang_err = hik.solve(z_new, reach_new)
                            if not valid_ik(hik, q, pos_err, ang_err):
                                retreat = min(0.004, args.depth_rate * max(0.0, min(dt, 0.1)))
                                reach_new = max(control.reach0 - args.depth_back, control.reach - retreat)
                                q, pos_err, ang_err = hik.solve(z_new, reach_new)
                        if not valid_ik(hik, q, pos_err, ang_err) and z_new != z_goal:
                            # Near a folded pose, even 4 mm can exceed the joint-step
                            # limit. Try smaller lifts without changing that limit.
                            dz = z_new - z_goal
                            reach_new = control.reach
                            for scale in (0.5, 0.25, 0.125):
                                z_new = z_goal + dz * scale
                                q, pos_err, ang_err = hik.solve(z_new, reach_new)
                                if valid_ik(hik, q, pos_err, ang_err):
                                    break
                        if valid_ik(hik, q, pos_err, ang_err):
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
                    if not isinstance(sent, dict):  # LeRobot returns what it sent; anything else: assume as asked
                        sent = action
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

            control.publish(mode="deactivated" if control.deactivated else
                            "frozen" if control.frozen else ("tracking" if held else "released"),
                held=held, target_visible=target is not None, motion_blocker=motion_blocker,
                voice_following=voice_name, voice_face_missing=voice_name is not None and not voice_face_known,
                seeking=seek_name if seek_on else None,
                orientation="landscape" if rotated else "portrait",
                height_enabled=args.height, height=z_goal if hik else None,
                camera_pitch=hik.camera_pitch(tilt_offset, args.camera_pitch_offset) if hik and args.height else None,
                at_height_limit=bool(hik and args.height and (
                    z_goal >= hik.z0 + args.height_up - 1e-6 or z_goal <= hik.z0 - args.height_down + 1e-6)))
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
                mode = "DEACTIVATED" if control.deactivated else "FROZEN" if control.frozen else ("DRY RUN" if not robot else ("TRACKING" if held else "RELEASED"))
                seen = target[4] if target is not None else "searching"
                status = f"{mode} | {seen} | ex={ex:+.2f} | cam {cam_fps:.0f} / proc {fps:.0f} fps"
                if control.depth:
                    status += f" | range {control.depth.distance:.2f}m" if control.depth.distance is not None else " | depth HOLD"
                if gestures:
                    pending = feedback.get("pending")
                    status += f" | {pending[0]} {pending[1]:.0%}" if pending else (
                        " | hand detected" if feedback.get("hands") else " | no hands")
                if enrolling:
                    enrollment = control.enrollment.snapshot()
                    status = f'ENROLLING {enrollment["name"]}: {enrollment["samples"]}/{enrollment["total"]}'
                preview.show(annotate(frame, boxes, target, smoothed, args.deadband, status, face_labels))
    except OSError as e:  # serial errors: the port vanished or another program is talking to the arm
        others = other_users_of(robot.config.port) if robot else []
        log.error(f"Lost contact with the arm ({e})" + (
            "\nAnother program is using it:\n  " + "\n  ".join(others) if others else ""))
    finally:
        control.publish(mode="stopped")
        if voice_listener:
            voice_listener.terminate()
            try:
                voice_listener.wait(timeout=3)
            except subprocess.TimeoutExpired:
                voice_listener.kill()
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
    p.add_argument("--home-reach", type=float, default=0.28, help="Reset pose: gripper tip metres in front of the base")
    p.add_argument("--home-height", type=float, default=0.26, help="Reset pose: gripper tip metres above the base")
    p.add_argument("--home-pitch", type=float, default=0.0,
                   help="Reset pose: degrees the gripper points above (+) or below (-) level")
    p.add_argument("--cover-seconds", type=float, default=2.0,
                   help="cover the camera with your hand this long to release the phone")
    p.add_argument("--cover-detail", type=float, default=15.0,
                   help="picture counts as covered below this detail level")
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
    p.add_argument("--follow-face", help="follow only this enrolled name using YuNet + SFace")
    p.add_argument("--face-db", default=os.path.join(MODELS_DIR, "faces.npz"), help="enrolled face database")
    p.add_argument("--face-threshold", type=float, default=0.45, help="minimum face cosine similarity")
    p.add_argument("--voice", metavar="NAME", nargs="+",
                   help="listen all the time; while one of these people is heard, follow their enrolled face and "
                        "sweep the base to find them if they're out of view (e.g. --voice Arnav Jaisree)")
    p.add_argument("--voice-python", default="/home/aup/miniforge3/envs/speakerid/bin/python",
                   help="Python of the speaker-ID environment (acoustic/README.md)")
    p.add_argument("--voice-local-mic", type=int, metavar="DEVICE",
                   help="use this microphone on the tracker machine instead of the forwarded laptop mic")
    p.add_argument("--follow-mode", choices=["manual", "speaker"], default="manual",
                   help="start in this following mode: manual = Go to buttons only; speaker = also go to whoever speaks")
    p.add_argument("--voice-min-confidence", type=float, default=0.45,
                   help="lowest speaker-ID score that makes the arm go to someone (lower = more eager)")
    p.add_argument("--voice-hold", type=float, default=4.0,
                   help="keep looking for a person this many seconds after their voice was last identified")
    p.add_argument("--search-speed", type=float, default=15.0, help="degrees per second while sweeping to find someone")
    p.add_argument("--search-heights", type=float, nargs="+", default=[0.0, 0.08], metavar="M",
                   help="phone heights (metres above the start pose) to search at, one per sweep, in turn")
    p.add_argument("--search-pitch", type=float, default=0.0,
                   help="camera pitch while searching, degrees (0 = level, positive = up)")
    p.add_argument("--search-lift-rate", type=float, default=0.04, help="metres per second to reach a search height")
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
    p.add_argument("--gestures", action="store_true",
                   help="enable closer/farther/freeze/resume and the L sign (rotate) via the vision worker")
    p.add_argument("--gesture-python", default=str(ROOT / ".venv-vision/bin/python"), help="isolated vision Python interpreter")
    p.add_argument("--gesture-step", type=float, default=0.02, help="metres per closer/farther command")
    p.add_argument("--swap-hands", action="store_true",
                   help="if the L sign's left/right come out reversed (left-hand L should rotate to landscape)")
    p.add_argument("--distance", type=float, help="follow at this approximate calibrated distance in metres")
    p.add_argument("--distance-reference", type=float, help="known camera-to-face calibration distance in metres")
    p.add_argument("--distance-face-width", type=float, help="face-box width in pixels at the reference distance")
    p.add_argument("--distance-tolerance", type=float, default=0.10, help="distance deadband in metres")
    p.add_argument("--depth-forward", type=float, default=0.05, help="max metres to extend from starting reach")
    p.add_argument("--depth-back", type=float, default=0.05, help="max metres to retract from starting reach")
    p.add_argument("--depth-rate", type=float, default=0.02, help="max reach speed in metres per second")
    p.add_argument("--urdf", default=DEFAULT_URDF, help="SO-101 URDF used for IK")
    p.add_argument("--camera-pitch-offset", type=float, default=0.0,
                   help="camera mounting pitch relative to gripper forward axis, degrees (positive = up)")
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
    p.add_argument("--release-amount", type=float, default=30.0, help="no longer used: release opens the gripper fully")
    p.add_argument("--grip-move-torque", type=float, default=50.0,
                   help="gripper force while opening, %% of max (50 = LeRobot's cap); holding uses --grip-torque")
    p.add_argument("--grip-close-torque", type=float, default=40.0,
                   help="gripper force for the closing stroke, %% of max; it drops to --grip-torque near the end "
                        "and as soon as the jaws touch the phone")
    p.add_argument("--test-motion-scale", type=float, default=1.0,
                   help="test_motion command: size of the moves relative to TEST_MOTION_DEG (0-1.5)")
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
    p.add_argument("--pan-limit", type=float, default=90.0,
                   help="max degrees either side of North the base may face (capped at 90 = due West/East; never south)")
    p.add_argument("--north-pan", type=float, default=0.0,
                   help="shoulder_pan angle that faces forward (North); 0 = centre of the calibrated travel")
    p.add_argument("--ambient-speed", type=float, default=10.0, help="degrees per second of the Ambient West-East sweep")
    p.add_argument("--dpad-pan-step", type=float, default=8.0, help="D-pad left/right: degrees per press")
    p.add_argument("--dpad-height-step", type=float, default=0.015, help="D-pad up/down: metres per press (height IK)")
    p.add_argument("--dpad-tilt-step", type=float, default=5.0, help="D-pad up/down without height IK: wrist degrees")
    p.add_argument("--tilt-limit", type=float, default=25.0, help="max degrees up/down of the start pose")
    p.add_argument("--smoothing", type=float, default=0.5, help="0 = raw detections, closer to 1 = smoother")
    p.add_argument("--lost-timeout", type=float, default=2.0, help="seconds before forgetting a lost person")
    p.add_argument("--preview-port", type=int, default=8000, help="0 disables the browser preview")
    p.add_argument("--max-seconds", type=float, help="stop automatically after this long")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if not -1 <= args.face_threshold <= 1:
        p.error("--face-threshold must be between -1 and 1")

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
    if not np.isfinite(args.camera_pitch_offset):
        p.error("--camera-pitch-offset must be finite")
    if not 0 <= args.preview_port <= 65535:
        p.error("--preview-port must be between 0 and 65535")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    # LeRobot warns each time a move is capped at 10 deg per command; Party mode does that on purpose
    # dozens of times a second (it's how it runs the servos at full speed), so keep the log readable.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
    # LeRobot can re-apply its own logging setup when it loads, so also drop the message at the output.
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: "had to be clamped to be safe" not in record.getMessage())
    run(args)


if __name__ == "__main__":
    main()
