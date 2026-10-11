# Named face recognition

`track.py` locates people; `faces.py` adds identity recognition using the same
robot-mounted camera. It uses OpenCV YuNet for face boxes and landmarks, and
SFace for aligned 128-dimensional face features. Enrollment stores labeled
feature vectors in `models/faces.npz`; it does **not** retrain SFace's neural
network weights. Adding people does not require a GPU or another Python package.

Use the existing environment with OpenCV and NumPy. OpenCV 4.10 or later is
recommended for these ONNX models. Download the SFace model once from OpenCV Zoo:

```bash
curl -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  -o tracker/models/face_recognition_sface_2021dec.onnx
```

YuNet (`tracker/models/face_detection_yunet_2023mar.onnx`) is already present.

Stop any running tracker before enrollment or recognition: each command opens
the camera, so run only one at a time. Use the same `--camera`, `--flip`, and
`--exposure` options as your normal tracker. From the repository root:

```bash
# Each command captures 20 samples, spaced at least half a second apart.
python tracker/faces.py enroll --name Alice --camera /dev/video0
python tracker/faces.py enroll --name Bob --camera /dev/video0

# Display names and cosine similarity; this command never connects to the arm.
python tracker/faces.py recognize --camera /dev/video0

# Check identity selection without moving motors, then run normal tracking.
python tracker/track.py --follow-face Alice --dry-run --camera /dev/video0
python tracker/track.py --follow-face Alice --camera /dev/video0
```

View the feed at `http://<arm-ip>:8000/`. During enrollment, keep only the named
person in frame and vary their angle, distance, and lighting. Frames containing
zero or multiple faces are skipped. Re-enrolling a name replaces its old samples;
interrupted or timed-out enrollment leaves the database unchanged. Only face
features and labels are stored, not photos. Keep enrollment data local; it is
excluded from Git.

Recognition chooses the nearest stored sample and rejects scores below
`--threshold` (standalone) or `--face-threshold` (tracking), default `0.45`.
This is a starting point, not a measured accuracy guarantee. Validate with both
enrolled and unfamiliar people on your camera; raise the threshold to reduce
false matches, or improve enrollment if genuine matches are rejected. Scores
are similarities, not probabilities.

`--follow-face` overrides the regular person detector. Only a recognized face
with that name can drive motion; the arm holds its position when no match is
visible, including when the person turns away. There is no body fallback in this
mode, because a body box cannot establish identity. Without `--follow-face`,
the existing tracking behavior remains the same.

Reference: [OpenCV's YuNet/SFace tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).

# Bounded depth following

Depth following is optional and uses the same height IK to extend/retract the
phone while pan and optional tilt/height tracking continue. Robot `x` is forward
reach; robot `z` remains vertical height.

First run `python tracker/track.py --dry-run` with your usual camera options.
Face labels in the preview show the detected width in pixels. Face the camera at
a measured camera-to-face distance and record a representative width. Use the
same user, detector, resolution, and camera settings for calibration and tracking.

For example, **only if the face box measures 100 pixels at 1 metre**, preview
estimated range without moving motors:

```bash
python tracker/track.py --dry-run --distance 1.0 \
  --distance-reference 1.0 --distance-face-width 100 \
  --distance-tolerance 0.10 --depth-forward 0.05 --depth-back 0.05 \
  --depth-rate 0.02 --tilt
```

Remove `--dry-run` to enable movement. The example requests an approximate
1-metre distance with a 10-cm deadband, at most 5 cm extension/retraction from
the starting reach, and a maximum reach speed of 2 cm/s. Add `--height` to follow
vertical position as well.
The IK model must match the arm, as for the existing height mode.

Distance uses `reference_distance * reference_face_width / current_face_width`.
Face angle and detector box variation affect accuracy; this is not a precise
range sensor. Missing, profile, body, and horizontally cropped detections hold
reach and clear distance smoothing. At a travel bound or rejected IK solution,
reach holds while aiming continues. A stationary arm cannot maintain the
requested distance beyond its reachable workspace.

Simulation checks (no camera or motors):

```bash
python -m unittest discover -s test -p 'test_tracker_depth.py' -v
```

# Shared commands and API

The existing preview port now hosts one FastAPI server: camera feed at `/stream`,
a small button interface at `/`, and interactive API documentation at `/docs`.
No separate API process is needed. FastAPI and Uvicorn are already included in
`requirements/runtime.in`; run in the project's runtime/LeRobot environment.

```bash
# Preview and manual commands; motors remain untouched.
python tracker/track.py --dry-run --tilt

# Add the four hand gestures using the existing isolated vision environment.
python tracker/track.py --dry-run --tilt --gestures
```

Remove `--dry-run` to enable physical tracking. `--gesture-python` defaults to
this checkout's `.venv-vision/bin/python`. The worker reuses the gesture model
and manifest in `models/gestures`; the local `followme` package must be installed
in that environment, as in the existing vision setup. MediaPipe is not imported
into the motor process. The worker uses a larger vertical gesture zone to allow
the fingers-down farther gesture below the face.

All gesture, HTTP, and terminal commands enter the same bounded queue and are
applied by the tracking loop. Type a command name and ENTER in an interactive
terminal (`r` and `g` remain release/grip shortcuts). HTTP usage:

```bash
curl -X POST http://localhost:8000/commands \
  -H 'Content-Type: application/json' -d '{"command":"freeze"}'
curl http://localhost:8000/status
```

- `freeze`: hold tracking until `resume`; cancel unfinished manual reach moves.
- `resume`: resume tracking; the phone must still be held and a target visible.
- `closer` / `farther`: adjust requested reach by `--gesture-step` (default 2 cm).
  Travel stays within `--depth-forward` / `--depth-back` (default 5 cm each),
  with speed limited by `--depth-rate` (default 2 cm/s) and the existing IK checks.
  With `--distance`, commands instead decrease/increase the following-distance
  target by the same step, so automatic following does not undo the command.
- `release` / `grip`: retain the existing gripper controls. Gripping does not
  override an explicit freeze; issue `resume` separately.

Closer/farther are ignored while frozen or the phone is released. The API's
`202 Accepted` means queued, not that a motor moved. `/status` reports the last
command's result, frozen/held state, target visibility, target/estimated
following distance, requested/current reach, travel bounds, and movement
blockers. Invalid commands return 422; a full queue returns 503. The old
`POST /release` and `POST /grip` endpoints remain available.

Reach is measured in the arm's forward plane from its base (robot `x`), not
camera-to-user range. Without `--distance`, a requested reach remains fixed as
the tracker aims at the user. Freeze holds pan/tilt/height/reach; explicit
gripper commands remain available. A gesture-worker error stops tracking.
`--preview-port 0` disables HTTP/preview while terminal and enabled gestures
remain available. The API runs in one worker with a shared in-memory queue.

Checks without moving hardware:

```bash
.venv/bin/python -m unittest discover -s test -p 'test_tracker_depth.py' -v
.venv-vision/bin/python -m unittest discover -s test -p 'test_gestures.py' -v
```
