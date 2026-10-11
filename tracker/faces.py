#!/usr/bin/env python
"""Enroll named faces or recognize them using the existing arm camera (no motors)."""

import argparse
import logging
from pathlib import Path
import time
import threading
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from .track import Camera, FACE_MODEL, MODELS_DIR, Preview
except ImportError:
    from track import Camera, FACE_MODEL, MODELS_DIR, Preview

SFACE_MODEL = Path(MODELS_DIR) / "face_recognition_sface_2021dec.onnx"
DEFAULT_DB = Path(MODELS_DIR) / "faces.npz"


class FaceRecognizer:
    def __init__(self, database=DEFAULT_DB, threshold=0.45, target_name=None):
        for model in (Path(FACE_MODEL), SFACE_MODEL):
            if not model.is_file():
                raise FileNotFoundError(f"Missing {model}; see tracker/README.md for downloads")
        self.detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.85)
        self.encoder = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        self.database = Path(database)
        self.threshold = threshold
        self.target_name = target_name
        self.names = np.empty(0, dtype=str)
        self.features = np.empty((0, 128), dtype=np.float32)
        if self.database.exists():
            with np.load(self.database, allow_pickle=False) as data:
                self.names = data["names"].copy()
                self.features = data["features"].copy()
            if (self.names.ndim != 1 or self.names.dtype.kind != "U"
                    or self.features.shape != (len(self.names), 128)
                    or not np.isfinite(self.features).all()
                    or np.any(np.linalg.norm(self.features, axis=1) == 0)):
                raise ValueError(f"Invalid face database: {self.database}")
            self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        if target_name is not None and target_name not in self.names:
            raise ValueError(f"Face {target_name!r} is not enrolled in {self.database}")

    def faces(self, frame):
        """Keep YuNet landmarks: SFace needs them to align each face before encoding."""
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else faces

    def feature(self, frame, face):
        aligned = self.encoder.alignCrop(frame, face)
        feature = self.encoder.feature(aligned).ravel()
        return feature / np.linalg.norm(feature)

    def identify(self, feature):
        if not len(self.names):
            return "unknown", 0.0
        scores = self.features @ feature
        best = int(np.argmax(scores))
        score = float(scores[best])
        return (str(self.names[best]) if score >= self.threshold else "unknown"), score

    def recognize(self, frame):
        """Return face landmarks, identity, and similarity for every visible face."""
        return [(face, *self.identify(self.feature(frame, face))) for face in self.faces(frame)]

    def detect(self, frame):
        """Same box interface as track.py; optionally return only the named person."""
        boxes = []
        for face, name, score in self.recognize(frame):
            if self.target_name is None or name == self.target_name:
                boxes.append((*map(int, face[:4]), f"face:{name} {score:.2f}"))
        return boxes

    def enroll(self, name, features):
        """Replace this person's templates; retain everyone else's enrollment."""
        keep = self.names != name
        names = np.concatenate((self.names[keep], np.full(len(features), name)))
        vectors = np.vstack((self.features[keep], features))
        self.database.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.database.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, names=names, features=vectors)
        temporary.replace(self.database)
        self.names, self.features = names, vectors


class FaceEnrollment:
    """HTTP requests change state; only the camera loop runs models and saves data."""

    ACTIVE = ("starting", "running", "cancelling")

    def __init__(self, database=DEFAULT_DB, threshold=0.45, recognizer=None,
                 samples=20, seconds=60, clock=time.monotonic):
        self.database, self.threshold = database, threshold
        self.recognizer = recognizer
        self.required, self.seconds, self.clock = samples, seconds, clock
        self.lock = threading.RLock()
        self.features = []
        self.state = {"state": "idle", "name": "", "samples": 0, "total": samples,
                      "message": "Enter a name to enroll a face."}

    @property
    def busy(self):
        with self.lock:
            return self.state["state"] in self.ACTIVE

    def snapshot(self):
        with self.lock:
            return self.state.copy()

    def start(self, name):
        name = name.strip()
        if not name or len(name) > 80 or name.lower() == "unknown" or not name.isprintable():
            raise ValueError("Use a name of 1-80 characters; 'unknown' is reserved.")
        with self.lock:
            if self.busy:
                raise RuntimeError("Enrollment is already running; cancel it first.")
            self.features = []
            self.deadline = self.clock() + self.seconds
            self.last_sample = float("-inf")
            self.state.update(state="starting", name=name, samples=0,
                              message=f"Starting enrollment for {name}...")
            return self.snapshot()

    def cancel(self):
        with self.lock:
            if self.busy:
                self.state.update(state="cancelling", message="Cancelling enrollment...")
            return self.snapshot()

    def labels(self, frame, boxes):
        """Attach known identities to existing boxes without changing tracking targets."""
        if not boxes:
            return {}
        if self.recognizer is None:
            if not Path(self.database).is_file() or getattr(self, "_label_load_failed", False):
                return {}
            try:
                self.recognizer = FaceRecognizer(self.database, self.threshold)
            except (OSError, ValueError, cv2.error) as exc:
                logging.warning("Face overlay unavailable: %s", exc)
                self._label_load_failed = True
                return {}
        if not len(self.recognizer.names):
            return {}
        matches = {}
        for face, name, score in self.recognizer.recognize(frame):
            x, y, w, h = face[:4]
            cx, cy = x + w / 2, y + h / 2
            containing = [i for i, (bx, by, bw, bh, _) in enumerate(boxes)
                          if bx <= cx <= bx + bw and by <= cy <= by + bh]
            if containing:
                # Prefer the face box over a larger overlapping body box.
                index = min(containing, key=lambda i: boxes[i][2] * boxes[i][3])
                matches.setdefault(index, []).append(name)
        return {i: names[0] for i, names in matches.items()
                if len(names) == 1 and names[0] != "unknown"}

    def update(self, frame):
        # Keep the lock through inference/save so cancel and restart cannot race a write.
        # Sampling runs at most twice a second; normal tracking never takes this lock for inference.
        with self.lock:
            if not self.busy:
                return
            if self.state["state"] == "cancelling":
                self.features = []
                self.state.update(state="cancelled", message="Enrollment cancelled; saved faces unchanged.")
                return
            try:
                now = self.clock()
                if now >= self.deadline:
                    raise TimeoutError("Enrollment timed out; saved faces unchanged. Try again with one face visible.")
                if self.recognizer is None:
                    self.recognizer = FaceRecognizer(self.database, self.threshold)
                if self.state["state"] == "starting":
                    self.state.update(state="running", message="Enrollment started. Keep one face in view and slowly turn your head.")
                if frame is None:
                    self.state["message"] = "Enrollment started; waiting for camera frames."
                    return
                if now - self.last_sample < 0.5:
                    return
                faces = self.recognizer.faces(frame)
                if len(faces) != 1:
                    self.state["message"] = ("No face detected; face the camera." if len(faces) == 0
                                             else "Multiple faces detected; keep only the enrolling person in view.")
                    return
                self.features.append(self.recognizer.feature(frame, faces[0]))
                self.last_sample = now
                self.state.update(samples=len(self.features), message="Enrollment started; collecting samples. Slowly vary your head angle.")
                if len(self.features) >= self.required:
                    name = self.state["name"]
                    self.recognizer.enroll(name, self.features)
                    self.features = []
                    self.state.update(state="completed", message=f"Done! Saved {name} ({self.required} samples).")
            except Exception as exc:
                self.features = []
                self.state.update(state="error", message=f"Enrollment failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["enroll", "recognize"])
    parser.add_argument("--name", help="person's label when enrolling")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=60, help="enrollment timeout")
    parser.add_argument("--preview-port", type=int, default=8000, help="0 disables preview")
    args = parser.parse_args()
    if args.mode == "enroll" and (not args.name or args.name.strip().lower() == "unknown"):
        parser.error("enroll requires --name (the label 'unknown' is reserved)")
    if args.samples <= 0 or args.seconds <= 0 or not -1 <= args.threshold <= 1:
        parser.error("samples/seconds must be positive; threshold must be between -1 and 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    recognizer = FaceRecognizer(args.database, args.threshold)
    if args.mode == "recognize" and not len(recognizer.names):
        parser.error("no enrolled faces; run enroll first")
    enrollment = FaceEnrollment(args.database, args.threshold, recognizer,
                                samples=args.samples, seconds=args.seconds)
    if args.mode == "enroll":
        enrollment.start(args.name)
    def reject_motion(command):
        raise ValueError("Motor commands are unavailable in face-only mode")
    control = SimpleNamespace(enrollment=enrollment, submit=reject_motion,
        snapshot=lambda: {"mode": args.mode, "enrollment": enrollment.snapshot()})
    preview = Preview(args.preview_port, control) if args.preview_port else None
    camera = Camera(args.camera, exposure=args.exposure, flip=args.flip)
    seq = 0
    logging.info("Keep only the enrolling person in view; vary head angle and distance slightly."
                 if args.mode == "enroll" else "Recognizing enrolled faces; Ctrl+C to stop.")
    try:
        while True:
            seq, frame = camera.read(seq)
            enrolling = enrollment.busy
            enrollment.update(frame)
            state = enrollment.snapshot()
            if args.mode == "enroll" and state["state"] == "completed":
                logging.info(state["message"])
                return
            if args.mode == "enroll" and state["state"] in ("error", "cancelled"):
                raise SystemExit(state["message"])
            if frame is None:
                continue
            if enrolling:
                status = f'{state["samples"]}/{state["total"]} | {state["message"]}'
            else:
                boxes = recognizer.detect(frame)
                for x, y, w, h, label in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                status = " | ".join(box[4] for box in boxes) or "No faces"
            if preview:
                cv2.putText(frame, status, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                preview.show(frame)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if preview:
            preview.close()


if __name__ == "__main__":
    main()
