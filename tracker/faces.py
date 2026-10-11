#!/usr/bin/env python
"""Enroll named faces or recognize them using the existing arm camera (no motors)."""

import argparse
import logging
from pathlib import Path
import time

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

    def detect(self, frame):
        """Same box interface as track.py; optionally return only the named person."""
        boxes = []
        for face in self.faces(frame):
            name, score = self.identify(self.feature(frame, face))
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
    preview = Preview(args.preview_port) if args.preview_port else None
    camera = Camera(args.camera, exposure=args.exposure, flip=args.flip)
    samples, seq, last_sample = [], 0, 0.0
    deadline = time.monotonic() + args.seconds
    logging.info("Keep only the enrolling person in view; vary head angle and distance slightly."
                 if args.mode == "enroll" else "Recognizing enrolled faces; Ctrl+C to stop.")
    try:
        while args.mode == "recognize" or time.monotonic() < deadline:
            seq, frame = camera.read(seq)
            if frame is None:
                continue
            if args.mode == "enroll":
                faces = recognizer.faces(frame)
                now = time.monotonic()
                # Multiple people would contaminate the label. Space samples out to vary poses.
                if len(faces) == 1 and now - last_sample >= 0.5:
                    samples.append(recognizer.feature(frame, faces[0]))
                    last_sample = now
                    logging.info("%s: %d/%d samples", args.name, len(samples), args.samples)
                status = f"{args.name}: {len(samples)}/{args.samples} | keep exactly one face in view"
                if len(samples) == args.samples:
                    recognizer.enroll(args.name, samples)
                    logging.info("Saved %s to %s", args.name, args.database)
                    return
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
        raise SystemExit(f"Enrollment timed out with {len(samples)}/{args.samples} samples; database unchanged")
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if preview:
            preview.server.shutdown()
            preview.server.server_close()


if __name__ == "__main__":
    main()
