"""MediaPipe hand-gesture wrapper; runs in the vision environment and never imports robot modules."""
from followme.environment import verified_model
from followme.gestures.interpreter import HandObservation
from followme.gestures.motion import hand_pose, hand_shape

WRIST, MIDDLE_KNUCKLE = 0, 9


class MediaPipeGestureRecognizer:
    def __init__(self, model, manifest, max_hands=2):
        verified_model(model, manifest)
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=max_hands)
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame_bgr, timestamp_ms):
        """Returns one HandObservation per detected hand. Timestamps must increase."""
        import numpy as np
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(image, int(timestamp_ms))
        hands, aspect = [], frame_bgr.shape[1] / frame_bgr.shape[0]
        for gestures, landmarks in zip(result.gestures, result.hand_landmarks):
            if not gestures:
                continue
            wrist, knuckle = landmarks[WRIST], landmarks[MIDDLE_KNUCKLE]
            points = [(point.x, point.y) for point in landmarks]
            openness, fingers = hand_shape(points, aspect)
            spread, angle = hand_pose(points, aspect)
            hands.append(HandObservation(gestures[0].category_name, float(gestures[0].score),
                                         (wrist.x + knuckle.x) / 2, (wrist.y + knuckle.y) / 2, openness, fingers,
                                         spread, angle))
        return hands

    def close(self):
        self._recognizer.close()


def main():
    """Run the existing recognizer/pipeline over the shared vision IPC transport."""
    import argparse
    import math
    import socket
    import sys
    from importlib.metadata import version
    import cv2
    import numpy as np
    from followme.gestures import GesturePipeline, GestureZone, GestureInterpreter
    from followme.protocol import JsonSocket, VERSION
    from followme.vision_worker import validate_frame

    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    channel = JsonSocket(socket.socket(fileno=args.fd))
    recognizer = None
    try:
        manifest = verified_model(args.model, args.manifest)
        recognizer = MediaPipeGestureRecognizer(args.model, args.manifest)
        # Hanging fingers-down gestures may be lower than the default held-gesture zone.
        pipeline = GesturePipeline(interpreter=GestureInterpreter(zone=GestureZone(below_face=.6, max_y_without_face=.9)))
        last_ms = -1
        channel.send({'type': 'ready', 'protocol_version': VERSION, 'interpreter': sys.executable,
                      'mediapipe_version': version('mediapipe'), 'model_sha256': manifest['sha256']})
        while True:
            try:
                message = channel.receive(timeout=3600)
            except EOFError:
                break
            if message.get('type') == 'shutdown':
                break
            identity = {key: message.get(key) for key in ('request_id', 'camera_role', 'sequence', 'capture_monotonic_ns')}
            try:
                raw = validate_frame(message)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[0] > 1080 or frame.shape[1] > 1920:
                    raise ValueError('Invalid frame or dimensions')
                ms = message['capture_monotonic_ns'] // 1_000_000
                if ms <= last_ms:
                    raise ValueError('Gesture timestamps must increase')
                face = message.get('face_center')
                if face is not None and (not isinstance(face, list) or len(face) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in face)):
                    raise ValueError('Invalid normalized face center')
                hands = recognizer.recognize(frame, ms)
                last_ms = ms
                command = pipeline.update(hands, message['capture_monotonic_ns'] / 1e9, face)
                channel.send({'type': 'result', **identity, 'command': command, 'error': None})
            except Exception as exc:
                channel.send({'type': 'result', **identity, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if recognizer:
            recognizer.close()
        channel.close()


if __name__ == '__main__':
    main()
