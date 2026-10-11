"""Recognise faces with the trained model.

    python face_recog/predict.py photo1.jpg photo2.jpg   # print who is in each photo
    python face_recog/predict.py --webcam                # live view, press q to quit
"""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from common import MODEL_PATH, UNKNOWN, detect_faces, get_device, load_image, load_model, make_detector
from train import EVAL_TF


@torch.no_grad()
def recognise(img, detector, model, classes, threshold, device):
    """Returns [(box, name, confidence)] for every face in a PIL image."""
    faces = detect_faces(img, detector)
    if not faces:
        return []
    batch = torch.stack([EVAL_TF(crop) for _, crop in faces]).to(device)
    probs = torch.softmax(model(batch), 1).cpu().numpy()
    results = []
    for (box, _), p in zip(faces, probs):
        name = classes[p.argmax()] if p.max() >= threshold else UNKNOWN
        results.append((box, name, float(p.max())))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--threshold", type=float, help="override the threshold stored with the model")
    args = ap.parse_args()

    device = get_device()
    model, classes, threshold = load_model(args.model, device)
    threshold = args.threshold or threshold
    detector = make_detector(device)

    for path in args.images:
        results = recognise(load_image(path), detector, model, classes, threshold, device)
        print(path, "->", ", ".join(f"{n} ({c:.2f})" for _, n, c in results) or "no face")

    if args.webcam:
        cap = cv2.VideoCapture(args.camera)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for box, name, conf in recognise(img, detector, model, classes, threshold, device):
                x0, y0, x1, y1 = np.round(box).astype(int)
                color = (0, 0, 255) if name == UNKNOWN else (0, 200, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("face recognition (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
