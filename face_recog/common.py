"""Shared pieces for the face recognition pipeline: paths, face cropping, model."""
from pathlib import Path

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RAW_DIR = DATA / "raw"        # raw/<person>/*.jpg  (the unzipped photos)
FACES_DIR = DATA / "faces"    # faces/<label>/*.jpg (cropped faces, incl. "unknown")
MODEL_PATH = REPO / "models" / "face_recog.pt"

UNKNOWN = "unknown"
CROP_SIZE = 200    # stored crops keep some margin so augmentation has room to move
INPUT_SIZE = 160   # network input; centre crop of a stored crop == tight face box
MARGIN = CROP_SIZE / INPUT_SIZE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_detector(device):
    return MTCNN(keep_all=True, device=device, min_face_size=40)


def load_image(path):
    # exif_transpose matters: phone photos are stored rotated/mirrored
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def detect_faces(img, detector, max_side=1600, min_prob=0.95):
    """Return [(box, crop)] sorted largest face first. box is (x0, y0, x1, y1) in
    original image pixels, crop is a CROP_SIZE square PIL image."""
    scale = min(1.0, max_side / max(img.size))
    small = img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img
    boxes, probs = detector.detect(small)
    if boxes is None:
        return []
    faces = []
    for box, prob in zip(boxes, probs):
        if prob < min_prob:
            continue
        x0, y0, x1, y1 = (float(v) / scale for v in box)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * MARGIN / 2
        crop = img.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
        crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
        faces.append(((x0, y0, x1, y1), crop))
    faces.sort(key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)
    return faces


class FaceClassifier(nn.Module):
    """FaceNet (InceptionResnetV1, pretrained on VGGFace2) + a small classification head."""

    def __init__(self, num_classes, pretrained="vggface2"):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained)  # -> L2-normalised 512-d
        self.backbone.logits = None  # VGGFace2's 8631-way classifier, unused here
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x) * 10.0)


def load_model(path=MODEL_PATH, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = FaceClassifier(len(ckpt["classes"]), pretrained=None)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], ckpt["threshold"]
