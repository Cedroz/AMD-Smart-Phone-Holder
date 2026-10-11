"""Fine-tune a pretrained FaceNet to recognise the people in data/faces (+ "unknown").

    python face_recog/train.py              # train on the train split, report on the held-out test split
    python face_recog/train.py --all-data   # final model: train on every photo, no evaluation

The split is done on the ORIGINAL photos before any augmentation, and unknown
identities never appear in both splits, so the test numbers are not leaked.
"""
import argparse
import io
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from common import FACES_DIR, IMG_EXTS, INPUT_SIZE, MODEL_PATH, UNKNOWN, FaceClassifier, get_device

NORMALIZE = T.Normalize([0.5] * 3, [128 / 255] * 3)  # facenet's (x - 127.5) / 128


class Degrade:
    """Randomly lower resolution / add JPEG artefacts, so sharp phone photos also
    cover webcam-quality input (and can't be told apart from LFW by sharpness alone)."""

    def __call__(self, img):
        if random.random() < 0.5:
            s = random.randint(48, 140)
            img = img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)
        if random.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(25, 75))
            img = Image.open(buf).convert("RGB")
        return img


TRAIN_TF = T.Compose([
    T.RandomAffine(degrees=15, translate=(0.04, 0.04), scale=(0.9, 1.15), shear=5),
    T.CenterCrop(INPUT_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.3, 0.03),
    T.RandomGrayscale(0.1),
    T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
    Degrade(),
    T.ToTensor(),
    NORMALIZE,
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
EVAL_TF = T.Compose([T.CenterCrop(INPUT_SIZE), T.ToTensor(), NORMALIZE])


class Faces(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, label = self.items[i]
        return self.transform(img), label


def load_split(test_frac, seed, all_data):
    """Returns classes, train items, test items; items are (PIL image, class index)."""
    classes = sorted(d.name for d in FACES_DIR.iterdir() if d.is_dir() and d.name != UNKNOWN) + [UNKNOWN]
    rng = random.Random(seed)
    train, test = [], []
    for idx, name in enumerate(classes):
        paths = sorted(p for p in (FACES_DIR / name).iterdir() if p.suffix.lower() in IMG_EXTS)
        # split unit: a photo for known people, an identity for unknowns
        groups = {}
        for p in paths:
            groups.setdefault(p.stem.split("__")[0] if name == UNKNOWN else p.stem, []).append(p)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_test = 0 if all_data else max(1, round(len(keys) * test_frac))
        for i, key in enumerate(keys):
            for p in groups[key]:
                (test if i < n_test else train).append((Image.open(p).convert("RGB"), idx))
    return classes, train, test


def set_trainable(model, finetune):
    """Phase 1 trains only the head; phase 2 also the last backbone block."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    if finetune:
        for m in (model.backbone.block8, model.backbone.last_linear, model.backbone.last_bn):
            for p in m.parameters():
                p.requires_grad = True


def run_epochs(model, loader, device, epochs, finetune, tag):
    set_trainable(model, finetune)
    groups = [{"params": model.head.parameters(), "lr": 1e-3}]
    if finetune:
        groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epochs):
        # backbone stays in eval mode: with this little data its BatchNorm statistics must not move
        model.eval()
        model.head.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
        print(f"[{tag}] epoch {epoch + 1}/{epochs}  loss {loss_sum / total:.4f}  train acc {correct / total:.3f}")


@torch.no_grad()
def predict_probs(model, items, transform, device):
    model.eval()
    probs = [torch.softmax(model(x.to(device)), 1).cpu()
             for x, _ in DataLoader(Faces(items, transform), batch_size=64)]
    return torch.cat(probs).numpy(), np.array([y for _, y in items])


def apply_threshold(probs, threshold, unknown_idx):
    """A face is only given a name if the model is at least `threshold` sure."""
    pred = probs.argmax(1)
    pred[probs.max(1) < threshold] = unknown_idx
    return pred


def report(title, probs, y, classes, threshold):
    unk = len(classes) - 1
    pred = apply_threshold(probs, threshold, unk)
    labels = list(range(len(classes)))
    print(f"\n=== {title}  ({len(y)} faces, threshold {threshold}) ===")
    print(classification_report(y, pred, labels=labels, target_names=classes, digits=3, zero_division=0))
    print("confusion matrix (rows = truth, cols = predicted):", classes)
    print(confusion_matrix(y, pred, labels=labels))
    known = y != unk
    print(f"known faces named correctly : {(pred[known] == y[known]).sum()}/{known.sum()}")
    print(f"known faces -> wrong person : {((pred[known] != y[known]) & (pred[known] != unk)).sum()}/{known.sum()}")
    print(f"known faces -> unknown      : {(pred[known] == unk).sum()}/{known.sum()}")
    print(f"strangers given a name      : {(pred[~known] != unk).sum()}/{(~known).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=3200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--unknown-share", type=float, default=0.4, help="fraction of each batch drawn from 'unknown'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all-data", action="store_true", help="train on every photo and skip evaluation")
    ap.add_argument("--out", default=str(MODEL_PATH))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    classes, train, test = load_split(args.test_frac, args.seed, args.all_data)
    unk = len(classes) - 1
    for name, items in (("train", train), ("test", test)):
        counts = Counter(y for _, y in items)
        print(name, {c: counts[i] for i, c in enumerate(classes)})

    # every person is drawn equally often regardless of how many photos they have
    counts = Counter(y for _, y in train)
    share = {i: args.unknown_share if i == unk else (1 - args.unknown_share) / unk for i in range(len(classes))}
    weights = [share[y] / counts[y] for _, y in train]
    loader = DataLoader(
        Faces(train, TRAIN_TF), batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, args.samples_per_epoch, replacement=True),
        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)

    model = FaceClassifier(len(classes)).to(device)
    run_epochs(model, loader, device, args.head_epochs, finetune=False, tag="head")
    run_epochs(model, loader, device, args.finetune_epochs, finetune=True, tag="finetune")

    if test:
        probs, y = predict_probs(model, test, EVAL_TF, device)
        report("TEST: held-out original photos", probs, y, classes, args.threshold)
        # same held-out photos, but under the training augmentations (25 variants per known photo)
        aug = [it for it in test for _ in range(1 if it[1] == unk else 25)]
        probs_aug, y_aug = predict_probs(model, aug, TRAIN_TF, device)
        report("TEST: held-out photos, augmented", probs_aug, y_aug, classes, args.threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "threshold": args.threshold}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
