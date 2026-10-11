"""Crop faces out of the raw photos and build the "unknown" class from LFW.

    python face_recog/prepare_data.py

Input : data/raw/<person>/*.jpg   (folder name = label)
Output: data/faces/<person>/*.jpg and data/faces/unknown/<identity>__<file>.jpg

A folder data/raw/unknown/ (optional) is treated as extra negatives: every face
found in those photos goes into the unknown class.
"""
import argparse
import random
import shutil
import tarfile

import requests
from tqdm import tqdm

from common import DATA, FACES_DIR, IMG_EXTS, RAW_DIR, UNKNOWN, detect_faces, get_device, load_image, make_detector

LFW_URL = "https://ndownloader.figshare.com/files/5976015"  # lfw-funneled.tgz (same file scikit-learn uses)
LFW_DIR = DATA / "lfw"


def download_lfw():
    root = LFW_DIR / "lfw_funneled"
    if root.exists():
        return root
    LFW_DIR.mkdir(parents=True, exist_ok=True)
    tgz = LFW_DIR / "lfw-funneled.tgz"
    with requests.get(LFW_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tgz, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="LFW download") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    with tarfile.open(tgz) as tar:
        tar.extractall(LFW_DIR, filter="data")
    tgz.unlink()
    return root


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=1500, help="number of LFW identities to use as 'unknown'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    detector = make_detector(get_device())
    if FACES_DIR.exists():
        shutil.rmtree(FACES_DIR)

    # --- the people we want to recognise: keep the largest face in each photo
    for person_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name != UNKNOWN):
        out = FACES_DIR / person_dir.name
        out.mkdir(parents=True)
        kept = 0
        for path in images_in(person_dir):
            faces = detect_faces(load_image(path), detector)
            if not faces:
                print(f"  ! no face found in {path}")
                continue
            if len(faces) > 1:
                print(f"  ! {len(faces)} faces in {path}, keeping the largest")
            faces[0][1].save(out / f"{path.stem}.jpg", quality=95)
            kept += 1
        print(f"{person_dir.name}: {kept} faces")

    # --- unknown class
    out = FACES_DIR / UNKNOWN
    out.mkdir(parents=True)
    kept = 0
    if (RAW_DIR / UNKNOWN).exists():
        for path in images_in(RAW_DIR / UNKNOWN):
            for i, (_, crop) in enumerate(detect_faces(load_image(path), detector)):
                crop.save(out / f"user-{path.stem}-{i}__{path.stem}.jpg", quality=95)
                kept += 1
        print(f"{UNKNOWN}: {kept} faces from {RAW_DIR / UNKNOWN}")

    # one image per identity so the negatives are as varied as possible
    identities = sorted(p for p in download_lfw().iterdir() if p.is_dir())
    random.Random(args.seed).shuffle(identities)
    lfw_kept = 0
    for ident in tqdm(identities[: args.negatives], desc="LFW faces"):
        path = images_in(ident)[0]
        faces = detect_faces(load_image(path), detector)
        if faces:
            faces[0][1].save(out / f"{ident.name}__{path.stem}.jpg", quality=95)
            lfw_kept += 1
    print(f"{UNKNOWN}: {lfw_kept} faces from LFW")


if __name__ == "__main__":
    main()
