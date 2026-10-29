"""Build a small fake ImageFolder dataset to smoke-test the scripts.

Plants one exact duplicate (train -> test) and one near-duplicate (train -> val,
re-saved with a slight brightness change) so dup_scan.py has something to find.
With --shortcut, the no-tumor class also gets a different image size and a
different intensity range from the tumor classes, so shortcut_baseline.py
should find it. The images are random blobs, not MRI. Never train on this data.

Usage:
    python3 scripts/make_synthetic.py <out_dir> [--per-class 14] [--shortcut]
Writes <out_dir>/data/, <out_dir>/config.yaml and <out_dir>/exclusions.txt
(outputs go to <out_dir>/outputs).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def blob(rng, notumor_shortcut):
    small = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    if notumor_shortcut:
        small = (small // 2 + 128).astype(np.uint8)  # brighter, narrower range
    return small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--per-class", type=int, default=14)
    ap.add_argument("--shortcut", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    n_train = args.per_class
    n_eval = max(4, n_train // 3)
    counts = {"train": n_train, "val": n_eval, "test": n_eval}

    rng = np.random.default_rng(123)
    for split, n in counts.items():
        for c in CLASSES:
            d = data / split / c
            d.mkdir(parents=True)
            odd = args.shortcut and c == "notumor"
            size = (100, 130) if odd else (128, 128)
            for k in range(n):
                img = Image.fromarray(blob(rng, odd)).resize(size, Image.BICUBIC).convert("L")
                img.save(d / f"{c}_{split}_{k:03d}.jpg", quality=95)

    src = data / "train" / "glioma" / "glioma_train_000.jpg"
    shutil.copy(src, data / "test" / "glioma" / "PLANTED_exact_dup.jpg")
    near = ImageEnhance.Brightness(Image.open(src)).enhance(1.04)
    near.save(data / "val" / "glioma" / "PLANTED_near_dup.jpg", quality=90)

    excl = out / "exclusions.txt"
    excl.write_text("# synthetic test\nPLANTED_exact_dup.jpg\n")
    cfg = {
        "seeds": [0, 1, 2],
        "outputs_dir": str(out / "outputs"),
        "data": {
            "root": str(data),
            "expected_counts": {"train": n_train * 4, "val": n_eval * 4 + 1, "test": n_eval * 4 + 1},
            "imbalance_ratio_threshold": 1.5,
            "dup_thresholds": [0, 5, 10],
            "no_tumor_class": "notumor",
            "exclusions_file": str(excl),
        },
        "training": {"input_size": 224},
        "triage": {"target_sensitivity": 0.98},
    }
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    print("Synthetic dataset written to", out, "(shortcut planted)" if args.shortcut else "")


if __name__ == "__main__":
    main()
