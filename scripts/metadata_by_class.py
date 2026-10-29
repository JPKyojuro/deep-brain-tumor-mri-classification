"""Check whether image format tracks the label (a shortcut-learning risk).

If, say, most no-tumor images are 512x512 and most tumor images are 224x224, a
network can score well by reading image size or colour mode instead of anatomy.
This prints how image mode, size bucket and "truly colour" images are spread
across classes and splits. Compare each column against the overall class mix.

Usage (from the project root):
    python3 scripts/metadata_by_class.py [--config config.yaml]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from common import SPLITS, list_dataset, load_config

COLOR_TOL = 8  # max per-pixel channel difference still treated as grayscale


def size_bucket(size):
    w, h = size
    if size == (512, 512):
        return "512x512"
    if size == (224, 224):
        return "224x224"
    return "other square" if w == h else "non-square"


def table(title, rows, cols, counts):
    print(f"\n{title}  (count, then % of that row's total)")
    print(f"{'':<14}" + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        tot = sum(counts[(r, c)] for c in cols)
        if tot == 0:
            continue
        cells = "".join(f"{counts[(r, c)]:>8} ({100 * counts[(r, c)] / tot:4.1f}%)" for c in cols)
        print(f"{r:<14}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    records = list_dataset(cfg["data"]["root"])
    classes = sorted({c for _, c, _ in records})

    by_mode, by_bucket, by_color, by_split_bucket = Counter(), Counter(), Counter(), Counter()
    for split, cls, path in records:
        with Image.open(path) as im:
            mode, size = im.mode, im.size
            if mode in ("RGB", "RGBA"):
                a = np.asarray(im.convert("RGB")).astype(np.int16)
                colored = int((a.max(axis=2) - a.min(axis=2)).max() > COLOR_TOL)
            else:
                colored = 0
        bucket = size_bucket(size)
        by_mode[(mode, cls)] += 1
        by_bucket[(bucket, cls)] += 1
        by_color[("colour" if colored else "grayscale", cls)] += 1
        by_split_bucket[(bucket, split)] += 1

    modes = sorted({m for m, _ in by_mode})
    buckets = ["512x512", "224x224", "other square", "non-square"]
    table("Image mode by class", modes, classes, by_mode)
    table("Size bucket by class", buckets, classes, by_bucket)
    table("Colour content by class (RGB/RGBA/L files)", ["grayscale", "colour"], classes, by_color)
    table("Size bucket by split", buckets, list(SPLITS), by_split_bucket)

    overall = Counter(c for _, c, _ in records)
    n = sum(overall.values())
    print("\nOverall class mix: " + ", ".join(f"{c} {100 * overall[c] / n:.1f}%" for c in classes))
    print("A row whose class mix differs a lot from the overall mix is a shortcut risk.")

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metadata_by_class.json", "w") as f:
        json.dump({
            "mode_by_class": {f"{m}|{c}": v for (m, c), v in by_mode.items()},
            "bucket_by_class": {f"{b}|{c}": v for (b, c), v in by_bucket.items()},
            "colour_by_class": {f"{k}|{c}": v for (k, c), v in by_color.items()},
            "bucket_by_split": {f"{b}|{s}": v for (b, s), v in by_split_bucket.items()},
        }, f, indent=2)
    print("Wrote", out / "metadata_by_class.json")


if __name__ == "__main__":
    main()
