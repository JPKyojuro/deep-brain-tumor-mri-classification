"""Write outputs/manifest.csv: one row per image.

Columns: path, split, cls, label, is_tumor, excluded, then raw-file metadata
(width, height, aspect, mode, bytes, size bucket, colour flag) and global
intensity features computed on the *preprocessed* image (mean, std, dark
fraction, 16-bin histogram). Excluded images (exclusions.txt) stay in the file
with excluded=1, so nothing is silently dropped; downstream code filters on it.

The manifest feeds scripts/shortcut_baseline.py now, and later the training
and evaluation stages (including per-format subgroup reporting).

Usage (from the project root):
    python3 scripts/build_manifest.py [--config config.yaml]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from class_balance import NO_TUMOR_PATTERN
from common import list_dataset, load_config, load_exclusions, to_square_gray
from metadata_by_class import COLOR_TOL, size_bucket

N_BINS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg.get("training", {}).get("input_size", 224)

    records = list_dataset(cfg["data"]["root"])
    classes = sorted({c for _, c, _ in records})
    no_tumor = cfg["data"].get("no_tumor_class")
    if no_tumor is None:
        hits = [c for c in classes if NO_TUMOR_PATTERN.match(c)]
        no_tumor = hits[0] if len(hits) == 1 else None
    if no_tumor not in classes:
        raise SystemExit(f"No-tumor class not found among {classes}; "
                         "set data.no_tumor_class in config.yaml.")
    label_of = {c: i for i, c in enumerate(classes)}

    excluded_names = load_exclusions(cfg)
    present = {p.name for _, _, p in records}
    missing = sorted(excluded_names - present)
    if missing:
        print(f"WARNING: {len(missing)} name(s) in the exclusions file were not found:", missing[:5])

    fields = (["path", "split", "cls", "label", "is_tumor", "excluded", "width", "height",
               "aspect", "mode", "bytes", "bucket", "colour", "mean", "std", "dark_frac"]
              + [f"h{i:02d}" for i in range(N_BINS)])
    out_dir = Path(cfg["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"

    print(f"Reading {len(records)} images ...")
    n_excl = Counter()
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, cls, path in records:
            with Image.open(path) as im:
                mode, (wd, ht) = im.mode, im.size
                if mode in ("RGB", "RGBA"):
                    a = np.asarray(im.convert("RGB")).astype(np.int16)
                    colour = int((a.max(axis=2) - a.min(axis=2)).max() > COLOR_TOL)
                else:
                    colour = 0
                arr = np.asarray(to_square_gray(im, size), dtype=np.float32)
            hist, _ = np.histogram(arr, bins=N_BINS, range=(0, 256))
            hist = hist / arr.size
            excluded = int(path.name in excluded_names)
            if excluded:
                n_excl[(split, cls)] += 1
            row = {
                "path": str(path), "split": split, "cls": cls, "label": label_of[cls],
                "is_tumor": int(cls != no_tumor), "excluded": excluded,
                "width": wd, "height": ht, "aspect": f"{wd / ht:.4f}", "mode": mode,
                "bytes": path.stat().st_size, "bucket": size_bucket((wd, ht)), "colour": colour,
                "mean": f"{arr.mean():.3f}", "std": f"{arr.std():.3f}",
                "dark_frac": f"{(arr < 10).mean():.4f}",
            }
            for i, v in enumerate(hist):
                row[f"h{i:02d}"] = f"{v:.5f}"
            w.writerow(row)

    print(f"Wrote {out_path}")
    print(f"No-tumor class: {no_tumor}. Classes: {classes}")
    print(f"Excluded images: {sum(n_excl.values())}", dict(n_excl) if n_excl else "")


if __name__ == "__main__":
    main()
