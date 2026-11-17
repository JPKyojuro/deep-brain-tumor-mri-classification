"""Write outputs/manifest.csv: one row per image.

Columns: path, split, cls, label, is_tumor, excluded, then raw-file metadata
(width, height, aspect, mode, bytes, size bucket, colour flag) and features
computed on the *preprocessed* 224x224 grayscale image:

  whole-image intensity   mean, std, dark_frac, h00..h15
                          (includes black padding and dark background)
  head geometry           head_frac, bbox_w, bbox_h
                          (head mask from common.head_mask, as fractions of the
                           NON-padded area, so letterboxing does not enter; the
                           remaining signal is how much of the frame the head fills)
  head-only intensity     m_mean, m_std, m_h00..m_h15
                          (only pixels inside the head mask)
  head-only, normalized   n_mean, n_std, n_h00..n_h15
                          (head pixels rescaled per image to the 1st-99th
                           percentile, so overall brightness scale is removed)

Excluded images (exclusions.txt) stay in the file with excluded=1, so nothing
is silently dropped; downstream code filters on it.

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
from common import (head_mask, list_dataset, load_config, list_external, load_exclusions, to_square_gray,
                    valid_box)
from metadata_by_class import COLOR_TOL, size_bucket

N_BINS = 16


def hist16(values, lo, hi):
    h, _ = np.histogram(values, bins=N_BINS, range=(lo, hi))
    return h / max(1, values.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--external", help="path to dataset [6]; writes manifest_external.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg.get("training", {}).get("input_size", 224)

    # Introduces external flag to switch to [6] dataset
    records = list_external(args.external) if args.external else list_dataset(cfg["data"]["root"])

    classes = sorted({c for _, c, _ in records})
    no_tumor = cfg["data"].get("no_tumor_class")
    if no_tumor is None:
        hits = [c for c in classes if NO_TUMOR_PATTERN.match(c)]
        no_tumor = hits[0] if len(hits) == 1 else None
    if no_tumor not in classes:
        raise SystemExit(f"No-tumor class not found among {classes}; "
                         "set data.no_tumor_class in config.yaml.")
    label_of = {c: i for i, c in enumerate(classes)}

    excluded_names = set() if args.external else load_exclusions(cfg)

    present = {p.name for _, _, p in records}
    missing = sorted(excluded_names - present)
    if missing:
        print(f"WARNING: {len(missing)} name(s) in the exclusions file were not found:", missing[:5])

    fields = (["path", "split", "cls", "label", "is_tumor", "excluded", "width", "height",
               "aspect", "mode", "bytes", "bucket", "colour", "mean", "std", "dark_frac"]
              + [f"h{i:02d}" for i in range(N_BINS)]
              + ["head_frac", "bbox_w", "bbox_h", "m_mean", "m_std"]
              + [f"m_h{i:02d}" for i in range(N_BINS)]
              + ["n_mean", "n_std"]
              + [f"n_h{i:02d}" for i in range(N_BINS)])
    out_dir = Path(cfg["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("manifest.csv" if not args.external else "manifest_external.csv")

    print(f"Reading {len(records)} images ...")
    n_excl = Counter()
    head_fracs = []
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

            hist = hist16(arr.ravel(), 0, 256)
            valid = valid_box((wd, ht), size)  # non-padded area; padding never enters the mask
            vw, vh = valid[2] - valid[0], valid[3] - valid[1]
            mask = head_mask(arr, valid)
            hf_img = mask.sum() / (vw * vh)
            ys, xs = np.where(mask)
            px = arr[mask]
            lo, hi = np.percentile(px, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((px - lo) / (hi - lo), 0.0, 1.0)
            head_fracs.append(hf_img)

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
                "head_frac": f"{hf_img:.4f}",
                "bbox_w": f"{(xs.max() - xs.min() + 1) / vw:.4f}",
                "bbox_h": f"{(ys.max() - ys.min() + 1) / vh:.4f}",
                "m_mean": f"{px.mean():.3f}", "m_std": f"{px.std():.3f}",
                "n_mean": f"{norm.mean():.4f}", "n_std": f"{norm.std():.4f}",
            }
            for i, v in enumerate(hist):
                row[f"h{i:02d}"] = f"{v:.5f}"
            for i, v in enumerate(hist16(px, 0, 256)):
                row[f"m_h{i:02d}"] = f"{v:.5f}"
            for i, v in enumerate(hist16(norm, 0.0, 1.0)):
                row[f"n_h{i:02d}"] = f"{v:.5f}"
            w.writerow(row)

    hf = np.array(head_fracs)
    print(f"Wrote {out_path}")
    print(f"No-tumor class: {no_tumor}. Classes: {classes}")
    print(f"Excluded images: {sum(n_excl.values())}", dict(n_excl) if n_excl else "")
    print(f"Head mask covers {100 * hf.mean():.0f}% of the image on average "
          f"(5th-95th percentile: {100 * np.percentile(hf, 5):.0f}%-{100 * np.percentile(hf, 95):.0f}%). "
          "Check a few masks by eye with scripts/view_masks.py.")


if __name__ == "__main__":
    main()
