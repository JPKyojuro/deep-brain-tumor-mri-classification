"""Eyeball the head mask: random images per class with the mask outline drawn on top.

The masked baseline is only as good as the mask, so look at a few before trusting it.
Each tile is the preprocessed 224x224 grayscale image; the red line is the head-mask
boundary. Watch for masks that include background noise, or cut off dark parts of the head.

Usage (from the project root):
    python3 scripts/view_masks.py [--config config.yaml] [--per-class 6] [--seed 0]
Writes outputs/setup/head_masks.png
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import head_mask, list_dataset, load_config, to_square_gray, valid_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg.get("training", {}).get("input_size", 224)

    by_class = defaultdict(list)
    for split, cls, path in list_dataset(cfg["data"]["root"]):
        if split != "test":  # keep the test split unseen
            by_class[cls].append(path)
    classes = sorted(by_class)
    rng = random.Random(args.seed)

    fig, axes = plt.subplots(len(classes), args.per_class,
                             figsize=(2.2 * args.per_class, 2.4 * len(classes)), squeeze=False)
    for r, cls in enumerate(classes):
        picks = rng.sample(by_class[cls], min(args.per_class, len(by_class[cls])))
        for c in range(args.per_class):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(picks):
                continue
            with Image.open(picks[c]) as im:
                wh = im.size
                arr = np.asarray(to_square_gray(im, size), dtype=np.float32)
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            mask = head_mask(arr, valid_box(wh, size))
            ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=0.8)
            ax.set_title(f"{cls}\n{picks[c].name}", fontsize=5)
    fig.tight_layout()
    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "head_masks.png", dpi=140)
    print("Wrote", out / "head_masks.png")


if __name__ == "__main__":
    main()
