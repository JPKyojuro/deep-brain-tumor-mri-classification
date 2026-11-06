"""Show random RAW images so you can see how no-tumor and tumor images differ.

Writes two contact sheets to outputs/setup/:
  samples_no_tumor.png   random no-tumor images
  samples_tumor.png      random tumor images, an equal number per tumor class
Each tile is the image as stored (not resized or normalized), titled with its
class, file name, pixel size and colour mode. Look for differences in imaging
plane, sequence or modality (e.g. bright skull on CT, bright CSF on FLAIR),
field of view, borders, and colour. Only train and val images are shown;
excluded images are skipped.

Usage (from the project root):
    python3 scripts/view_samples.py [--config config.yaml] [--n 24] [--cols 6] [--seed 0]
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from class_balance import NO_TUMOR_PATTERN
from common import list_dataset, load_config, load_exclusions


def sheet(items, cols, dest, title):
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (cls, path) in zip(axes.ravel(), items):
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            ax.imshow(im.convert("L"), cmap="gray")
        ax.set_title(f"{cls}\n{path.name}\n{w}x{h} {mode}", fontsize=5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print("Wrote", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    excluded = load_exclusions(cfg)

    by_class = defaultdict(list)
    for split, cls, path in list_dataset(cfg["data"]["root"]):
        if split != "test" and path.name not in excluded:  # keep the test split unseen
            by_class[cls].append(path)
    classes = sorted(by_class)
    no_tumor = cfg["data"].get("no_tumor_class")
    if no_tumor not in by_class:
        hits = [c for c in classes if NO_TUMOR_PATTERN.match(c)]
        if len(hits) != 1:
            raise SystemExit(f"Cannot tell which class is no-tumor among {classes}; "
                             "set data.no_tumor_class in config.yaml.")
        no_tumor = hits[0]
    tumor_classes = [c for c in classes if c != no_tumor]
    rng = random.Random(args.seed)

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)

    picks = rng.sample(by_class[no_tumor], min(args.n, len(by_class[no_tumor])))
    sheet([(no_tumor, p) for p in picks], args.cols, out / "samples_no_tumor.png",
          f"Random {no_tumor} images (raw)")

    per = max(1, args.n // len(tumor_classes))
    items = []
    for c in tumor_classes:
        items += [(c, p) for p in rng.sample(by_class[c], min(per, len(by_class[c])))]
    sheet(items, args.cols, out / "samples_tumor.png",
          f"Random tumor images (raw), {per} per class: {', '.join(tumor_classes)}")


if __name__ == "__main__":
    main()
