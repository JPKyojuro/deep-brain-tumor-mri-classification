"""Show near-duplicate pairs from dup_scan.py side by side for visual review.

Reads outputs/setup/dup_scan_pairs.csv, keeps pairs with Hamming distance
<= --max-dist, and writes montage PNGs (a few pairs per page) to outputs/setup/.
Each pair shows split / class / file name and the hash distance, so you can
decide: same image filed twice, or just two similar-looking slices?

Usage (from the project root):
    python3 scripts/view_pairs.py [--config config.yaml] [--max-dist 5] [--per-page 6]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-dist", type=int, default=5)
    ap.add_argument("--per-page", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["outputs_dir"]) / "setup"
    csv_path = out / "dup_scan_pairs.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run scripts/dup_scan.py first.")

    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["distance"]) <= args.max_dist]
    rows.sort(key=lambda r: (int(r["distance"]), r["file_a"]))
    if not rows:
        print(f"No pairs at Hamming distance <= {args.max_dist}.")
        return

    pages = [rows[i:i + args.per_page] for i in range(0, len(rows), args.per_page)]
    for k, page in enumerate(pages, 1):
        fig, axes = plt.subplots(len(page), 2, figsize=(7, 3.4 * len(page)), squeeze=False)
        for ax_row, r in zip(axes, page):
            for ax, side in zip(ax_row, "ab"):
                path = r[f"file_{side}"]
                with Image.open(path) as im:
                    ax.imshow(im.convert("L"), cmap="gray")
                head = (f"dist {r['distance']} | ncc {r.get('ncc') or '-'} | {r['kind']}\n"
                        if side == "a" else "\n")
                ax.set_title(
                    f"{head}{r['split_' + side]} / {r['class_' + side]}\n{Path(path).name}",
                    fontsize=7,
                )
                ax.axis("off")
        fig.tight_layout()
        dest = out / f"pairs_d{args.max_dist}_page{k}.png"
        fig.savefig(dest, dpi=130)
        plt.close(fig)
        print("Wrote", dest)
    print(f"{len(rows)} pair(s) on {len(pages)} page(s).")


if __name__ == "__main__":
    main()
