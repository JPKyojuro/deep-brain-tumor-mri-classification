"""Step 3: class balance per split, stratification check, and class weights.

Also reports how many tumor-positive images the validation split has, and how
many misses a 98% (config.triage.target_sensitivity) sensitivity target allows.
That number is what the triage threshold will be tuned against, so it is worth
knowing before training starts.

Usage (from the project root):
    python scripts/class_balance.py [--config config.yaml] [--no-tumor-class NAME]
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SPLITS, list_dataset, load_config

NO_TUMOR_PATTERN = re.compile(r"^(no[ _-]?tumou?r|notumou?r|normal|healthy)$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-tumor-class", default=None,
                    help="folder name of the no-tumor class (auto-detected if omitted)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    records = list_dataset(cfg["data"]["root"])
    classes = sorted({c for _, c, _ in records})
    counts = {s: Counter() for s in SPLITS}
    for split, cls, _ in records:
        counts[split][cls] += 1

    no_tumor = args.no_tumor_class
    if no_tumor is None:
        hits = [c for c in classes if NO_TUMOR_PATTERN.match(c)]
        no_tumor = hits[0] if len(hits) == 1 else None
    print("Classes:", classes)
    print("No-tumor class:", no_tumor or "NOT DETECTED (pass --no-tumor-class)")

    print(f"\n{'class':<14}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'total':>9}  "
          + "  ".join(f"%{s}" for s in SPLITS))
    for c in classes:
        tot = sum(counts[s][c] for s in SPLITS)
        pct = "  ".join(f"{100 * counts[s][c] / tot:4.1f}" for s in SPLITS)
        print(f"{c:<14}" + "".join(f"{counts[s][c]:>10}" for s in SPLITS) + f"{tot:>9}  {pct}")

    # stratification: per-class share of each split vs the dataset-wide share
    split_tot = {s: sum(counts[s].values()) for s in SPLITS}
    grand = sum(split_tot.values())
    max_dev = 0.0
    for s in SPLITS:
        for c in classes:
            overall = sum(counts[x][c] for x in SPLITS) / grand
            max_dev = max(max_dev, abs(counts[s][c] / split_tot[s] - overall))
    print(f"\nMax class-share deviation from dataset-wide share across splits: "
          f"{100 * max_dev:.2f} percentage points")

    train = counts["train"]
    ratio = max(train.values()) / min(train.values())
    thr = cfg["data"]["imbalance_ratio_threshold"]
    use_weights = ratio > thr
    print(f"Train imbalance ratio (max/min): {ratio:.2f} (threshold {thr}) -> "
          f"class-weighted loss {'RECOMMENDED' if use_weights else 'not needed'}")
    n_train = sum(train.values())
    weights = {c: n_train / (len(classes) * train[c]) for c in classes}  # inverse frequency
    print("Inverse-frequency class weights (train):",
          {c: round(w, 3) for c, w in weights.items()})

    result = {"classes": classes, "counts": {s: dict(counts[s]) for s in SPLITS},
              "max_share_deviation_pp": 100 * max_dev, "train_imbalance_ratio": ratio,
              "use_class_weights": use_weights, "class_weights": weights,
              "no_tumor_class": no_tumor}

    if no_tumor:
        target = cfg["triage"]["target_sensitivity"]
        print(f"\nTriage view (tumor vs {no_tumor}), target sensitivity {target:.0%}:")
        triage = {}
        for s in SPLITS:
            pos = split_tot[s] - counts[s][no_tumor]
            allowed = math.floor(pos * (1 - target))
            print(f"  {s:<6} tumor-positive: {pos:>5}  no-tumor: {counts[s][no_tumor]:>5}  "
                  f"misses allowed at target: {allowed}")
            triage[s] = {"positives": pos, "negatives": counts[s][no_tumor],
                         "misses_allowed_at_target": allowed}
        result["triage"] = triage
        vp, vn = triage["val"]["positives"], triage["val"]["negatives"]
        print(f"  Val resolution: one missed tumor = {100 / vp:.2f} points of sensitivity; "
              f"one false alarm = {100 / vn:.2f} points of specificity.")
        print(f"  Specificity rests on only {vn} no-tumor validation images, "
              f"so expect it to be the noisier number.")

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "class_balance.json", "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    for k, s in enumerate(SPLITS):
        ax.bar([i + (k - 1) * width for i in range(len(classes))],
               [counts[s][c] for c in classes], width, label=s)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("images")
    ax.set_title("Class counts per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    print(f"\nWrote {out / 'class_balance.json'} and {out / 'class_balance.png'}")


if __name__ == "__main__":
    main()
