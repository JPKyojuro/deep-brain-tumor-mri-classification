"""Can a model separate the classes WITHOUT looking at anatomy?

Trains a small gradient-boosting model on features that carry no spatial
information about the brain, one feature family at a time:

  metadata     file size, aspect, mode, bytes, colour flag
  intensity    mean, std, dark fraction and 16-bin histogram of the WHOLE
               preprocessed image (includes black padding and background)
  geometry     how much of the frame the head fills (head_frac, bbox_w, bbox_h)
  masked       mean, std and histogram of pixels inside the head mask only
  masked_norm  same, after rescaling each image to its own 1st-99th percentile
  all          everything above

Fit on the training split, scored on the VALIDATION split only. The test
split is dropped as soon as the manifest is read, so it stays sealed.

How to read it: always predicting "tumor" gives AUC 0.5 and accuracy equal to
the tumor share of validation. If "intensity" is high but "masked" and
"masked_norm" are near 0.5, the whole-image histogram was mostly picking up
padding and frame size. If "masked_norm" stays high, the signal sits in the
brain pixels themselves (different scanners, sequences, or patient leakage).
"geometry" high means the field of view differs between classes.

The AUC interval is a 95% bootstrap over validation images (300 resamples).
Validation has few no-tumor images (118), so treat gaps of a few AUC points
as noise.

Usage (from the project root, after build_manifest.py):
    python3 scripts/shortcut_baseline.py [--config config.yaml]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from common import load_config

H = [f"h{i:02d}" for i in range(16)]
COLS = {
    "intensity": ["mean", "std", "dark_frac"] + H,
    "geometry": ["head_frac", "bbox_w", "bbox_h"],
    "masked": ["m_mean", "m_std"] + [f"m_{h}" for h in H],
    "masked_norm": ["n_mean", "n_std"] + [f"n_{h}" for h in H],
}
FAMILIES = ["metadata", "intensity", "geometry", "masked", "masked_norm", "all"]


def vec(r, family):
    kinds = ["metadata", "intensity", "geometry", "masked", "masked_norm"] if family == "all" else [family]
    v = []
    for k in kinds:
        if k == "metadata":
            v += [float(r["width"]), float(r["height"]), float(r["aspect"]),
                  float(np.log1p(float(r["bytes"]))),
                  float(r["mode"] == "L"), float(r["mode"] == "RGB"),
                  float(r["mode"] == "RGBA"), float(r["colour"])]
        else:
            v += [float(r[c]) for c in COLS[k]]
    return v


def auc_ci(y, p, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = Path(cfg["outputs_dir"]) / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/build_manifest.py first.")

    with open(path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows or "m_mean" not in all_rows[0]:
        raise SystemExit("manifest.csv is from an older version (no head-mask columns). "
                         "Rerun scripts/build_manifest.py.")
    rows = [r for r in all_rows if r["excluded"] == "0" and r["split"] != "test"]  # test sealed
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    classes = sorted({r["cls"] for r in rows},
                     key=lambda c: int(next(r["label"] for r in rows if r["cls"] == c)))
    no_tumor_idx = next(int(r["label"]) for r in rows if r["is_tumor"] == "0")

    y_tr = np.array([int(r["label"]) for r in train])
    y_va = np.array([int(r["label"]) for r in val])
    yb_va = np.array([int(r["is_tumor"]) for r in val])
    counts = np.bincount(y_tr, minlength=len(classes))
    sample_w = (len(y_tr) / (len(classes) * counts))[y_tr]  # inverse-frequency weights

    base_acc = yb_va.mean()
    n_neg = int((yb_va == 0).sum())
    print(f"Train {len(train)} / val {len(val)} images (test untouched, excluded images removed).")
    print(f"Reference: always predicting 'tumor' -> binary accuracy {base_acc:.3f}, AUC 0.500")
    print(f"Validation has {n_neg} no-tumor images, so AUC intervals are wide.\n")
    print(f"{'features':<13}{'4-class acc':>12}{'macro-F1':>10}{'binary AUC':>12}{'95% CI':>16}"
          f"{'sens':>7}{'spec':>7}")

    results = {}
    for fam in FAMILIES:
        X_tr = np.array([vec(r, fam) for r in train])
        X_va = np.array([vec(r, fam) for r in val])
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        model.fit(X_tr, y_tr, sample_weight=sample_w)
        proba = model.predict_proba(X_va)
        pred = model.classes_[proba.argmax(axis=1)]
        p_tumor = 1.0 - proba[:, list(model.classes_).index(no_tumor_idx)]
        pred_b = (pred != no_tumor_idx).astype(int)
        lo, hi = auc_ci(yb_va, p_tumor)
        res = {
            "acc4": accuracy_score(y_va, pred),
            "macro_f1": f1_score(y_va, pred, average="macro"),
            "auc": roc_auc_score(yb_va, p_tumor),
            "auc_ci95": [lo, hi],
            "sensitivity": recall_score(yb_va, pred_b, pos_label=1),
            "specificity": recall_score(yb_va, pred_b, pos_label=0),
        }
        results[fam] = res
        print(f"{fam:<13}{res['acc4']:>12.3f}{res['macro_f1']:>10.3f}{res['auc']:>12.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>16}{res['sensitivity']:>7.3f}{res['specificity']:>7.3f}")

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shortcut_baseline.json", "w") as f:
        json.dump({"reference_binary_accuracy": float(base_acc), "results": results,
                   "classes": classes, "n_train": len(train), "n_val": len(val),
                   "n_val_no_tumor": n_neg}, f, indent=2)
    print(f"\nWrote {out / 'shortcut_baseline.json'}")
    print("Compare intensity vs masked / masked_norm: a big drop means the whole-image "
          "histogram was mostly reading padding and frame size.")


if __name__ == "__main__":
    main()
