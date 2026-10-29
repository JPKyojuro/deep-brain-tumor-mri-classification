"""Can a model separate the classes WITHOUT looking at anatomy?

Trains a small gradient-boosting model on features that carry no spatial
information about the brain: (1) file metadata (size, aspect, mode, bytes,
colour flag), (2) global intensity statistics (mean, std, dark fraction,
16-bin histogram of the preprocessed image), and (3) both. Fit on the training
split, scored on the VALIDATION split only. The test split is dropped as soon
as the manifest is read, so it stays sealed.

How to read the result: always predicting "tumor" gives AUC 0.5 and accuracy
equal to the tumor share of validation. A feature family that scores well above
that is a shortcut a CNN can also use. AUC near 0.5 means that family does not
reveal the label.

Usage (from the project root, after build_manifest.py):
    python3 scripts/shortcut_baseline.py [--config config.yaml]
Needs scikit-learn.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from common import load_config

META_COLS = ["width", "height", "aspect", "bytes"]
INTEN_COLS = ["mean", "std", "dark_frac"] + [f"h{i:02d}" for i in range(16)]


def features(rows, kind):
    cols = []
    for r in rows:
        v = []
        if kind in ("metadata", "both"):
            v += [float(r[c]) for c in META_COLS]
            v[3] = np.log1p(v[3])
            v += [float(r["mode"] == "L"), float(r["mode"] == "RGB"),
                  float(r["mode"] == "RGBA"), float(r["colour"])]
        if kind in ("intensity", "both"):
            v += [float(r[c]) for c in INTEN_COLS]
        cols.append(v)
    return np.array(cols, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = Path(cfg["outputs_dir"]) / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/build_manifest.py first.")

    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["excluded"] == "0" and r["split"] != "test"]  # test stays sealed
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    classes = sorted({r["cls"] for r in rows}, key=lambda c: int(
        next(r["label"] for r in rows if r["cls"] == c)))
    no_tumor_idx = next(int(r["label"]) for r in rows if r["is_tumor"] == "0")

    y_tr = np.array([int(r["label"]) for r in train])
    y_va = np.array([int(r["label"]) for r in val])
    yb_va = np.array([int(r["is_tumor"]) for r in val])
    counts = np.bincount(y_tr, minlength=len(classes))
    sample_w = (len(y_tr) / (len(classes) * counts))[y_tr]  # inverse-frequency weights

    base_acc = yb_va.mean()
    print(f"Train {len(train)} / val {len(val)} images (test untouched, excluded images removed).")
    print(f"Reference: always predicting 'tumor' -> binary accuracy {base_acc:.3f}, AUC 0.500\n")
    print(f"{'features':<11}{'4-class acc':>12}{'macro-F1':>10}{'binary AUC':>12}"
          f"{'sens':>8}{'spec':>8}{'no-tumor recall':>17}")

    results = {}
    for kind in ("metadata", "intensity", "both"):
        X_tr, X_va = features(train, kind), features(val, kind)
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        model.fit(X_tr, y_tr, sample_weight=sample_w)
        proba = model.predict_proba(X_va)
        pred = model.classes_[proba.argmax(axis=1)]
        p_tumor = 1.0 - proba[:, list(model.classes_).index(no_tumor_idx)]
        pred_b = (pred != no_tumor_idx).astype(int)
        res = {
            "acc4": accuracy_score(y_va, pred),
            "macro_f1": f1_score(y_va, pred, average="macro"),
            "auc": roc_auc_score(yb_va, p_tumor),
            "sensitivity": recall_score(yb_va, pred_b, pos_label=1),
            "specificity": recall_score(yb_va, pred_b, pos_label=0),
            "no_tumor_recall_4class": float(((pred == no_tumor_idx) & (y_va == no_tumor_idx)).sum()
                                            / max(1, (y_va == no_tumor_idx).sum())),
        }
        results[kind] = res
        print(f"{kind:<11}{res['acc4']:>12.3f}{res['macro_f1']:>10.3f}{res['auc']:>12.3f}"
              f"{res['sensitivity']:>8.3f}{res['specificity']:>8.3f}"
              f"{res['no_tumor_recall_4class']:>17.3f}")

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shortcut_baseline.json", "w") as f:
        json.dump({"reference_binary_accuracy": float(base_acc), "results": results,
                   "classes": classes, "n_train": len(train), "n_val": len(val)}, f, indent=2)
    print(f"\nWrote {out / 'shortcut_baseline.json'}")
    print("AUC well above 0.5 for a feature family = that family leaks the label.")


if __name__ == "__main__":
    main()
