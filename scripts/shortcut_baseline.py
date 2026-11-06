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

Three tables are printed:

  1. Four-class results plus binary tumor-vs-no-tumor AUC (95% bootstrap CI).
  2. Triage: P(tumor) = 1 - P(no_tumor). The threshold is chosen so that
     5-fold cross-validated TRAIN predictions reach the target sensitivity
     (config triage.target_sensitivity), then applied once to validation.
     The "best-case" column picks the threshold on validation itself, which is
     optimistic. CV folds are image-level, so slices from one patient can sit
     in train and held-out folds; the train threshold is then too aggressive
     and validation sensitivity can land below the target.
  3. Tumor-only three-class (glioma / meningioma / pituitary, no-tumor images
     removed). This is the cleaner test of whether format cues separate the
     tumor TYPES; the majority-class rate is the do-nothing reference.

How to read it: always predicting "tumor" gives binary AUC 0.5. A feature
family far above the reference is a shortcut a CNN could also use. Validation
has few no-tumor images (118), so gaps of a few AUC points are noise.

Usage (from the project root, after build_manifest.py):
    python3 scripts/shortcut_baseline.py [--config config.yaml]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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


def fit_weighted(X, y, n_classes):
    """Gradient boosting with inverse-frequency sample weights. Labels must be 0..n_classes-1."""
    counts = np.bincount(y, minlength=n_classes)
    w = (len(y) / (n_classes * np.maximum(counts, 1)))[y]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    model.fit(X, y, sample_weight=w)
    return model


def p_tumor_of(model, X, no_tumor_idx):
    return 1.0 - model.predict_proba(X)[:, list(model.classes_).index(no_tumor_idx)]


def oof_p_tumor(X, y, no_tumor_idx, n_classes, seed=0):
    """Out-of-fold P(tumor) on the training split (5-fold, stratified by class)."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = fit_weighted(X[tr], y[tr], n_classes)
        p[te] = p_tumor_of(m, X[te], no_tumor_idx)
    return p


def threshold_for_sens(p_pos, target):
    """Largest threshold t such that at least `target` of positives have p >= t."""
    return float(np.quantile(p_pos, 1.0 - target, method="lower"))


def sens_spec(p, y_tumor, t):
    pred = p >= t
    return float(pred[y_tumor == 1].mean()), float((~pred[y_tumor == 0]).mean())


def boot_ci(stat, n, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat(rng.integers(0, n, n))
        if v is not None:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    target = cfg.get("triage", {}).get("target_sensitivity", 0.98)
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
    n_classes = len(classes)
    no_tumor_idx = next(int(r["label"]) for r in rows if r["is_tumor"] == "0")

    y_tr = np.array([int(r["label"]) for r in train])
    y_va = np.array([int(r["label"]) for r in val])
    yb_tr = np.array([int(r["is_tumor"]) for r in train])
    yb_va = np.array([int(r["is_tumor"]) for r in val])

    # tumor-only three-class setup
    tumor_labels = sorted(set(y_tr[yb_tr == 1]))
    remap = {lab: i for i, lab in enumerate(tumor_labels)}
    ytt = np.array([remap[v] for v in y_tr[yb_tr == 1]])
    yvt = np.array([remap[v] for v in y_va[yb_va == 1]])
    tumor_names = [classes[lab] for lab in tumor_labels]
    maj3 = float(np.bincount(yvt).max() / len(yvt))

    base_acc = yb_va.mean()
    n_neg = int((yb_va == 0).sum())
    print(f"Train {len(train)} / val {len(val)} images (test untouched, excluded images removed).")
    print(f"Reference: always predicting 'tumor' -> binary accuracy {base_acc:.3f}, AUC 0.500")
    print(f"Validation has {n_neg} no-tumor images, so intervals are wide.")
    print("Fitting (5-fold CV on train for the triage threshold takes a minute or two) ...\n")

    results = {}
    for fam in FAMILIES:
        X_tr = np.array([vec(r, fam) for r in train])
        X_va = np.array([vec(r, fam) for r in val])
        res = {}

        # --- four-class model, binary AUC
        model = fit_weighted(X_tr, y_tr, n_classes)
        proba = model.predict_proba(X_va)
        pred = model.classes_[proba.argmax(axis=1)]
        p_va = p_tumor_of(model, X_va, no_tumor_idx)

        def auc_stat(idx, y=yb_va, p=p_va):
            return None if y[idx].min() == y[idx].max() else roc_auc_score(y[idx], p[idx])

        res["acc4"] = float(accuracy_score(y_va, pred))
        res["macro_f1"] = float(f1_score(y_va, pred, average="macro"))
        res["auc"] = float(roc_auc_score(yb_va, p_va))
        res["auc_ci95"] = boot_ci(auc_stat, len(yb_va))

        # --- triage threshold from cross-validated train predictions
        p_oof = oof_p_tumor(X_tr, y_tr, no_tumor_idx, n_classes)
        t_cv = threshold_for_sens(p_oof[yb_tr == 1], target)
        res["triage_threshold_cv"] = t_cv
        res["val_sens_at_cv_threshold"], res["val_spec_at_cv_threshold"] = sens_spec(p_va, yb_va, t_cv)
        t_best = threshold_for_sens(p_va[yb_va == 1], target)
        _, res["val_spec_best_case"] = sens_spec(p_va, yb_va, t_best)

        # --- tumor-only three-class
        m3 = fit_weighted(X_tr[yb_tr == 1], ytt, len(tumor_labels))
        pred3 = m3.predict(X_va[yb_va == 1])
        res["acc3"] = float(accuracy_score(yvt, pred3))
        res["macro_f1_3"] = float(f1_score(yvt, pred3, average="macro"))
        res["acc3_ci95"] = boot_ci(lambda idx: float((pred3[idx] == yvt[idx]).mean()), len(yvt))
        results[fam] = res

    print("Table 1: four-class model, and binary tumor vs no-tumor")
    print(f"{'features':<13}{'4-class acc':>12}{'macro-F1':>10}{'binary AUC':>12}{'95% CI':>18}")
    for fam in FAMILIES:
        r = results[fam]
        lo, hi = r["auc_ci95"]
        print(f"{fam:<13}{r['acc4']:>12.3f}{r['macro_f1']:>10.3f}{r['auc']:>12.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")

    print(f"\nTable 2: triage at {target:.0%} target sensitivity "
          f"(threshold from 5-fold CV on train, applied to val)")
    print(f"{'features':<13}{'threshold':>10}{'val sens':>10}{'val spec':>10}{'best-case spec':>16}")
    for fam in FAMILIES:
        r = results[fam]
        print(f"{fam:<13}{r['triage_threshold_cv']:>10.3f}{r['val_sens_at_cv_threshold']:>10.3f}"
              f"{r['val_spec_at_cv_threshold']:>10.3f}{r['val_spec_best_case']:>16.3f}")
    print(f"  Best-case spec picks the threshold on validation itself (optimistic). "
          f"Specificity rests on {n_neg} images; one false alarm = {100 / n_neg:.2f} points.")

    print(f"\nTable 3: tumor-only {len(tumor_names)}-class ({', '.join(tumor_names)}), "
          f"{len(yvt)} validation images")
    print(f"Reference: always predicting the majority tumor class -> accuracy {maj3:.3f}")
    print(f"{'features':<13}{'accuracy':>10}{'95% CI':>18}{'macro-F1':>10}")
    for fam in FAMILIES:
        r = results[fam]
        lo, hi = r["acc3_ci95"]
        print(f"{fam:<13}{r['acc3']:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{r['macro_f1_3']:>10.3f}")

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shortcut_baseline.json", "w") as f:
        json.dump({"reference_binary_accuracy": float(base_acc), "target_sensitivity": target,
                   "tumor_classes": tumor_names, "majority_tumor_class_acc": maj3,
                   "results": results, "classes": classes, "n_train": len(train),
                   "n_val": len(val), "n_val_no_tumor": n_neg}, f, indent=2)
    print(f"\nWrote {out / 'shortcut_baseline.json'}")


if __name__ == "__main__":
    main()
