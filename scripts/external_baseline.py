"""scripts/external_baseline.py: train on BDNeuro train, score on the [6] residual set."""
import csv
from pathlib import Path

import numpy as np

from common import load_config
from shortcut_baseline import (FAMILIES, fit_weighted, oof_p_tumor, p_tumor_of,
                               sens_spec, threshold_for_sens, vec)

cfg = load_config()
target = cfg.get("triage", {}).get("target_sensitivity", 0.98)
out = Path(cfg['outputs_dir'])

def read(name):
    with open(out / name, newline="") as f:
        return list(csv.DictReader(f))


def features(r, fam):
    v = vec(r, fam)
    if fam in ("metadata", "all"):
        v[4:8] = [0.0] * 4             # mode and color flags: [6] is all RGB, s ohide them
    return v

def arrays(rows, fam):
    X = np.array([features(r, fam) for r in rows])
    j = np.array([int(r['label']) for r in rows])
    k = np.array([int(r['is_tumor']) for r in rows])
    return X, j, k

bd = [r for r in read("manifest.csv") if r["excluded"] == "0"]
train = [r for r in bd if r["split"] == "train"]
val = [r for r in bd if r["split"] == "val"]          # BDNeuro test stays sealed
keep = set((out / "setup" / "external_residual.txt").read_text().splitlines())
ext = [r for r in read("manifest_external.csv") if r["path"] in keep]
print(len(ext), "residual images")

assert {r["cls"]: r["label"] for r in train} == {r["cls"]: r["label"] for r in ext}, "label mismatch"
no_tumor_idx = next(int(r["label"]) for r in train if r["is_tumor"] == "0")

# check (c): is the size cue still in the residual?
for cls in sorted({r["cls"] for r in ext}):
    rs = [r for r in ext if r["cls"] == cls]
    share = np.mean([r["width"] == "512" and r["height"] == "512" for r in rs])
    print(f"{cls:<12}{len(rs):>5} images, 512x512 share {share:.2f}")

for fam in FAMILIES:
    X_tr, y_tr, yb_tr = arrays(train, fam)
    model = fit_weighted(X_tr, y_tr, 4)
    t = threshold_for_sens(oof_p_tumor(X_tr, y_tr, no_tumor_idx, 4)[yb_tr == 1], target)
    for name, rows in (("BDNeuro val", val), ("[6] residual", ext)):
        X, y, yb = arrays(rows, fam)
        p = p_tumor_of(model, X, no_tumor_idx)
        sens, spec = sens_spec(p, yb, t)
        # TODO: binary AUC with boot_ci, four-class accuracy, tumor-only three-class accuracy
        print(f"{fam:<12}{name:<14}sens {sens:.3f}  spec {spec:.3f}")