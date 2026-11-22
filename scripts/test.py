"""Evaluate trained CNN checkpoints (from train.py) on the validation or test split.

Usage (from the project root):
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed0.pt                 # val: dry run
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed{0,1,2}.pt --split test --final
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed0.pt --external external_residual_strict.txt

The test split stays sealed (D9): it is only read with --split test --final.
For each checkpoint: temperature scaling and the 98%-sensitivity triage threshold are fit on
VALIDATION, then applied to the chosen split. With several checkpoints, the bootstrap resamples
images and averages the metric over checkpoints inside each resample (D14); the spread across
checkpoints is reported separately.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from common import EXTERNAL_CLASS_MAP, load_config
from train import MRIDataset, SmallCNN, predict

NO_TUMOR = "no_tumor"
TARGET_SENS = 0.98
KEYS = ["acc", "f1", "sens", "spec"]


def fit_temperature(logits, y):
    # Single temperature T minimising NLL on (validation) logits.
    z, y = torch.tensor(logits), torch.tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def ece(p, event, bins=10):
    # Expected calibration error:
    # p = predicted probability
    # event = 0/1 outcome
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(event[m].mean() - p[m].mean())
    return total


def scores(y, pred, score, thr, nt):
    # acc, macro-F1, and triage sensitivity/specificity
    # (flag 'tumor' if score >= thr)
    pos, flag = y != nt, score >= thr
    return np.array([(y == pred).mean(), f1_score(y, pred, average="macro"), 
                     flag[pos].mean(), (~flag[~pos]).mean()])


def ext_items(list_file, classes):
    # (Path, label) per line of an external list file: full paths, class = parent folder name.
    p = Path(list_file)
    if not p.exists():
        p = Path("outputs/setup") / p  # where external_overlap.py writes the lists
    items = []
    for line in p.read_text().splitlines():
        if line.strip():
            path = Path(line.strip().replace("\\", "/"))
            items.append((path, classes.index(EXTERNAL_CLASS_MAP[path.parent.name])))
    return items


def evaluate(cfg, ckpt_path, split, device, workers, ext=None):
    ck = torch.load(ckpt_path, map_location=device)
    a, classes = ck["args"], ck["classes"]
    model = SmallCNN(len(classes), a["width"], a["dropout"]).to(device)
    model.load_state_dict(ck["model_state"])

    out = {}
    for s in {"val", split}:  # val is always needed for calibration
        ds = MRIDataset(cfg, "val" if s == "external" else s, classes=classes,
                        size=a["input_size"], norm=ck["norm"])
        if s == "external":  # same preprocessing as val, different file list
            ds.items = ext_items(ext, classes)
        out[s] = (ds, *predict(model, DataLoader(ds, a["batch_size"], num_workers=workers), device))

    nt = classes.index(NO_TUMOR)
    _, yv, zv = out["val"]
    T = fit_temperature(zv, yv)
    sv = 1 - F.softmax(torch.tensor(zv) / T, dim=1).numpy()[:, nt]   # P(tumor) on val
    thr = float(np.quantile(sv[yv != nt], 1 - TARGET_SENS))          # threshold on val

    ds, y, z = out[split]
    prob = F.softmax(torch.tensor(z) / T, dim=1).numpy()
    return dict(ds=ds, classes=classes, nt=nt, y=y, prob=prob, pred=prob.argmax(1),
                score=1 - prob[:, nt], T=T, thr=thr, epoch=ck["epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--external", help="list file of dataset [6] paths (e.g. external_residual_strict.txt)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.external:
        args.split = "external"
    tag = Path(args.external).stem if args.external else args.split
    if args.split == "test" and not args.final:
        raise SystemExit("Test split is sealed (D9). Pass --final only for the final evaluation.")

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for path in args.ckpt:
        R = evaluate(cfg, path, args.split, device, args.workers, args.external)
        y, nt = R["y"], R["nt"]
        s = scores(y, R["pred"], R["score"], R["thr"], nt)
        conf = R["prob"].max(1)
        print(f"\n{path}  (best epoch {R['epoch']}, T={R['T']:.2f}, triage threshold={R['thr']:.3f})")
        print("  " + "  ".join(f"{k} {v:.3f}" for k, v in zip(KEYS, s)))
        print(f"  binary AUC {roc_auc_score(y != nt, R['score']):.3f}"
              f"  ECE 4-class {ece(conf, (R['pred'] == y).astype(float)):.3f}"
              f"  ECE binary {ece(R['score'], (y != nt).astype(float)):.3f}")
        print(f"  confusion (rows = true, cols = pred; {R['classes']}):")
        print(confusion_matrix(y, R["pred"]))

        out_csv = Path(cfg["outputs_dir"]) / "eval" / f"{tag}_{Path(path).stem}_preds.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "pred", "p_tumor"] + [f"p_{c}" for c in R["classes"]])
            for (p, _), t, pr, sc, row in zip(R["ds"].items, y, R["pred"], R["score"], R["prob"]):
                w.writerow([str(p), R["classes"][t], R["classes"][pr], f"{sc:.5f}"]
                           + [f"{v:.5f}" for v in row])
        runs.append(R)

    # D14: resample images, average over checkpoints inside each resample
    y, nt, n = runs[0]["y"], runs[0]["nt"], len(runs[0]["y"])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.boot):
        i = rng.integers(0, n, n)
        boot.append(np.mean([scores(y[i], R["pred"][i], R["score"][i], R["thr"], nt)
                             for R in runs], axis=0))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    per_run = np.array([scores(y, R["pred"], R["score"], R["thr"], nt) for R in runs])
    print(f"\n{tag} summary over {len(runs)} checkpoint(s), {n} images ({(y == nt).sum()} no_tumor), "
          "95% bootstrap CI over images:")
    for k, name in enumerate(KEYS):
        print(f"  {name:5s} {per_run[:, k].mean():.3f}  [{lo[k]:.3f}, {hi[k]:.3f}]"
              f"  seed SD {per_run[:, k].std(ddof=1) if len(runs) > 1 else float('nan'):.3f}")


if __name__ == "__main__":
    main()