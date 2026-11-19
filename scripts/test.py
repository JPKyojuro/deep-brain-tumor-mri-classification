"""Evaluate trained CNN checkpoints (from train.py) on the validation or test split.

Usage (from the project root):
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed0.pt                 # val: dry run
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed{0,1,2}.pt --split test --final

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

from common import load_config
from train import MRIDataset, SmallCNN, predict

NO_TUMOR = "no_tumor"
TARGET_SENS = 0.98
KEYS = ["acc", "f1", "sens", "spec"]

def fit_temperature():
    raise NotImplementedError

def ece():
    raise NotImplementedError

def scores():
    raise NotImplementedError

def evaluate():
    raise NotImplementedError

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Test split is sealed (D9). " \
        "Pass --final only for the final evaluation")

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []



if __name__ == "__main__":
    main()