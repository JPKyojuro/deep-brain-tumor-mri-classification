"""Time the pieces of a training epoch: image loading, the DataLoader, and the model step.

Usage (from the project root):
    python3 scripts/profile_epoch.py              # no D8
    python3 scripts/profile_epoch.py --d8 --workers 0
Takes about a minute. Compare the three epoch estimates to see which part dominates.
"""
import argparse
import os
import random
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from common import load_config, to_square_gray
from train import MRIDataset, SmallCNN

BATCH = 32
NORM = (0.1719, 0.1885)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d8", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    ds = MRIDataset(cfg, "train", size=224, train=True, d8=args.d8, norm=NORM)
    batches = len(ds) / BATCH
    print(f"{len(ds)} train images, {batches:.0f} batches/epoch, cpu cores={os.cpu_count()}, "
          f"torch threads={torch.get_num_threads()}")

    # 1. one image, in pieces: decode + letterbox + resize, then the whole __getitem__
    idx = random.sample(range(len(ds)), 50)
    t = time.perf_counter()
    for i in idx:
        to_square_gray(Image.open(ds.items[i][0]), 224)
    decode = (time.perf_counter() - t) / len(idx)
    t = time.perf_counter()
    for i in idx:
        ds[i]
    full = (time.perf_counter() - t) / len(idx)
    print(f"per image: decode+resize {decode * 1000:.1f} ms, full __getitem__ {full * 1000:.1f} ms "
          f"(augmentation+tensor {(full - decode) * 1000:.1f} ms)")

    # 2. the DataLoader on its own (no model)
    it = iter(DataLoader(ds, BATCH, shuffle=True, num_workers=args.workers))
    next(it)  # warm-up: worker start-up
    t = time.perf_counter()
    for _ in range(10):
        next(it)
    loader = (time.perf_counter() - t) / 10
    print(f"DataLoader (workers={args.workers}): {loader:.2f} s/batch -> {loader * batches / 60:.1f} min/epoch")

    # 3. forward + backward + optimiser step on one fixed random batch (no data loading)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(4).to(device)
    opt = torch.optim.AdamW(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(BATCH, 1, 224, 224, device=device)
    y = torch.randint(0, 4, (BATCH,), device=device)

    def step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()

    step()  # warm-up
    t = time.perf_counter()
    for _ in range(5):
        step()
    compute = (time.perf_counter() - t) / 5
    print(f"model step ({device}): {compute:.2f} s/batch -> {compute * batches / 60:.1f} min/epoch")


if __name__ == "__main__":
    main()
