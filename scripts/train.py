"""Train a small CNN on BDNeuro-MRI and save the best checkpoint.

Usage (from the project root):
    python3 scripts/train.py                      # seed 0, no D8 augmentation
    python3 scripts/train.py --d8                 # same, plus the D8 format-cue augmentation
    python3 scripts/train.py --seed 1 --lr 3e-4 --width 48

Any flag you leave out falls back to config.yaml (or the default below).
Model selection uses validation macro-F1; the test split is never touched here.
Besides the checkpoint, the best epoch's per-image validation logits are saved
(<out>_val_preds.csv) so D14/D15 and error comparisons with the baseline can be done later.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import list_dataset, load_config, load_exclusions, to_square_gray

# D8 ranges: TODO, tune on validation only. These are placeholders
ASPECT_RATIO = (0.8, 1.25)      # aspect jitter (RandomResizedCrop ratio)
CROP_SCALE = (0.85, 1.0)        # fraction of the image kept by that crop
DOWNSCALE = (0.4, 1.0)          # random downscale factor before resizing back up
BLUR_SIGMA = (0.1, 1.5)         # Gaussian blur strength

class RandomDownscale:
    # Shrink by a random factor, then resize back: makes sharpness/resolution a weaker cue.

    def __init__(self, size, factor):
        self.size, self.factor = size, factor

    def __call__(self, img):
        f = random.uniform(*self.factor)
        small = max(8, int(self.size * f))
        return img.resize((small, small), 
                          Image.BICUBIC).resize((self.size, self.size), Image.BICUBIC)


class MRIDataset(Dataset):

    def __init__(self, cfg, split, classes=None, size=224, train=False):
        skip = load_exclusions(cfg)
        recs = [r for r in list_dataset(cfg["data"]["root"])
                if r[0] == split and r[2].name not in skip]
        self.classes = classes or sorted({c for _, c, _ in recs})
        self.items = [(p, self.classes.index(c)) for _, c, p in recs]
        self.size = size
        steps = []
        if train:  # TODO: try the aspect-jitter / downscale-blur ideas from config.yaml
            steps += [T.RandomRotation(cfg["training"]["augmentation"]["rotation_deg"]),
                      T.RandomHorizontalFlip()]
        steps += [T.ToTensor(), T.Normalize(0.5, 0.5)]  # TODO: dataset mean/std instead?
        self.tf = T.Compose(steps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = to_square_gray(Image.open(path), self.size)
        return self.tf(img), label


class SmallCNN(nn.Module):
    """4 conv blocks -> global average pool -> linear. Input: 1 x size x size."""

    def __init__(self, n_classes, width=32, dropout=0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, width), block(width, width * 2),
                                      block(width * 2, width * 4), block(width * 4, width * 8))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(width * 8, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, predicted labels) as numpy arrays."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    cfg = load_config(pre.parse_known_args()[0].config)
    tr = cfg["training"]

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--lr", type=float, default=tr["baseline"]["lr"])
    ap.add_argument("--epochs", type=int, default=tr["baseline"]["max_epochs"])
    ap.add_argument("--batch-size", type=int, default=tr["batch_size"])
    ap.add_argument("--weight-decay", type=float, default=tr["weight_decay"] or 1e-4)
    ap.add_argument("--patience", type=int, default=tr["early_stopping"]["patience"])
    ap.add_argument("--input-size", type=int, default=tr["input_size"])
    ap.add_argument("--width", type=int, default=32, help="channels in the first conv block")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=cfg["seeds"][0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(Path(cfg["outputs_dir"]) / "checkpoints" / "cnn.pt"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MRIDataset(cfg, "train", size=args.input_size, train=True)
    val_ds = MRIDataset(cfg, "val", classes=train_ds.classes, size=args.input_size)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    # class-weighted loss only when train is imbalanced (threshold from config.yaml)
    counts = np.bincount([y for _, y in train_ds.items], minlength=len(train_ds.classes))
    weight = None
    if counts.max() / counts.min() > cfg["data"]["imbalance_ratio_threshold"]:
        weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    model = SmallCNN(len(train_ds.classes), args.width, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, bad = -1.0, 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(train_ds)} train / {len(val_ds)} val images, device={device}, classes={train_ds.classes}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()

        y_true, y_pred = predict(model, val_dl, device)
        f1 = f1_score(y_true, y_pred, average="macro")
        acc = (y_true == y_pred).mean()
        print(f"epoch {epoch:2d}  train loss {total / len(train_ds):.4f}  val acc {acc:.3f}  val macro-F1 {f1:.3f}")

        if f1 > best:
            best, bad = f1, 0
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes,
                        "args": vars(args), "epoch": epoch, "val_macro_f1": float(f1)}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no val macro-F1 gain for {args.patience} epochs)")
                break

    print(f"best val macro-F1 {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()