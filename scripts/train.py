"""Train a small CNN on BDNeuro-MRI and save the best checkpoint.

Usage (from the project root):
    python3 scripts/train.py
    python3 scripts/train.py --lr 3e-4 --epochs 40 --width 48 --dropout 0.5

Any flag you leave out falls back to config.yaml (or the default below).
Model selection uses validation macro-F1; the test split is never touched here.
"""
import argparse
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


