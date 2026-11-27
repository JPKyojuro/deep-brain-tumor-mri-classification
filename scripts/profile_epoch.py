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

