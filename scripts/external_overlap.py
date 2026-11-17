"""scripts/external_overlap.py: BDNeuro vs [6], exact (MD5) and near-duplicate (pHash + NCC)."""

import csv
from pathlib import Path

import numpy as np

from common import list_dataset, list_external, load_config
from dup_scan import LIKELY_DUP_NCC, md5_of, ncc, near_pairs, phash64

cfg = load_config()
bd = list_dataset(cfg["data"]["root"])                  # (split, cls, path)
ex = list_external("data/external_brain_tumor_mri")     # (folder, cls, path)
recs = bd + ex                                          # BDNeuro first, [6] after
nb = len(bd)

md5s = [md5_of(r[2]) for r in recs]
hashes = np.array([phash64(r[2]) for r in recs], dtype=np.uint64)
