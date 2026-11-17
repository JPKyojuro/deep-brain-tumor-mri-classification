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

print("Finding exact copies...")
md5s = [md5_of(r[2]) for r in recs]
print("Collecting hashes...")
hashes = np.array([phash64(r[2]) for r in recs], dtype=np.uint64)

rows = [] # one row per BDNeuro <-> [6] pair at Hamming distance <= 5
print("Finding near pairs...")
for i, j, d in near_pairs(hashes, 5):
    # i < j always; keep only i in BDNeuro and j in [6]
    if i >= nb or j < nb:
        continue
    rows.append({"ext": recs[j][2].name, "ext_folder": recs[j][0], "ext_cls": recs[j][1],
                 "bd": recs[i][2].name, "bd_split": recs[i][0], "bd_cls": recs[i][1],
                 "dist": d, "exact": int(md5s[i] == md5s[j]),
                 "ncc": round(ncc(recs[i][2], recs[j][2]), 3)})

out = Path(cfg["outputs_dir"]) / "setup" / "external_overlaps_pairs.csv"
with open(out, 'w', newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

matched = {r['ext'] for r in rows if r['exact'] or r['ncc'] >= LIKELY_DUP_NCC}
print(f"[6] images: {len(ex)}; with a BDNeuro match: {len(matched)}")

# TODO 1: matches by [6] folder and BDNeuro split (train/val/test)
# TODO 2: matched pairs whose class labels disagree
# TODO 3: unmatched [6] images per class and folder (the residual set)
# TODO 4: [6]-internal duplicate groups: span train and test? mixed labels?

