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
    rows.append({"ext": recs[j][2].name, "ext_path": str(recs[j][2]), 
                 "ext_folder": recs[j][0], "ext_cls": recs[j][1],
                 "bd": recs[i][2].name, "bd_split": recs[i][0], "bd_cls": recs[i][1],
                 "dist": d, "exact": int(md5s[i] == md5s[j]),
                 "ncc": round(ncc(recs[i][2], recs[j][2]), 3)})

out = Path(cfg["outputs_dir"]) / "setup" / "external_overlaps_pairs.csv"
with open(out, 'w', newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

matched = {r['ext_path'] for r in rows if r['exact'] or r['ncc'] >= LIKELY_DUP_NCC}
print(f"[6] images: {len(ex)}; with a BDNeuro match: {len(matched)}")


import pandas as pd

df = pd.DataFrame(rows)
hits = df[(df['exact'] == 1) | (df['ncc'] >= LIKELY_DUP_NCC)]

# one row per [6] image; md5s[nb:] lines up with ex because recs = bd + ex
ex_df = pd.DataFrame({"path": [str(r[2]) for r in ex], 
                      "folder": [r[0] for r in ex],
                      "cls": [r[1] for r in ex], "md5": md5s[nb:]})

# matches by [6] folder and BDNeuro split (train/val/test)
print(pd.crosstab(hits['ext_folder'], hits['bd_split']))

# matched pairs whose labels disagree
bad = hits[hits['ext_cls'] != hits['bd_cls']]
print(len(bad), "of", len(hits), "matched pairs disagree on class")
print(pd.crosstab(bad['ext_cls'], bad['bd_cls'])) # which class pairs get confused

# unmatched [6] images per class and folder (the residual set)

ex_df['matched'] = ex_df['path'].isin(set(hits['ext_path']))
print(pd.crosstab([ex_df['folder'], ex_df['cls']], ex_df['matched']))


# [6]-internal duplicate groups
g = ex_df.groupby("md5").agg(n=("path", "size"), folders=("folder", "nunique"),
                             classes=("cls", "nunique"))

d = g[g['n'] > 1]
print(len(d), "duplicate groups;", (d["folders"] > 1).sum(), "span train and test;",
      (d["classes"] > 1).sum(), "have mixed labels")
print("images beyond the first copy:", (d["n"] - 1).sum())
print("file names reused:", ex_df["path"].map(lambda p: Path(p).name).duplicated().sum())


# Checking whether the overlaps come from the prior overlaps previously found
from common import load_exclusions
print(bad[["ext_path", "ext_cls", "bd", "bd_cls", "bd_split", "ncc"]])
print(bad["bd"].isin(load_exclusions(cfg)).sum(), "of", len(bad), "involve an excluded image")

# Finding how many images exist in the grey zone ncc (0.9-0.98)
# Are these duplicates exact copies, making a pool of copied images
# or are they just very, very similar

best = df.groupby("ext_path")["ncc"].max()
ex_df["best_ncc"] = ex_df["path"].map(best)
res = ex_df[~ex_df["matched"]]
print(res["best_ncc"].isna().sum(), "residual images with no hash-close BDNeuro image")
print(((res["best_ncc"] >= 0.9) & (res["best_ncc"] < 0.98)).sum(), "in the 0.90-0.98 grey zone")

# Matched images are already out, and no group has mixed labels, 
# so the only cleanup left is one copy per duplicate group

keep = res.drop_duplicates('md5')
output_path = Path(cfg['outputs_dir']) / "setup" / "external_residual.txt" 
keep['path'].to_csv(output_path, index=False, header=False)

print(len(keep), keep['cls'].value_counts().to_dict())

# 3 of the 4 disagreeing pairs involve an excluded BDNeuro image

# This isolates the 4th pair
K = bad[~bad["bd"].isin(load_exclusions(cfg))][["ext_path", "bd", "bd_split", "bd_cls", "ext_cls"]]
print(K.to_string())
strict = keep[keep['best_ncc'].isna() | (keep['best_ncc'] < 0.90)]
print(len(strict), strict['cls'].value_counts().to_dict())
strict['path'].to_csv(Path(cfg['outputs_dir']) / 'setup' / 'external_residual_strict.txt',
                      index=False, header=False)