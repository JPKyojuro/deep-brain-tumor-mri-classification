"""scripts/gray_pairs.py: sample gray-zone pairs for view_pairs.py."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common import list_dataset, load_config

cfg = load_config()
out = Path(cfg['outputs_dir']) / 'setup'
pairs = pd.read_csv(out / "external_overlaps_pairs.csv")
keep = set((out / "external_residual.txt").read_text().splitlines())

# best BDNeuro match per [6] image
best = pairs.loc[pairs.groupby("ext_path")['ncc'].idxmax()]
gray = best[best['ext_path'].isin(keep) & (best['ncc'] >= 0.90) & (best['ncc'] < 0.98)]
gray = gray.sort_values('ncc')
print(len(gray), "gray-zone images")            # expect about 266
pick = gray.iloc[np.linspace(0, len(gray) - 1, 12).astype(int)]

bd_path = {p.name: p for _, _, p in list_dataset(cfg['data']['root'])}
with open(out / "gray_pairs.csv", 'w', newline="") as f:
    w = csv.writer(f)
    w.writerow(["distance", "split_a", "class_a", "file_a",
                "split_b", "class_b", "file_b", "kind", "ncc"])
    for r in pick.itertuples():
        w.writerow([r.dist, r.bd_split, r.bd_cls, bd_path[r.bd],
                    r.ext_folder, r.ext_cls, r.ext_path, "gray_zone", r.ncc])
