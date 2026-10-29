"""Step 2: duplicate scan within and across splits.

Finds (a) exact duplicates by MD5 and (b) near-duplicates by 64-bit perceptual
hash, reported at each Hamming-distance threshold in config.data.dup_thresholds.
Also flags near-duplicate pairs whose class labels disagree.

The perceptual hash over-matches on MRI: many different slices with similar
framing land within a few bits of each other. So for pairs at Hamming distance
<= 5 we also compute a pixel correlation (NCC, on 64x64 grayscale). Real
duplicates score ~0.98-1.0; different slices score far lower. The "ncc" column
in the CSV separates the two. Look at the images before excluding anything
(scripts/view_pairs.py) and record decisions in exclusions.txt.

Usage (from the project root):
    python3 scripts/dup_scan.py [--config config.yaml]
Exit code 1 if any exact (MD5) duplicate crosses splits.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from common import list_dataset, load_config, to_square_gray

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
NCC_MAX_DIST = 5       # compute pixel correlation only for pairs this close
LIKELY_DUP_NCC = 0.98  # NCC at or above this: very likely the same image


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash64(path):
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("L"), hash_size=8)
    bits = "".join("1" if b else "0" for b in h.hash.flatten())
    return np.uint64(int(bits, 2))


def ncc(path_a, path_b, size=64):
    """Normalized cross-correlation of two images on a size x size grayscale grid."""
    arrs = []
    for p in (path_a, path_b):
        with Image.open(p) as im:
            a = np.asarray(to_square_gray(im, size), dtype=np.float32).ravel()
        s = a.std()
        arrs.append(None if s == 0 else (a - a.mean()) / s)
    if arrs[0] is None or arrs[1] is None:
        return 0.0
    return float((arrs[0] * arrs[1]).mean())


def near_pairs(hashes, max_dist, block=256):
    """Yield (i, j, dist) for all i < j with Hamming distance <= max_dist."""
    n = len(hashes)
    cols = np.arange(n)[None, :]
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        xor = hashes[i0:i1, None] ^ hashes[None, :]
        d = POPCOUNT[xor.view(np.uint8).reshape(i1 - i0, n, 8)].sum(axis=2)
        rows = (i0 + np.arange(i1 - i0))[:, None]
        d[cols <= rows] = 255  # keep only the upper triangle
        for r, c in np.argwhere(d <= max_dist):
            yield i0 + int(r), int(c), int(d[r, c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    thresholds = sorted(cfg["data"]["dup_thresholds"])

    records = list_dataset(cfg["data"]["root"])
    n = len(records)
    print(f"Hashing {n} images ...")
    md5s = [md5_of(p) for _, _, p in records]
    hashes = np.array([phash64(p) for _, _, p in records], dtype=np.uint64)

    # ---- exact duplicates
    groups = defaultdict(list)
    for idx, m in enumerate(md5s):
        groups[m].append(idx)
    exact_groups = [g for g in groups.values() if len(g) > 1]
    exact_cross = [g for g in exact_groups if len({records[i][0] for i in g}) > 1]

    # ---- near duplicates
    pairs = list(near_pairs(hashes, thresholds[-1]))
    summary = {"n_images": n, "exact_duplicate_groups": len(exact_groups),
               "exact_cross_split_groups": len(exact_cross), "by_threshold": {}}
    for t in thresholds:
        within, cross, conflict = Counter(), Counter(), 0
        for i, j, d in pairs:
            if d > t:
                continue
            si, sj = records[i][0], records[j][0]
            if si == sj:
                within[si] += 1
            else:
                cross["-".join(sorted((si, sj)))] += 1
            if records[i][1] != records[j][1]:
                conflict += 1
        summary["by_threshold"][str(t)] = {
            "within_split_pairs": dict(within),
            "cross_split_pairs": dict(cross),
            "cross_split_total": sum(cross.values()),
            "label_conflict_pairs": conflict,
        }

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    likely = []  # (ncc, record_a, record_b, distance)
    with open(out / "dup_scan_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "split_a", "class_a", "file_a",
                    "split_b", "class_b", "file_b", "kind", "ncc"])
        for g in exact_cross:
            for k in range(1, len(g)):
                a, b = records[g[0]], records[g[k]]
                w.writerow([0, a[0], a[1], a[2], b[0], b[1], b[2], "exact_md5_cross_split", "1.000"])
                likely.append((1.0, a, b, 0))
        for i, j, d in sorted(pairs, key=lambda x: x[2]):
            a, b = records[i], records[j]
            if md5s[i] == md5s[j] and a[0] != b[0]:
                continue  # already written above as an exact cross-split duplicate
            if a[0] != b[0] or a[1] != b[1]:  # cross-split or label conflict
                kind = ("cross_split" if a[0] != b[0] else "") + (
                    "+label_conflict" if a[1] != b[1] else "")
                score = ""
                if d <= NCC_MAX_DIST:
                    v = ncc(a[2], b[2])
                    score = f"{v:.3f}"
                    if v >= LIKELY_DUP_NCC:
                        likely.append((v, a, b, d))
                w.writerow([d, a[0], a[1], a[2], b[0], b[1], b[2], kind.lstrip("+"), score])
    summary["likely_duplicate_pairs"] = len(likely)
    with open(out / "dup_scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExact (MD5) duplicate groups: {len(exact_groups)} "
          f"({len(exact_cross)} cross-split)")
    print(f"{'max Hamming':>12}{'cross-split':>13}{'within-split':>14}{'label conflicts':>17}")
    for t in thresholds:
        s = summary["by_threshold"][str(t)]
        print(f"{t:>12}{s['cross_split_total']:>13}"
              f"{sum(s['within_split_pairs'].values()):>14}{s['label_conflict_pairs']:>17}")

    print(f"\nPairs with pixel correlation >= {LIKELY_DUP_NCC} (very likely the same image): "
          f"{len(likely)}")
    for v, a, b, d in sorted(likely, key=lambda x: (-x[0], x[3])):
        flag = "" if a[1] == b[1] else "  LABEL CONFLICT"
        print(f"  ncc {v:.3f} dist {d}: {a[0]}/{a[1]}/{a[2].name}  <->  "
              f"{b[0]}/{b[1]}/{b[2].name}{flag}")
    print(f"\nPairs written to {out / 'dup_scan_pairs.csv'}")
    raise SystemExit(1 if exact_cross else 0)


if __name__ == "__main__":
    main()
