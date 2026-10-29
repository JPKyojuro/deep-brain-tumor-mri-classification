"""Step 1: verify the downloaded dataset matches the published split counts.

Also confirms every image opens, and reports image modes and sizes so the
channel-replication / resize assumptions in the proposal can be checked.

Usage (from the project root):
    python scripts/check_splits.py [--config config.yaml]
Exit code 1 if any split count differs from config.data.expected_counts.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from common import SPLITS, list_dataset, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["data"]["root"]
    expected = cfg["data"]["expected_counts"]

    records = list_dataset(root)
    counts = Counter(split for split, _, _ in records)

    ok = True
    print(f"Dataset root: {root}")
    print(f"{'split':<6}{'found':>8}{'expected':>10}  status")
    for split in SPLITS:
        match = counts[split] == expected[split]
        ok &= match
        print(f"{split:<6}{counts[split]:>8}{expected[split]:>10}  {'OK' if match else 'MISMATCH'}")
    total, exp_total = sum(counts.values()), sum(expected.values())
    print(f"{'total':<6}{total:>8}{exp_total:>10}")

    modes, sizes, unreadable = Counter(), Counter(), []
    for _, _, path in records:
        try:
            with Image.open(path) as im:
                modes[im.mode] += 1
                sizes[im.size] += 1
                im.load()
        except Exception as e:  # corrupt or truncated file
            unreadable.append((str(path), repr(e)))

    print("\nImage modes:", dict(modes))
    print("Distinct image sizes:", len(sizes), "| most common:", sizes.most_common(3))
    if unreadable:
        ok = False
        print(f"\n{len(unreadable)} unreadable image(s), first 5:")
        for p, e in unreadable[:5]:
            print("  ", p, e)

    out = Path(cfg["outputs_dir"]) / "setup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_check.json", "w") as f:
        json.dump(
            {
                "counts": dict(counts),
                "expected": expected,
                "modes": dict(modes),
                "n_distinct_sizes": len(sizes),
                "top_sizes": [[list(s), n] for s, n in sizes.most_common(5)],
                "unreadable": unreadable,
                "passed": ok,
            },
            f,
            indent=2,
        )
    print("\nRESULT:", "PASS" if ok else "FAIL", f"(details: {out / 'split_check.json'})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
