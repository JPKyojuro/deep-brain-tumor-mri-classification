"""Shared helpers for the BDNeuro-MRI scripts."""
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
SPLITS = ("train", "val", "test")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_exclusions(cfg):
    """Set of image file names listed in cfg['data']['exclusions_file'].

    One name per line; '#' starts a comment. Missing file or key -> empty set.
    """
    path = cfg.get("data", {}).get("exclusions_file")
    if not path or not Path(path).exists():
        return set()
    names = set()
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                names.add(line)
    return names


def to_square_gray(img, size=224):
    """The one preprocessing function every stage should use.

    Grayscale first (removes colour/RGBA cues), pad non-square images to a
    square with black (keeps anatomy undistorted; MRI backgrounds are already
    black), then resize with a single filter so every image gets the same
    resampling treatment. Returns a PIL 'L' image of size x size.
    """
    img = img.convert("L")
    w, h = img.size
    if w != h:
        s = max(w, h)
        canvas = Image.new("L", (s, s), 0)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        img = canvas
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


def find_split_dirs(root):
    """Return {'train': Path, 'val': Path, 'test': Path}.

    Tolerates one or two wrapper folders left over from unzipping, and the
    names 'val' / 'valid' / 'validation'.
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root}")
    for _ in range(3):
        found = {
            SPLIT_ALIASES[p.name.lower()]: p
            for p in root.iterdir()
            if p.is_dir() and p.name.lower() in SPLIT_ALIASES
        }
        if found:
            missing = [s for s in SPLITS if s not in found]
            if missing:
                raise SystemExit(f"Found {sorted(found)} under {root} but missing {missing}")
            return found
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            break
        root = subdirs[0]
    raise SystemExit(
        f"No train/val/test folders found under {root}. "
        "Expected ImageFolder layout: <split>/<class>/<image>."
    )


def list_dataset(root):
    """Return a sorted list of (split, class_name, Path) for every image."""
    splits = find_split_dirs(root)
    records = []
    for split in SPLITS:
        for cls_dir in sorted(p for p in splits[split].iterdir() if p.is_dir()):
            for img in sorted(cls_dir.rglob("*")):
                if img.is_file() and img.suffix.lower() in IMG_EXTS:
                    records.append((split, cls_dir.name, img))
    if not records:
        raise SystemExit(f"No images found under {root}")
    return records
