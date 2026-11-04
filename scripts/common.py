"""Shared helpers for the BDNeuro-MRI scripts."""
from pathlib import Path

import numpy as np
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


HEAD_MIN_THRESHOLD = 8.0  # gray levels (0-255); floor for the head/background threshold


def otsu_threshold(arr):
    """Otsu threshold of a 0-255 grayscale array (returns a gray level)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.where(valid, m0 / np.maximum(w0, 1), 0.0)
    mu1 = np.where(valid, (mt - m0) / np.maximum(w1, 1), 0.0)
    between = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
    return float(between.argmax())


def valid_box(wh, size=224):
    """(x0, y0, x1, y1) of the non-padding area inside the letterboxed size x size image.

    Mirrors to_square_gray. Edges next to padding are shrunk by 1 pixel to skip the
    blended boundary. Pass it to head_mask so padding cannot influence the mask
    threshold (Otsu depends on how much black is in the frame).
    """
    w, h = wh
    if w == h:
        return (0, 0, size, size)
    s = max(w, h)
    sc = size / s
    x0, y0, x1, y1 = 0, 0, size, size
    if w < h:
        ox = (s - w) // 2
        x0 = int(np.ceil(ox * sc)) + 1
        x1 = int(np.floor((ox + w) * sc)) - 1
    else:
        oy = (s - h) // 2
        y0 = int(np.ceil(oy * sc)) + 1
        y1 = int(np.floor((oy + h) * sc)) - 1
    return (x0, y0, x1, y1)


def head_mask(arr, valid=None):
    """Boolean mask of the head region of a 0-255 grayscale array.

    `valid` = (x0, y0, x1, y1) restricts the computation to the non-padded area
    (see valid_box); pixels outside it are always False.

    Threshold at max(8, 0.5 * Otsu) so black padding and dark background are
    left out while dark tissue inside the head is kept; keep the largest
    connected component and fill holes (ventricles, sinuses). An approximation:
    it is not a brain segmentation, and noisy backgrounds can leak into it.
    Falls back to all valid pixels if nothing passes the threshold.
    """
    from scipy import ndimage

    full = np.zeros(arr.shape, dtype=bool)
    x0, y0, x1, y1 = valid if valid is not None else (0, 0, arr.shape[1], arr.shape[0])
    sub = arr[y0:y1, x0:x1]
    t = max(HEAD_MIN_THRESHOLD, 0.5 * otsu_threshold(sub))
    fg = sub > t
    labels, n = ndimage.label(fg)
    if n == 0:
        full[y0:y1, x0:x1] = True
        return full
    sizes = ndimage.sum(fg, labels, index=np.arange(1, n + 1))
    full[y0:y1, x0:x1] = ndimage.binary_fill_holes(labels == (1 + int(np.argmax(sizes))))
    return full


def crop_to_head(img, size=224, margin=0.03):
    """to_square_gray applied after cropping the original image to the head box.

    Removes most of the dark border (a field-of-view cue). Not used by the
    baseline yet; meant for the training pipeline once the team decides.
    """
    g = img.convert("L")
    mask = head_mask(np.asarray(g, dtype=np.float32))
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return to_square_gray(g, size)
    h, w = mask.shape
    my, mx = int(margin * h), int(margin * w)
    box = (max(int(xs.min()) - mx, 0), max(int(ys.min()) - my, 0),
           min(int(xs.max()) + 1 + mx, w), min(int(ys.max()) + 1 + my, h))
    return to_square_gray(g.crop(box), size)


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
