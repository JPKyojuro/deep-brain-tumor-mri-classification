# BDNeuro-MRI brain tumor classification: setup stage

Covers the "Setup" stage of the proposal: download v9, duplicate scan, class balance, plus the
format and shortcut checks added after the first run. Training, evaluation and the demo come later
(`requirements-train.txt`).

## 1. Install

    pip install -r requirements.txt

## 2. Data

Download **BDNeuro-MRI v9** from Mendeley Data (DOI 10.17632/zwr4ntf94j.9,
https://data.mendeley.com/datasets/zwr4ntf94j/9). `config.yaml` expects:

    data/Brain_Tumor_MRI_Dataset_Final/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg

Pin v9: the SSRN abstract and the authors' GitHub repo still point to v7.

## 3. Checks (run from this folder, in this order)

    python3 scripts/check_splits.py        # counts vs 4,160 / 892 / 889, unreadable files, image modes
    python3 scripts/dup_scan.py            # MD5 + perceptual hash + pixel correlation (ncc), within and across splits
    python3 scripts/view_pairs.py          # side-by-side images of near-duplicate pairs, for review by eye
    python3 scripts/class_balance.py       # class counts, class weights, tumor-positive counts per split
    python3 scripts/metadata_by_class.py   # image mode / size / colour vs class (shortcut risk)
    python3 scripts/build_manifest.py      # outputs/manifest.csv: metadata + intensity features, applies exclusions.txt
    python3 scripts/shortcut_baseline.py   # models that never see anatomy; scored on validation only

Results go to `outputs/setup/` (and `outputs/manifest.csv`).

## What the first run found (BDNeuro-MRI v9)

- Counts match the published split. No exact (MD5) duplicates. Classes are stratified identically.
- The perceptual hash over-matches on MRI. Of 22 near-duplicate pairs at Hamming distance <= 5,
  four are the same image under two different labels (8 images, listed in `exclusions.txt`); the
  rest are different slices. Every pair had conflicting labels.
- Image format tracks the label: no_tumor images are rarely 512x512 or grayscale-mode, are mostly odd
  sizes, and include all RGBA and all truly-colour files. Several no_tumor examples in the pair review
  also looked like other modalities or sequences (CT, FLAIR, DWI, gradient-echo style), although the
  dataset page describes the set as T1-weighted contrast-enhanced. This is a visual impression, not a
  radiologist's reading.

## Decisions encoded in config.yaml (team to confirm)

- Exclude the 8 images in `exclusions.txt`; keep the authors' split otherwise.
- One preprocessing function for every stage (`common.to_square_gray`): grayscale first, letterbox
  non-square images to a square with black, one bicubic resize.
- Proposed augmentation to hide format cues: aspect jitter and random downscale-and-blur.
- Report test metrics by format subgroup (the manifest has the columns for it).

## Smoke test without the real data

    python3 scripts/make_synthetic.py /some/scratch/dir --per-class 60 --shortcut
    python3 scripts/build_manifest.py --config /some/scratch/dir/config.yaml
    python3 scripts/shortcut_baseline.py --config /some/scratch/dir/config.yaml

Drop `--shortcut` for a dataset with no planted shortcut (the baseline should hover near chance).

## Open items (also marked TODO in config.yaml)

- Weight decay is not specified.
- Normalization for the from-scratch baseline (ImageNet mean/std vs dataset statistics).
- Concrete brightness/contrast and aspect/blur augmentation ranges.
- Which model and seed(s) the bootstrap CI covers.
- Two unconfirmed possible duplicate pairs at the bottom of `exclusions.txt`.
