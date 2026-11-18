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

**External dataset [6]** (DOI 10.17632/gt5t2dycvp.1) goes in `data/external_brain_tumor_mri/` with
its original `train/` and `Test/` folders (class folders `glioma_tumor`, `meningioma_tumor`,
`no_tumor`, `pituitary_tumor`). It has no val folder, so use `list_external` from `common.py`, not
`list_dataset`.

## 3. Checks (run from this folder, in this order)

    python3 scripts/check_splits.py        # counts vs 4,160 / 892 / 889, unreadable files, image modes
    python3 scripts/dup_scan.py            # MD5 + perceptual hash + pixel correlation (ncc), within and across splits
    python3 scripts/view_pairs.py          # side-by-side images of near-duplicate pairs, for review by eye
    python3 scripts/class_balance.py       # class counts, class weights, tumor-positive counts per split
    python3 scripts/metadata_by_class.py   # image mode / size / colour vs class (shortcut risk)
    python3 scripts/build_manifest.py      # outputs/manifest.csv: metadata, whole-image and head-only intensity features, exclusions
    python3 scripts/view_masks.py          # outlines of the head mask on random images: look before trusting the masked rows
    python3 scripts/view_samples.py        # contact sheets of random raw no-tumor and tumor images: compare plane, sequence, borders
    python3 scripts/shortcut_baseline.py   # models that never see anatomy; scored on validation only
    python3 scripts/external_overlap.py    # BDNeuro vs [6]: exact and near-duplicate matches (run before any external scoring)

The baseline reports six feature families. `intensity` uses the whole image (including black
padding and background); `masked` and `masked_norm` use only pixels inside the head mask, computed
on the non-padded area so letterboxing cannot leak in; `geometry` is how much of the frame the head
fills. A big drop from `intensity` to `masked` means the whole-image histogram was mostly reading
padding and frame size. `masked_norm` stays high only if the signal is in the brain pixels themselves.

It prints three tables: four-class and binary results; triage (threshold for the 98% sensitivity
target chosen by 5-fold CV on train, applied once to validation, with an optimistic best-case
column); and a tumor-only three-class table (no-tumor removed) that shows whether format cues
separate the tumor types. Specificity rests on ~118 validation images, so treat it as noisy.

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

## What the external-dataset check found (dataset [6])
 
- 2,363 of [6]'s 3,264 images (72%) match a BDNeuro image (exact hash, or perceptual hash plus pixel
  correlation >= 0.98), across all three BDNeuro splits. So [6] is not an independent test set.
- The residual (no detected match) is 901 images: glioma 324, meningioma 199, no_tumor 98,
  pituitary 280. It is a sanity check for image-level overlap, not a patient-independent test, and its
  no_tumor count is small, so triage specificity on it needs a wide interval.
- [6] has its own leakage: 64 duplicate groups span its train and Test folders, and 392 file names
  are reused, so key every [6] record on the full path.
- Image size tracks the label in [6] as well (no_tumor is almost never 512x512).
- Provenance of [6] is not yet verified.

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
Use `--tumor-shape` to give meningioma and pituitary different shapes (the tumor-only table's
`metadata` row should reach 1.0). Use `--shape-only` to plant a shortcut that exists only in image shape: `metadata` and `intensity`
should reach AUC 1.0 while `geometry`, `masked` and `masked_norm` stay near 0.5.

## Open items (also marked TODO in config.yaml)

- Weight decay is not specified.
- Normalization for the from-scratch baseline (ImageNet mean/std vs dataset statistics).
- Concrete brightness/contrast and aspect/blur augmentation ranges.
- Which model and seed(s) the bootstrap CI covers.
- Two unconfirmed possible duplicate pairs at the bottom of `exclusions.txt`.
- External set: confirm the residual definition (no detected BDNeuro match, one copy per duplicate group), check the 0.90-0.98 NCC grey zone, and write `outputs/setup/external_residual.txt`
