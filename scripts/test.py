"""Evaluate trained CNN checkpoints (from train.py) on the validation or test split.

Usage (from the project root):
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed0.pt                 # val: dry run
    python3 scripts/test.py --ckpt outputs/checkpoints/cnn_seed{0,1,2}.pt --split test --final

The test split stays sealed (D9): it is only read with --split test --final.
For each checkpoint: temperature scaling and the 98%-sensitivity triage threshold are fit on
VALIDATION, then applied to the chosen split. With several checkpoints, the bootstrap resamples
images and averages the metric over checkpoints inside each resample (D14); the spread across
checkpoints is reported separately.
"""