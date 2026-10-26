"""Score the validation probabilities saved by kernels/validation-probe.

Usage: python scripts/analyze_validation.py <dir with val_*.npy files>

Note: the tascj0 checkpoints were trained on train.csv, so these numbers are optimistic.
Use them to catch bugs, not to tune blend weights.
"""
import sys

import numpy as np

d = sys.argv[1]
y = np.load(f"{d}/val_labels.npy")


def logloss(p):
    p = np.clip(p / p.sum(1, keepdims=True), 1e-15, 1)
    return float(-(y * np.log(p)).sum(1).mean())


probs = {k: np.load(f"{d}/val_{k}.npy") for k in ("gemma_ab", "gemma_ba", "llama_ab", "llama_ba")}
for name, p in probs.items():
    print(f"{name:10s} log loss {logloss(p):.4f}")
for m in ("gemma", "llama"):
    print(f"{m} + swap-TTA log loss {logloss((probs[m + '_ab'] + probs[m + '_ba']) / 2):.4f}")
