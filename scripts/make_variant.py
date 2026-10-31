"""Generate a Kaggle notebook + metadata for one pipeline variant.

Each variant patches a few settings of src/lmsys_boosted_ensemble.py at run time, so
experiments never fork the pipeline itself.

Example:
    python scripts/make_variant.py variant-t090-tie106 --temperature 0.90 --tie-scale 1.06 \
        --out build --username my_kaggle_name
    kaggle kernels push -p build/variant-t090-tie106
"""
import argparse
import json
import os

DATASETS = [
    "{user}/lmsys-boosted-ensemble-script",
    "tascj0/lmsys-checkpoints-0-0805",
    "tascj0/lmsys-checkpoints-3-0805",
    "tascj0/lmsys-modules-0805",
    "tascj0/lmsys-packages",
]


def build_cell(a):
    lines = [
        "import glob",
        "src = open(glob.glob('/kaggle/input/**/lmsys_boosted_ensemble.py', recursive=True)[0]).read()",
        "def patch(old, new):",
        "    global src",
        "    assert old in src, old",
        "    src = src.replace(old, new)",
        "patch(" + repr('POOL = "arith"') + ", " + repr(f'POOL = "{a.pool}"') + ")",
    ]
    if a.gemma_weight is not None:
        w = {"gemma": a.gemma_weight, "llama": round(1 - a.gemma_weight, 4)}
        lines.append("patch(" + repr('WEIGHTS = {"gemma": 0.57, "llama": 0.43}') + ", " + repr(f"WEIGHTS = {w!r}") + ")")
    # Post-processing applied to the clipped probabilities just before normalisation.
    post = []
    if a.tie_scale != 1.0:
        post.append(f"p[:, 2] *= {a.tie_scale}")
    if a.temperature != 1.0:
        post.append(f"p = p ** (1 / {a.temperature})")
    if post:
        old = "sub[COLS] = p / p.sum(axis=1, keepdims=True)"
        lines.append("patch(" + repr(old) + ", " + repr("; ".join(post) + "; " + old) + ")")
    lines.append("exec(src)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--pool", default="geo", choices=["arith", "geo"])
    ap.add_argument("--temperature", type=float, default=1.0, help="<1 sharpens, >1 softens")
    ap.add_argument("--tie-scale", type=float, default=1.0, help="multiplier on the tie probability")
    ap.add_argument("--gemma-weight", type=float, default=None, help="blend weight of Gemma2 (Llama gets the rest)")
    ap.add_argument("--out", default="build")
    ap.add_argument("--username", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--docker-image", default=None, help="pinned Python 3.10 image (see README)")
    a = ap.parse_args()

    d = os.path.join(a.out, a.name)
    os.makedirs(d, exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "source": build_cell(a), "metadata": {"trusted": True},
                   "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.13"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    with open(os.path.join(d, "notebook.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    meta = {
        "id": f"{a.username}/lmsys-{a.name}",
        "title": "LMSYS " + a.name.replace("-", " ").title(),
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [s.format(user=a.username) for s in DATASETS],
        "kernel_sources": [], "competition_sources": ["llm-classification-finetuning"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    if a.docker_image:
        meta["docker_image"] = a.docker_image
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
