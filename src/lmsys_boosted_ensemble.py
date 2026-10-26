"""LMSYS / LLM Classification Finetuning: Gemma2-9B + Llama3-8B ensemble with swap-TTA, a
guaranteed-complete submission.csv, and a diagnostics dashboard.

Run on Kaggle: Python 3.10 environment, accelerator GPU T4 x2, internet off. Attach:
  competition data, tascj0/lmsys-packages, tascj0/lmsys-modules-0805,
  tascj0/lmsys-checkpoints-0-0805 (Gemma2-9B), tascj0/lmsys-checkpoints-3-0805 (Llama3-8B).
Paste into one notebook cell (or `%run` / `!python` it).

Outputs (in /kaggle/working):
  submission.csv          always complete and validated (see "submission safety" below)
  prob_<model>_<dir>.npy  raw probabilities of every finished pass
  run_summary.json        passes, timings, weights
  charts/dashboard.png    9-panel diagnostics of the test predictions
  charts/validation.png   only when VALIDATE_N > 0 (labelled train rows; optimistic, see below)

Submission safety
  * A valid prior-only submission.csv is written before any GPU work, so the file always exists.
  * Every pass / model load is wrapped: a failure is logged and the run continues with what finished.
  * The file is written atomically, checked against sample_submission.csv (ids, columns, no NaN,
    rows sum to 1) and any missing id is filled with the class prior.
  * If no model pass finished, the script raises at the end instead of silently submitting priors.

Time safety
  The two baseline passes (Gemma2 A/B, Llama3 B/A) always run. Each extra TTA pass runs only if the
  measured pass time says it still fits before BUDGET_S, so the same file works on the 3-row sample
  and on the ~25k-row rerun.
"""
import gc
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

T0 = time.time()

# ------------------------------- settings -------------------------------
BUDGET_S = 8.0 * 3600            # stop starting optional passes after this (Kaggle limit is 9h)
WEIGHTS = {"gemma": 0.57, "llama": 0.43}   # weights tascj0's pipeline was tuned with
POOL = "arith"                   # "arith" = weighted mean of probabilities, "geo" = weighted geometric mean
LLAMA_TO_GEMMA_TIME = 0.9        # assumed cost of a Llama3-8B pass relative to a Gemma2-9B pass
PROB_FLOOR = 1e-4                # smallest probability written to the CSV
NUM_WORKERS = 0                  # >0 forks loader workers; in this notebook JAX warns that a fork can deadlock, so keep 0
MAKE_CHARTS = True
VALIDATE_N = 0                   # >0: also score this many labelled train rows for the validation chart.
                                 # The checkpoints were probably trained on train.csv, so those numbers are
                                 # optimistic: use them to spot bugs, not to tune weights. Adds a few minutes.
# Off by default: not validated against clean held-out data, so treat as experiments.
SMART_TRUNCATION = False         # head+tail truncation of prompt / responses instead of the stock processor
LENGTH_PRIOR = 0.0               # e.g. 0.03 nudges towards the longer response

os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = "/kaggle/working"
COLS = ["winner_model_a", "winner_model_b", "winner_tie"]


def log(msg):
    print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)


# ------------------------------- environment ----------------------------
if sys.version_info[:2] != (3, 10):
    raise RuntimeError(
        f"Python {sys.version.split()[0]}: the triton/xformers wheels are cp310 builds. "
        "Use a notebook created from an older (Python 3.10) Kaggle environment."
    )


def find_dir(*names, max_depth=4):
    """Directory named one of `names` under /kaggle/input (mount layout differs between sessions)."""
    base = "/kaggle/input".count("/")
    for root, dirs, _ in os.walk("/kaggle/input"):
        if os.path.basename(root) in names:
            return root
        if root.count("/") - base >= max_depth:
            dirs[:] = []
    return None


def find_file(name, prefer=""):
    hits = [os.path.join(r, name) for r, _, fs in os.walk("/kaggle/input") if name in fs]
    hits.sort(key=lambda h: prefer not in h)
    return hits[0] if hits else None


PKG_DIR = find_dir("lmsys-packages")
MOD_SRC = find_dir("lmsys-modules-0805", "lmsys-module-0805")
CK = {"gemma": find_dir("lmsys-checkpoints-0-0805"), "llama": find_dir("lmsys-checkpoints-3-0805")}
TEST_CSV = find_file("test.csv", prefer="llm-classification")
SUB_CSV = find_file("sample_submission.csv", prefer="llm-classification")
TRAIN_CSV = find_file("train.csv", prefer="llm-classification")
for label, path in [("lmsys-packages", PKG_DIR), ("lmsys-modules-0805", MOD_SRC),
                    ("lmsys-checkpoints-0-0805", CK["gemma"]), ("lmsys-checkpoints-3-0805", CK["llama"]),
                    ("test.csv", TEST_CSV), ("sample_submission.csv", SUB_CSV)]:
    assert path, f"Missing input: {label} - attach it via Add Input"

# ------------------------------- wheels + modules -----------------------
for pattern in ("triton-*-cp310-*.whl", "xformers-*-cp310-*.whl"):
    whl = sorted(glob.glob(f"{PKG_DIR}/{pattern}"))[0]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "--no-index", "--no-deps", "--force-reinstall", whl])
importlib.invalidate_caches()

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import torch                                           # noqa: E402
from torch.utils.data import DataLoader                # noqa: E402
from transformers import AutoTokenizer                 # noqa: E402
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask  # noqa: E402

assert torch.cuda.device_count() >= 2, "Need 2 GPUs: set the accelerator to GPU T4 x2"

DST = f"{WORK}/human_pref"
shutil.rmtree(DST, ignore_errors=True)
shutil.copytree(MOD_SRC, DST)
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from human_pref.data.collators import ShardedMaxTokensCollator, VarlenCollator   # noqa: E402
from human_pref.data.dataset import LMSYSDataset                                  # noqa: E402
from human_pref.data.processors import ProcessorPAB                               # noqa: E402
from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification     # noqa: E402
from human_pref.models.modeling_llama import LlamaForSequenceClassification       # noqa: E402
from human_pref.utils import to_device                                            # noqa: E402

log(f"env ok | torch {torch.__version__} | gpus {torch.cuda.device_count()}")

# ------------------------------- data views -----------------------------
df_test = pd.read_csv(TEST_CSV)
N = len(df_test)


def write_views(df, ab_path, ba_path):
    df.to_parquet(ab_path, index=False)
    sw = df.copy()
    sw["response_a"], sw["response_b"] = df["response_b"], df["response_a"]
    sw.to_parquet(ba_path, index=False)


view = df_test.copy()
view["winner_model_a"], view["winner_model_b"], view["winner_tie"] = 1, 0, 0     # dummy labels
VIEWS = {"test": {"ab": f"{WORK}/test.parquet", "ba": f"{WORK}/test_swap.parquet"}}
write_views(view, VIEWS["test"]["ab"], VIEWS["test"]["ba"])

val_labels = None
if VALIDATE_N and TRAIN_CSV:
    tr = pd.read_csv(TRAIN_CSV).sample(n=VALIDATE_N, random_state=0).reset_index(drop=True)
    val_labels = tr[COLS].values.astype(np.float64)
    VIEWS["val"] = {"ab": f"{WORK}/val.parquet", "ba": f"{WORK}/val_swap.parquet"}
    write_views(tr, VIEWS["val"]["ab"], VIEWS["val"]["ba"])
log(f"test rows: {N}" + (f" | validation rows: {len(val_labels)}" if val_labels is not None else ""))

# ------------------------------- submission safety ----------------------
if TRAIN_CSV:
    PRIOR = pd.read_csv(TRAIN_CSV, usecols=COLS).mean().values.astype(np.float64)
else:
    PRIOR = np.full(3, 1 / 3)
PRIOR = PRIOR / PRIOR.sum()


def save_submission(prob, tag):
    ref = pd.read_csv(SUB_CSV)[["id"]]
    frame = pd.DataFrame({"id": df_test["id"].values, COLS[0]: prob[:, 0], COLS[1]: prob[:, 1], COLS[2]: prob[:, 2]})
    sub = ref.merge(frame, on="id", how="left")
    for j, c in enumerate(COLS):
        sub[c] = sub[c].fillna(PRIOR[j])              # any id the models did not cover
    p = np.clip(sub[COLS].values.astype(np.float64), PROB_FLOOR, None)
    sub[COLS] = p / p.sum(axis=1, keepdims=True)

    assert list(sub.columns) == ["id"] + COLS, sub.columns
    assert len(sub) == len(ref) and sub["id"].is_unique, "id mismatch"
    assert np.isfinite(sub[COLS].values).all(), "non-finite probabilities"
    assert np.allclose(sub[COLS].sum(axis=1), 1.0, atol=1e-6), "rows not normalised"
    tmp = f"{WORK}/submission.csv.tmp"
    sub.to_csv(tmp, index=False)
    os.replace(tmp, f"{WORK}/submission.csv")          # atomic: never a half-written file
    log(f"submission.csv written ({tag}) | {len(sub)} rows")
    return sub


save_submission(np.tile(PRIOR, (N, 1)), "prior-only safety net")


# ------------------------------- processors -----------------------------
class SmartProcessorPAB(ProcessorPAB):
    """Head+tail truncation per field so both the start and the end of long texts survive."""
    HEAD_FRACTION = 0.60
    PROMPT_SHARE = 0.35

    def _truncate(self, text, budget):
        text = "" if text is None else str(text)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        head = int(budget * self.HEAD_FRACTION)
        tail = budget - head
        ids = ids[:head] + (ids[-tail:] if tail > 0 else [])
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def __call__(self, prompt, response_a, response_b):
        p_budget = int(self.max_length * self.PROMPT_SHARE)
        r_budget = (self.max_length - p_budget) // 2
        return super().__call__(self._truncate(prompt, p_budget),
                                self._truncate(response_a, r_budget),
                                self._truncate(response_b, r_budget))


def make_processor(kind):
    tok = AutoTokenizer.from_pretrained(CK[kind], local_files_only=True)
    if kind == "llama":
        tok.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    cls = SmartProcessorPAB if SMART_TRUNCATION else ProcessorPAB
    return cls(tokenizer=tok, max_length=4096, support_system_role=(kind == "llama"))


# ------------------------------- inference ------------------------------
SPEC = {
    "gemma": dict(cls=Gemma2ForSequenceClassification, layers=42),
    "llama": dict(cls=LlamaForSequenceClassification, layers=32),
}
events = []          # (label, kind, start_s, end_s) for the timeline chart


def device_map(n):
    dm = {"model.embed_tokens": "cuda:0", "model.norm": "cuda:1", "score": "cuda:1"}
    for i in range(n // 2):
        dm[f"model.layers.{i}"] = "cuda:0"
    for i in range(n // 2, n):
        dm[f"model.layers.{i}"] = "cuda:1"
    return dm


def load_model(kind):
    s = time.time() - T0
    gc.collect()
    torch.cuda.empty_cache()
    m = SPEC[kind]["cls"].from_pretrained(CK[kind], torch_dtype=torch.float16,
                                          device_map=device_map(SPEC[kind]["layers"]),
                                          local_files_only=True)
    m.eval()
    events.append((f"load {kind}", "load", s, time.time() - T0))
    log(f"{kind} loaded in {(time.time() - T0 - s) / 60:.1f} min")
    return m


def free_model(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def infer_parquet(model, kind, processor, parquet_path, max_tokens, num_workers, batch_size=80):
    """Two-GPU pipelined inference (first half of the layers on cuda:0, second half on cuda:1)."""
    ds = LMSYSDataset(csv_file=parquet_path, query=None, processor=processor,
                      include_swap=False, is_parquet=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=ShardedMaxTokensCollator(max_tokens=max_tokens, base_collator=VarlenCollator()))

    cfg = model.config
    dim = cfg.head_dim if kind == "gemma" else cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv0, inv1 = inv.to("cuda:0"), inv.to("cuda:1")

    outs, first, prev_h, prev_s = [], True, None, None
    t = time.time()
    for bi, batch in enumerate(dl, 1):
        for micro in batch:
            input_ids = to_device(micro["input_ids"], "cuda:0")
            seq_info = to_device(dict(
                cu_seqlens=micro["cu_seqlens"],
                position_ids=micro["position_ids"],
                max_seq_len=micro["max_seq_len"],
                attn_bias=BlockDiagonalCausalMask.from_seqlens(micro["seq_lens"]),
            ), "cuda:0")
            with torch.cuda.amp.autocast():
                if first:
                    hidden = model.forward_part1(input_ids, seq_info, inv0)
                    first = False
                else:
                    logits = model.forward_part2(prev_h, prev_s, inv1)
                    hidden = model.forward_part1(input_ids, seq_info, inv0)
                    outs.append(logits.cpu())
            prev_s, prev_h = to_device([seq_info, hidden], "cuda:1")
        if bi % 10 == 0 or bi == len(dl):
            done = time.time() - t
            log(f"  {kind} {os.path.basename(parquet_path)}: batch {bi}/{len(dl)} "
                f"| ETA {done / bi * (len(dl) - bi) / 60:.0f} min")
    with torch.cuda.amp.autocast():
        outs.append(model.forward_part2(prev_h, prev_s, inv1).cpu())
    return torch.cat(outs, dim=0).float().softmax(-1).numpy().astype(np.float32)


def run_infer(model, kind, processor, path):
    """Retry with smaller micro-batches on CUDA out-of-memory, and without workers if they break."""
    for workers in (NUM_WORKERS, 0):
        for max_tokens in (8192, 4096, 2048):
            try:
                return infer_parquet(model, kind, processor, path, max_tokens, workers)
            except torch.cuda.OutOfMemoryError:
                log(f"  CUDA OOM at max_tokens={max_tokens}; retrying smaller")
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if workers and "DataLoader worker" in str(e):
                    log(f"  DataLoader workers failed ({e}); retrying with num_workers=0")
                    break
                raise
    raise RuntimeError("inference failed even with the smallest settings")


# ------------------------------- blending -------------------------------
def pool(probs, weights=None):
    w = np.ones(len(probs)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if POOL == "geo":
        p = np.exp(sum(wi * np.log(pi + 1e-6) for wi, pi in zip(w, probs)))
    else:
        p = sum(wi * pi for wi, pi in zip(w, probs))
    return p / p.sum(axis=1, keepdims=True)


def length_prior(prob):
    if not LENGTH_PRIOR:
        return prob
    la = df_test["response_a"].str.len().values.astype(np.float64)
    lb = df_test["response_b"].str.len().values.astype(np.float64)
    f = 1.0 + LENGTH_PRIOR * (la - lb) / (la + lb + 1e-6)
    p = prob.copy()
    p[:, 0] *= f
    p[:, 1] *= 2.0 - f
    return p / p.sum(axis=1, keepdims=True)


results = {}       # (kind, direction) -> test probabilities in the original A/B orientation
val_results = {}   # same for the labelled validation rows
pass_time = {}     # kind -> seconds for one full test pass


def blend():
    """(per-model probabilities, final probabilities) from whatever passes have finished."""
    kinds = [k for k in WEIGHTS if any(r[0] == k for r in results)]
    per = {k: length_prior(pool([p for (kk, _), p in results.items() if kk == k])) for k in kinds}
    return per, pool([per[k] for k in kinds], [WEIGHTS[k] for k in kinds])


def do_pass(model, kind, direction, processor, split="test"):
    s = time.time() - T0
    p = run_infer(model, kind, processor, VIEWS[split][direction])
    if direction == "ba":
        p = p[:, [1, 0, 2]]            # swap A/B columns back
    n_expected = N if split == "test" else len(val_labels)
    assert len(p) == n_expected, (len(p), n_expected)
    events.append((f"{kind} {direction}" + (" (val)" if split == "val" else ""), kind, s, time.time() - T0))
    if split == "val":
        val_results[(kind, direction)] = p
        return
    results[(kind, direction)] = p
    pass_time.setdefault(kind, time.time() - T0 - s)
    np.save(f"{WORK}/prob_{kind}_{direction}.npy", p)
    log(f"{kind} {direction} done in {(time.time() - T0 - s) / 60:.1f} min")
    save_submission(blend()[1], f"interim, passes: {sorted(results)}")


def attempt(desc, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        log(f"FAILED {desc}:\n{traceback.format_exc()}")
        gc.collect()
        torch.cuda.empty_cache()
        return None


def fits(seconds_needed):
    return time.time() - T0 + seconds_needed <= BUDGET_S


def run_model(kind, first_dir, second_dir, reserve_after_first=lambda: 0.0):
    """Load once, run the baseline direction, optional labelled validation, then the TTA direction if it fits."""
    model = attempt(f"load {kind}", load_model, kind)
    if model is None:
        return
    proc = attempt(f"processor {kind}", make_processor, kind)
    if proc is not None:
        attempt(f"{kind} {first_dir}", do_pass, model, kind, first_dir, proc)
    if (kind, first_dir) in results:
        if "val" in VIEWS:
            for d in ("ab", "ba"):
                attempt(f"{kind} {d} validation", do_pass, model, kind, d, proc, "val")
        if fits(1.1 * pass_time[kind] + reserve_after_first()):
            attempt(f"{kind} {second_dir}", do_pass, model, kind, second_dir, proc)
        else:
            log(f"skipping {kind} {second_dir} TTA pass: not enough time budget")
    free_model(model)


# ------------------------------- run ------------------------------------
# Gemma2-9B: (A, B) baseline, then (B, A) TTA if the Llama3 baseline still fits afterwards.
run_model("gemma", "ab", "ba", lambda: LLAMA_TO_GEMMA_TIME * pass_time["gemma"] + 15 * 60)
# Llama3-8B: (B, A) baseline (same orientation as tascj0's predict_m3), then (A, B) TTA.
run_model("llama", "ba", "ab")

if results:
    per_model, final_prob = blend()
    sub = save_submission(final_prob, f"FINAL, passes: {sorted(results)}")
    print(sub.head())
else:
    log("NO MODEL PASS FINISHED - submission.csv only contains the class prior")


# ------------------------------- report ---------------------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CLS_C = {"A wins": "#2a78d6", "B wins": "#eb6834", "Tie": "#1baf7a"}          # validated, all-pairs
MODEL_C = {"gemma": "#4a3aa7", "llama": "#008300", "final": INK, "load": "#b8b7b0"}
MODEL_NAME = {"gemma": "Gemma2-9B", "llama": "Llama3-8B", "final": "Ensemble"}


def logloss(p, y):
    return float(-(y * np.log(np.clip(p, 1e-15, 1.0))).sum(axis=1).mean())


def make_report():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(f"{WORK}/charts", exist_ok=True)
    seq = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#3987e5", "#0d366b"])
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "font.size": 10, "axes.edgecolor": "#c9c8c2", "text.color": INK})

    def style(ax, title, xlabel="", ylabel="", grid=True):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel(ylabel, color=INK2)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def note(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=10)

    per, final = blend()
    kinds = list(per)
    names = [MODEL_NAME[k] for k in kinds] + ["Ensemble"]
    allp = [per[k] for k in kinds] + [final]
    enough = N >= 30

    fig, axes = plt.subplots(3, 3, figsize=(19, 15.5), constrained_layout=True)
    fig.suptitle(f"LMSYS ensemble diagnostics | {N:,} test rows | passes: "
                 + ", ".join(f"{k}-{d}" for k, d in sorted(results)), fontsize=14, fontweight="bold", x=0.01, ha="left")

    # 1 predicted class mix
    ax = axes[0, 0]
    xs = np.arange(len(names))
    for j, (cname, colr) in enumerate(CLS_C.items()):
        share = [100 * np.mean(p.argmax(1) == j) for p in allp]
        ax.bar(xs + (j - 1) * 0.24, share, width=0.22, color=colr, label=cname)
    ax.set_xticks(xs, names)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    style(ax, "Predicted class mix", ylabel="% of rows (argmax)")

    # 2 confidence
    ax = axes[0, 1]
    for k, p in zip(kinds + ["final"], allp):
        ax.hist(p.max(1), bins=20, range=(1 / 3, 1), histtype="step", lw=2 if k == "final" else 1.5,
                color=MODEL_C[k], label=MODEL_NAME[k])
    ax.legend(frameon=False, fontsize=9)
    style(ax, "Confidence (highest class probability)", "max probability", "rows")

    # 3 simplex
    ax = axes[0, 2]
    idx = np.random.default_rng(0).choice(N, min(N, 8000), replace=False)
    pf = final[idx]
    x = pf[:, 1] + 0.5 * pf[:, 2]
    y = np.sqrt(3) / 2 * pf[:, 2]
    for j, (cname, colr) in enumerate(CLS_C.items()):
        m = pf.argmax(1) == j
        ax.scatter(x[m], y[m], s=7, alpha=0.35, color=colr, linewidths=0, label=cname)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="#c9c8c2", lw=1)
    for lab, (tx, ty) in {"A wins": (-0.02, -0.05), "B wins": (1.02, -0.05), "Tie": (0.5, 0.9)}.items():
        ax.text(tx, ty, lab, ha="center", color=INK2, fontsize=10)
    ax.set_xlim(-0.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, fontsize=9, loc="upper left", markerscale=2)
    ax.set_title("Ensemble predictions on the outcome simplex", loc="left", fontsize=11, fontweight="bold")

    # 4 model agreement
    ax = axes[1, 0]
    if len(kinds) == 2 and enough:
        a, b = per["gemma"][:, 0], per["llama"][:, 0]
        hb = ax.hexbin(a, b, gridsize=32, bins="log", mincnt=1, cmap=seq, extent=(0, 1, 0, 1))
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.text(0.03, 0.94, f"r = {np.corrcoef(a, b)[0, 1]:.3f}", transform=ax.transAxes, color=INK)
        fig.colorbar(hb, ax=ax, label="rows (log)", shrink=0.8)
    else:
        note(ax, "needs both models and >= 30 rows")
    style(ax, "Model agreement: P(A wins)", "Gemma2-9B", "Llama3-8B")

    # 5 TTA position bias
    ax = axes[1, 1]
    drawn = False
    for k in kinds:
        if (k, "ab") in results and (k, "ba") in results:
            d = results[(k, "ab")][:, 0] - results[(k, "ba")][:, 0]
            flip = 100 * np.mean(results[(k, "ab")].argmax(1) != results[(k, "ba")].argmax(1))
            ax.hist(d, bins=40, range=(-1, 1), histtype="step", lw=1.6, color=MODEL_C[k],
                    label=f"{MODEL_NAME[k]} (verdict flips {flip:.1f}%)")
            drawn = True
    if drawn:
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    else:
        note(ax, "swap-TTA pass not run")
    style(ax, "Position bias: P(A) as-given minus swapped", "P(A wins | A,B) - P(A wins | B,A)", "rows")

    # 6 length bias
    ax = axes[1, 2]
    if enough:
        lr = np.log((df_test["response_a"].str.len() + 1) / (df_test["response_b"].str.len() + 1)).values
        order = np.argsort(lr)
        groups = np.array_split(order, 12)
        gx = [lr[g].mean() for g in groups]
        gy = np.array([(final[g, 0] - final[g, 1]).mean() for g in groups])
        ge = np.array([(final[g, 0] - final[g, 1]).std() / np.sqrt(len(g)) for g in groups])
        ax.fill_between(gx, gy - ge, gy + ge, color=MODEL_C["final"], alpha=0.12, lw=0)
        ax.plot(gx, gy, color=MODEL_C["final"], lw=2, marker="o", ms=5)
        ax.axhline(0, color=INK2, lw=1, ls="--")
    else:
        note(ax, "needs >= 30 rows")
    style(ax, "Length bias of the ensemble", "log(len A / len B)  ->  A longer", "mean P(A) - P(B)")

    # 7 tie vs closeness
    ax = axes[2, 0]
    if enough:
        hb = ax.hexbin(np.abs(final[:, 0] - final[:, 1]), final[:, 2], gridsize=30, bins="log", mincnt=1, cmap=seq)
        fig.colorbar(hb, ax=ax, label="rows (log)", shrink=0.8)
    else:
        note(ax, "needs >= 30 rows")
    style(ax, "Ties sit where A and B are close", "|P(A) - P(B)|", "P(tie)")

    # 8 timeline
    ax = axes[2, 1]
    for i, (label, kind, s, e) in enumerate(events):
        ax.barh(i, max((e - s) / 60, 0.02), left=s / 60, height=0.55, color=MODEL_C.get(kind, MODEL_C["load"]))
    ax.set_yticks(range(len(events)), [e[0] for e in events])
    ax.invert_yaxis()
    end = max((e[3] for e in events), default=1) / 60
    if BUDGET_S / 60 <= 1.5 * end:
        ax.axvline(BUDGET_S / 60, color=INK2, ls="--", lw=1)
        ax.text(BUDGET_S / 60, -0.7, " budget", color=INK2, fontsize=9)
    style(ax, f"Run timeline (budget {BUDGET_S / 3600:.1f} h)", "minutes since start")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MODEL_C[k], label=MODEL_NAME.get(k, "model load")) for k in kinds + ["load"]],
              frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)

    # 9 weight sensitivity
    ax = axes[2, 2]
    if len(kinds) == 2:
        ws = np.linspace(0, 1, 21)
        base = pool([per["gemma"], per["llama"]], [WEIGHTS["gemma"], WEIGHTS["llama"]]).argmax(1)
        changed = [100 * np.mean(pool([per["gemma"], per["llama"]], [w + 1e-9, 1 - w + 1e-9]).argmax(1) != base) for w in ws]
        ax.plot(ws, changed, color=MODEL_C["final"], lw=2)
        ax.axvline(WEIGHTS["gemma"], color=MODEL_C["gemma"], ls="--", lw=1.2)
        ax.text(WEIGHTS["gemma"], max(changed) * 0.92, f" current {WEIGHTS['gemma']:.2f}", color=INK2, fontsize=9)
    else:
        note(ax, "needs both models")
    style(ax, "How much the blend weight matters", "weight on Gemma2-9B", "% of verdicts that differ from current")

    out = f"{WORK}/charts/dashboard.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"saved {out}")

    # ---- labelled validation (optional) ----
    if val_labels is not None and val_results:
        pv = {k: pool([p for (kk, _), p in val_results.items() if kk == k]) for k in kinds if any(r[0] == k for r in val_results)}
        if pv:
            fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
            fig.suptitle(f"Validation on {len(val_labels)} labelled train rows - optimistic: the checkpoints "
                         "were probably trained on train.csv", fontsize=12, fontweight="bold", x=0.01, ha="left")
            fin_v = pool(list(pv.values()), [WEIGHTS[k] for k in pv])
            rows = [(f"{MODEL_NAME[k]} {d}", k, logloss(p, val_labels)) for (k, d), p in sorted(val_results.items())]
            rows += [(f"{MODEL_NAME[k]} + swap", k, logloss(p, val_labels)) for k, p in pv.items()
                     if sum(1 for r in val_results if r[0] == k) == 2]
            rows += [("Ensemble", "final", logloss(fin_v, val_labels)),
                     ("Class prior", "load", logloss(np.tile(PRIOR, (len(val_labels), 1)), val_labels))]
            ax = axes[0]
            for i, (lab, k, v) in enumerate(rows):
                ax.barh(i, v, height=0.55, color=MODEL_C[k])
                ax.text(v, i, f" {v:.3f}", va="center", color=INK2, fontsize=9)
            ax.set_yticks(range(len(rows)), [r[0] for r in rows])
            ax.invert_yaxis()
            style(ax, "Log loss (lower is better)", "log loss")

            ax = axes[1]
            pf, yf = fin_v.ravel(), val_labels.ravel()
            bins = np.linspace(0, 1, 11)
            bi = np.clip(np.digitize(pf, bins) - 1, 0, 9)
            mx = [pf[bi == b].mean() for b in range(10) if (bi == b).any()]
            my = [yf[bi == b].mean() for b in range(10) if (bi == b).any()]
            cnt = [(bi == b).sum() for b in range(10) if (bi == b).any()]
            ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
            ax.scatter(mx, my, s=np.array(cnt) / max(cnt) * 220 + 20, color=MODEL_C["final"], zorder=3)
            style(ax, "Reliability of the ensemble", "predicted probability", "observed frequency")

            ax = axes[2]
            if len(pv) == 2:
                ws = np.linspace(0, 1, 21)
                ll = [logloss(pool([pv["gemma"], pv["llama"]], [w + 1e-9, 1 - w + 1e-9]), val_labels) for w in ws]
                ax.plot(ws, ll, color=MODEL_C["final"], lw=2)
                ax.axvline(WEIGHTS["gemma"], color=MODEL_C["gemma"], ls="--", lw=1.2)
                ax.text(WEIGHTS["gemma"], min(ll), f" current {WEIGHTS['gemma']:.2f}", color=INK2, fontsize=9)
            else:
                note(ax, "needs both models")
            style(ax, "Blend weight sweep", "weight on Gemma2-9B", "log loss")
            out = f"{WORK}/charts/validation.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            log(f"saved {out}")


if MAKE_CHARTS and results:
    attempt("charts", make_report)
    try:
        from IPython.display import Image, display
        for name in ("dashboard", "validation"):
            path = f"{WORK}/charts/{name}.png"
            if os.path.exists(path):
                display(Image(path))
    except Exception:
        pass

with open(f"{WORK}/run_summary.json", "w") as f:
    json.dump({"rows": N, "passes": [f"{k}-{d}" for k, d in sorted(results)],
               "pass_minutes": {k: round(v / 60, 2) for k, v in pass_time.items()},
               "weights": WEIGHTS, "pool": POOL, "total_minutes": round((time.time() - T0) / 60, 1)}, f, indent=2)
log(f"done | passes run: {sorted(results)} | weights {WEIGHTS} | pool {POOL}")
if not results:
    raise RuntimeError("No model pass finished; see the FAILED messages above. submission.csv is prior-only.")
