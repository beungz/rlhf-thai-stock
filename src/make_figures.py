"""Figures for the write-up (reads outputs/train_logs/*, writes outputs/figures/*.png).

    python src/make_figures.py
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "train_logs"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SURF = "#1d1d1b", "#6b6b66", "#e6e6e3", "#fcfcfb"
RUNS = {  # fixed categorical order (validated palette slots 1-3) + marker as secondary encoding
    "ppo_v1": dict(color="#2a78d6", marker="o", label="ppo_v1  (rm_v2, KL coef 0.05)"),
    "ppo_lowkl": dict(color="#eb6834", marker="s", label="ppo_lowkl  (rm_v2, KL coef 0.005)"),
    "ppo_rmv1": dict(color="#1baf7a", marker="^", label="ppo_rmv1  (rm_v1, KL coef 0.01)"),
}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
                     "lines.markersize": 7, "legend.frameon": False})


# ---- helpers
def rcsv(p):
    """Read a CSV file into a list of dict rows."""
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x):
    """Float, or NaN for empty / non-numeric cells."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def smooth(xs, k=10):
    """Trailing moving average over the last k values (NaN values ignored)."""
    out = []
    for i in range(len(xs)):
        w = [x for x in xs[max(0, i - k + 1):i + 1] if not math.isnan(x)]
        out.append(sum(w) / len(w) if w else math.nan)
    return out


def save(fig, name):
    """Save a figure to outputs/figures/ and close it."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


# ---- load the PPO logs (per-update steps and per-checkpoint evaluations)
steps = {r: rcsv(LOGS / f"{r}_steps.csv") for r in RUNS}
ckpts = {r: rcsv(LOGS / f"{r}_checkpoints.csv") for r in RUNS}


def ckpt_series(run, col):
    """(steps, values) of one column of a run's checkpoint table, SFT (step 0) and every step_N."""
    rows = [x for x in ckpts[run] if x["checkpoint"] == "sft" or x["checkpoint"].startswith("step_")]
    return [int(x["step"]) for x in rows], [f(x[col]) for x in rows]


# 1. Proxy vs gold over PPO steps (two panels, one axis each)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for r, st in RUNS.items():
    x, y = ckpt_series(r, "rm_reward_mean")
    ax[0].plot(x, y, color=st["color"], marker=st["marker"], label=st["label"])
    x, y = ckpt_series(r, "gold_mean")
    ax[1].plot(x, y, color=st["color"], marker=st["marker"], label=st["label"])
base = ckpts["ppo_v1"][0]
ax[1].axhline(f(base["gold_mean"]), color=MUTED, lw=1, ls="--")
ax[1].text(1, f(base["gold_mean"]) + 0.01, "base model", color=MUTED, fontsize=9)
ax[0].set(title="Proxy: score from the RM each run optimised", xlabel="PPO update (0 = SFT)", ylabel="mean RM score")
ax[1].set(title="Gold: automatic quality score (never seen by PPO)", xlabel="PPO update (0 = SFT)", ylabel="mean gold (0-1)")
ax[0].legend(loc="upper left", fontsize=9)
fig.suptitle("Proxy reward vs true quality on 70 held-out prompts (greedy)", fontweight="bold")
save(fig, "1_proxy_vs_gold.png")

# 2. Rollout dynamics during training (small multiples, 10-update moving average)
panels = [("objective/scores", "RM score of rollouts (proxy)"), ("rollout/gold_mean", "Gold of rollouts"),
          ("objective/kl", "KL to SFT policy (nats)"), ("rollout/eos_rate", "Share of rollouts that end (EOS)"),
          ("rollout/tokens_mean", "Rollout length (tokens)"), ("rollout/severe_rate", "Severe-error rate (auto)")]
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for a, (col, title) in zip(axs.flat, panels):
    for r, st in RUNS.items():
        xs = [int(x["step"]) for x in steps[r]]
        a.plot(xs, smooth([f(x.get(col)) for x in steps[r]]), color=st["color"], label=st["label"])
    a.set(title=title, xlabel="PPO update")
axs[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("PPO training rollouts (T = 0.7, 8 per update; 10-update moving average)", fontweight="bold")
save(fig, "2_rollout_dynamics.png")

# 3. PPO optimisation statistics, main run
panels = [("objective/scores", "RM score"), ("objective/rlhf_reward", "RLHF reward (RM - KL penalty)"),
          ("objective/kl", "KL (nats)"), ("policy/entropy_avg", "Policy entropy"),
          ("loss/policy_avg", "Policy loss"), ("loss/value_avg", "Value loss")]
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for a, (col, title) in zip(axs.flat, panels):
    for r, st in RUNS.items():
        xs = [int(x["step"]) for x in steps[r]]
        ys = [f(x.get(col)) for x in steps[r]]
        a.plot(xs, ys, color=st["color"], alpha=0.18, lw=1)
        a.plot(xs, smooth(ys), color=st["color"], label=st["label"])
    a.set(title=title, xlabel="PPO update")
axs[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("PPO training statistics per update (faint = raw, solid = 10-update average)", fontweight="bold")
save(fig, "3_ppo_training_stats.png")

# 4. Reward model: learning curve + probe robustness
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
lc = [("rm_lc25", 0.25), ("rm_lc50", 0.50), ("rm_lc75", 0.75), ("rm_v1", 1.0)]
xs, ys = [], []
for name, frac in lc:
    d = json.loads((LOGS / f"{name}.json").read_text(encoding="utf-8"))
    xs.append(d["n_train_pairs"]); ys.append(d["test"]["accuracy"])
ax[0].plot(xs, ys, color="#2a78d6", marker="o", label="rm_v1 family (judge pairs only)")
for x_, y_ in zip(xs, ys):
    ax[0].annotate(f"{y_:.2f}", (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
d2 = json.loads((LOGS / "rm_v2.json").read_text(encoding="utf-8"))
ax[0].plot([609], [d2["test"]["accuracy"]], color="#eb6834", marker="s", ls="none", label="rm_v2 (+483 synthetic pairs)")
ax[0].annotate(f"{d2['test']['accuracy']:.2f}", (609, d2["test"]["accuracy"]), textcoords="offset points", xytext=(0, -14),
               ha="center", fontsize=9)
ax[0].axhline(0.5, color=MUTED, lw=1, ls="--")
ax[0].set(title="RM learning curve (B_test pairwise accuracy)", xlabel="judge-labelled training pairs",
          ylabel="accuracy", ylim=(0.45, 0.85))
ax[0].legend(loc="upper left", fontsize=9)
probes = ["empty", "template", "truncated", "wrong_but_formatted", "long", "keyword_stuffed"]
d1 = json.loads((LOGS / "rm_v1.json").read_text(encoding="utf-8"))
w = 0.38
for j, (d, name, col) in enumerate([(d1, "rm_v1", "#2a78d6"), (d2, "rm_v2", "#eb6834")]):
    vals = [d["probes"]["by_probe"][p]["correct_beats_pct"] for p in probes]
    pos = [i + (j - 0.5) * w for i in range(len(probes))]
    ax[1].bar(pos, vals, width=w - 0.04, color=col, label=name)
    for x_, v in zip(pos, vals):
        ax[1].text(x_, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
ax[1].axhline(80, color=MUTED, lw=1, ls="--")
ax[1].text(len(probes) - 0.5, 82, "gate 80%", color=MUTED, fontsize=8, ha="right")
ax[1].set_xticks(range(len(probes)))
ax[1].set_xticklabels(["empty", "template", "truncated", "wrong but\nformatted", "long*", "keyword\nstuffed*"])
ax[1].set(title="Probe suite: % of prompts where correct beats the probe", ylabel="% correct > probe",
          ylim=(0, 110))
ax[1].legend(loc="upper right", fontsize=9)
ax[1].text(0, -0.2, "* never used in rm_v2's synthetic training pairs (held-out hacks)", transform=ax[1].transAxes,
           fontsize=8, color=MUTED)
save(fig, "4_reward_model.png")

# 5. SFT curves
d = json.loads((LOGS / "sft_v1.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
tc = d["train_curve"]
ax[0].plot([x["step"] for x in tc], [x["loss"] for x in tc], color="#2a78d6", alpha=0.35, lw=1)
ax[0].plot([x["step"] for x in tc], smooth([x["loss"] for x in tc], 5), color="#2a78d6", label="train (5-step avg)")
ax[0].plot([x["step"] for x in d["eval_curve"]], [x["eval_loss"] for x in d["eval_curve"]], color="#eb6834", marker="s",
           label="A_val (per epoch)")
ax[0].set(title="SFT loss (assistant tokens only)", xlabel="optimizer step (13 per epoch)", ylabel="cross-entropy")
ax[0].legend()
rows = rcsv(LOGS / "sft_v1_steps.csv")
ev = [(int(r["step"]), f(r["eval_mean_token_accuracy"])) for r in rows if r.get("eval_mean_token_accuracy")]
tr = [(int(r["step"]), f(r["mean_token_accuracy"])) for r in rows if r.get("mean_token_accuracy")]
ax[1].plot([s for s, _ in tr], smooth([v for _, v in tr], 5), color="#2a78d6", label="train (5-step avg)")
ax[1].plot([s for s, _ in ev], [v for _, v in ev], color="#eb6834", marker="s", label="A_val (per epoch)")
ax[1].set(title="SFT next-token accuracy", xlabel="optimizer step", ylabel="token accuracy")
ax[1].legend()
save(fig, "5_sft.png")
