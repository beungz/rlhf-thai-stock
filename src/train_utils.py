"""Shared helpers for train_sft.py / train_rm.py / train_ppo.py.

StepLogger    : per-step wall time + peak GPU memory + trainer logs -> outputs/train_logs/{run}_steps.csv
                and a run summary -> outputs/train_logs/{run}_summary.json.
AdaptiveKL    : Ziegler et al. (2019) adaptive KL controller for TRL's PPOTrainer, which only has a fixed kl_coef.
                PPOTrainer reads args.kl_coef at every rollout, so updating it from on_log changes the next update.
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "train_logs"
MODELS = ROOT / "models"  # git-ignored (model weights are downloaded separately; see README)

GB = 1024 ** 3


# ---- run metadata and per-step logging
def versions() -> dict:
    """Library versions and GPU name, stored in every run summary."""
    import peft
    import transformers
    import trl
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / GB, 2)
            if torch.cuda.is_available() else None}


class StepLogger(TrainerCallback):
    """Trainer callback that records wall time, peak GPU memory and every logged metric per optimisation step,
    and rewrites the steps CSV and the run summary as training goes (so a crashed run still leaves its logs).
    """
    def __init__(self, run: str, stage: str, settings: dict):
        """run: file name prefix; stage: sft / rm / ppo; settings: the run's arguments (stored in the summary)."""
        self.run, self.stage, self.settings = run, stage, settings
        self.rows: dict[int, dict] = {}
        self.t_last = None
        self.t_start = None
        self.status = "running"
        LOGS.mkdir(parents=True, exist_ok=True)

    def _row(self, step):
        """The log row of one step (created on first use)."""
        return self.rows.setdefault(step, {"step": step})

    def on_train_begin(self, args, state, control, **kw):
        """Start the clock and reset the peak-memory counter."""
        torch.cuda.reset_peak_memory_stats()
        self.t_start = self.t_last = time.time()

    def on_step_end(self, args, state, control, **kw):
        """Record the step's duration and peak memory, then rewrite the logs."""
        now = time.time()
        row = self._row(state.global_step)
        row["step_seconds"] = round(now - self.t_last, 2)
        row["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
        row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / GB, 3)
        self.t_last = now
        torch.cuda.reset_peak_memory_stats()
        self.write()

    def on_log(self, args, state, control, logs=None, **kw):
        """Copy the numeric metrics the trainer logged into the step's row (and the current KL coefficient)."""
        row = self._row(state.global_step)
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)):
                row[k] = v
        if "kl_coef" in getattr(args, "__dict__", {}):
            row["kl_coef"] = args.kl_coef

    def on_train_end(self, args, state, control, **kw):
        """Mark the run as finished and write the final logs."""
        self.status = "ok"
        self.write()

    def write(self, error: str | None = None):
        """Write {run}_steps.csv and {run}_summary.json; returns the summary dict."""
        rows = [self.rows[k] for k in sorted(self.rows)]
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(LOGS / f"{self.run}_steps.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        timed = [r for r in rows if "step_seconds" in r]
        summary = {
            "run": self.run, "stage": self.stage, "status": "error" if error else self.status, "error": error,
            "steps_done": len(timed),
            "peak_alloc_gb": max((r["peak_alloc_gb"] for r in timed), default=None),
            "peak_reserved_gb": max((r["peak_reserved_gb"] for r in timed), default=None),
            # step 1 includes warm-up / compilation, so report the mean over the rest too
            "mean_step_seconds": round(sum(r["step_seconds"] for r in timed) / len(timed), 2) if timed else None,
            "mean_step_seconds_after_first": round(sum(r["step_seconds"] for r in timed[1:]) / (len(timed) - 1), 2)
            if len(timed) > 1 else None,
            "total_seconds": round(time.time() - self.t_start, 1) if self.t_start else None,
            "settings": self.settings, "versions": versions(),
        }
        (LOGS / f"{self.run}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


# ---- adaptive KL controller for PPO
class AdaptiveKL(TrainerCallback):
    """kl_coef <- kl_coef * (1 + clip(KL/target - 1, -0.2, 0.2) * n_steps / horizon)   (Ziegler et al. 2019).

    KL is TRL's `objective/kl` (sum over response tokens, mean over the batch); n_steps = episodes per update.
    """

    def __init__(self, target: float, horizon: int):
        """target: KL target in nats per response; horizon: episodes over which the coefficient adapts."""
        self.target, self.horizon = target, horizon

    def on_log(self, args, state, control, logs=None, **kw):
        """Move kl_coef towards the KL target after each update."""
        if not logs or "objective/kl" not in logs:
            return
        err = max(-0.2, min(0.2, logs["objective/kl"] / self.target - 1))
        args.kl_coef *= 1 + err * args.batch_size / self.horizon


# ---- GPU memory guard and OOM-safe training
def guard_vram(fraction: float = 0.95) -> None:
    """Cap PyTorch's allocator below the card's dedicated memory.

    On Windows the NVIDIA driver (default 'sysmem fallback') can silently spill into shared system RAM instead of
    raising OOM, and training then runs about 100x slower.
    With the cap, overflow raises torch.OutOfMemoryError instead.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def run_guarded(trainer, logger: StepLogger):
    """trainer.train(), but an out-of-memory error is written to the summary instead of only to the console."""
    try:
        trainer.train()
    except torch.OutOfMemoryError as e:
        logger.write(error=f"OOM: {str(e).splitlines()[0]}")
        raise
    return logger.write()
