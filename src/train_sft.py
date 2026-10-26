"""SFT of the policy (Qwen2.5-1.5B-Instruct + LoRA) on the chosen summaries.

Input: jsonl with {"messages": [system, user, assistant]} (aggregate_labels.py --mode sft).
The loss is on the assistant summary only (converted to TRL prompt/completion format).

    # 10-step test on 80 examples
    python src/train_sft.py --run sft_test --train data/sft/A_train.jsonl --max_steps 10 --limit 80
    # real run: eval on A_val every epoch, early stop on val loss, best epoch's adapter is kept
    python src/train_sft.py --run sft_v1 --train data/sft/A_train.jsonl --eval data/sft/A_val.jsonl

Adapter -> models/{run}/ ; logs -> outputs/train_logs/{run}_steps.csv, {run}_summary.json,
and with --eval also {run}.json (train/val loss curves, best step, peak memory, data length stats).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_utils import LOGS, MODELS, StepLogger, guard_vram, run_guarded  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_prompt_completion(ex):
    """Split a chat example into TRL prompt / completion form, so the loss covers the assistant summary only."""
    return {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}


def write_report(args, trainer, summary, base_eval, target_tokens):
    """Write outputs/train_logs/{run}.json: loss curves, best step, memory, time and SFT-target length statistics."""
    st = trainer.state
    hist = st.log_history
    best_step = int(Path(st.best_model_checkpoint).name.split("-")[-1]) if st.best_model_checkpoint else None
    evals = [{"step": 0, "epoch": 0.0, "eval_loss": base_eval}] + [
        {"step": h["step"], "epoch": round(h["epoch"], 3), "eval_loss": h["eval_loss"]} for h in hist if "eval_loss" in h]
    report = {
        "run": args.run, "model": args.model, "adapter_dir": str(Path(trainer.args.output_dir).relative_to(MODELS.parent)),
        "best_step": best_step, "best_eval_loss": st.best_metric,
        "best_epoch": next((e["epoch"] for e in evals if e["step"] == best_step), None),
        "epochs_run": round(st.epoch, 3), "max_epochs": args.epochs, "steps_per_epoch": st.max_steps / args.epochs,
        "early_stopped": st.epoch < args.epochs - 1e-6,
        "eval_curve": evals,
        "train_curve": [{"step": h["step"], "epoch": round(h["epoch"], 3), "loss": h["loss"],
                         "mean_token_accuracy": h.get("mean_token_accuracy"), "lr": h.get("learning_rate")}
                        for h in hist if "loss" in h],
        "peak_alloc_gb": summary["peak_alloc_gb"], "peak_reserved_gb": summary["peak_reserved_gb"],
        "mean_step_seconds": summary["mean_step_seconds_after_first"], "total_seconds": summary["total_seconds"],
        "data": {sp: {"n": len(t), "target_tokens_mean": round(sum(t) / len(t)), "target_tokens_max": max(t),
                      "targets_over_500_tokens": sum(n > 500 for n in t)} for sp, t in target_tokens.items()},
        "settings": summary["settings"], "versions": summary["versions"],
    }
    (LOGS / f"{args.run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best step {best_step} (epoch {report['best_epoch']}) val loss {st.best_metric:.4f} "
          f"(base {base_eval:.4f}) -> {LOGS / (args.run + '.json')}")


def main():
    """Fine-tune the policy with a LoRA adapter on the SFT targets; with --eval, evaluate each epoch and keep the best."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, help="use only the first N training examples")
    ap.add_argument("--epochs", type=float, default=6, help="maximum; with --eval, early stopping usually ends sooner")
    ap.add_argument("--patience", type=int, default=2, help="epochs without val-loss improvement before stopping")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=2560, help="longest prompt+summary is ~2.4k tokens")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    guard_vram()

    # ---- data in prompt/completion form, target lengths, base model and LoRA config
    files = {"train": args.train} | ({"eval": args.eval} if args.eval else {})
    ds = load_dataset("json", data_files=files)
    ds = ds.map(to_prompt_completion, remove_columns=ds["train"].column_names)
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))

    tok = AutoTokenizer.from_pretrained(args.model)
    target_tokens = {sp: [len(tok(ex["completion"][0]["content"])["input_ids"]) for ex in d] for sp, d in ds.items()}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")

    # ---- training configuration (early stopping on validation loss when --eval is given)
    cfg = SFTConfig(
        output_dir=str(MODELS / args.run),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,  # transformers 5: float < 1 = ratio
        max_length=args.max_length, completion_only_loss=True,
        gradient_checkpointing=not args.no_grad_ckpt, bf16=True,
        logging_steps=1, eval_strategy="epoch" if args.eval else "no",
        # with --eval: keep one checkpoint per epoch, reload the best (lowest val loss) at the end
        save_strategy="epoch" if args.eval else "no", save_total_limit=args.patience + 1,
        load_best_model_at_end=bool(args.eval), metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=args.seed,
    )
    # ---- train, save the adapter and the report
    settings = {k: v for k, v in vars(args).items()} | {"n_train": len(ds["train"])}
    logger = StepLogger(args.run, "sft", settings)
    callbacks = [logger] + ([EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.eval else [])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds.get("eval"),
                         processing_class=tok, peft_config=peft_config, callbacks=callbacks)
    base_eval = trainer.evaluate()["eval_loss"] if args.eval else None  # step 0 = untouched base model
    summary = run_guarded(trainer, logger)
    trainer.save_model(cfg.output_dir)
    if args.eval:
        write_report(args, trainer, summary, base_eval, target_tokens)
    print(f"peak reserved {summary['peak_reserved_gb']} GB | {summary['mean_step_seconds_after_first']} s/step "
          f"-> {cfg.output_dir}")


if __name__ == "__main__":
    main()
