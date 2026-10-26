"""PPO of the SFT policy against the reward model, with a KL penalty (fixed or adaptive).

TRL: PPOTrainer was REMOVED in TRL 1.13 -> this needs trl==1.12.0 (trl.experimental.ppo). TRL's PPO has a fixed
kl_coef only; adaptive KL (Ziegler et al. 2019) is added with train_utils.AdaptiveKL (--kl_target > 0).
Note: the controller changes kl_coef by at most 20% x (rollouts per update / --kl_horizon) per update, so with
8 rollouts per update and the default horizon of 10,000 it barely moves; use --kl_target 0 for a fixed coefficient.

Models
- policy    : Qwen2.5-1.5B-Instruct with the SFT LoRA merged in, plus a NEW LoRA trained by PPO.
              Reference = the same model with the PPO adapter disabled (= the SFT policy), so no second 1.5B copy.
- reward    : trained RM (train_rm.py), frozen, bf16.
- value     : a second copy of the RM, trained (full fine-tune) together with the policy adapter.

Prompts: guided_v2 on markdown cards for the companies in --split (company x T1/T2).

Rollout logging (RolloutStats): TRL's get_reward is wrapped so each update's scored responses are decoded; the steps CSV
gets their mean token length, end-of-text rate, gated gold (auto_rubric, facts of each prompt's symbol x snapshot),
severe / truncated / over-500 rates, and every rollout goes to outputs/ppo_rollouts/{run}.jsonl.

    # 10-step memory test on the 80 longest prompts
    python src/train_ppo.py --run ppo_test --sft_adapter models/sft_v1 --reward_model models/rm_v2 --steps 10 --longest 80
    # main run
    python src/train_ppo.py --run ppo_v1 --sft_adapter models/sft_v1 --reward_model models/rm_v2 --split C_ppo reserve --steps 120 --save_every 20

Adapter -> models/{run}/ ; logs -> outputs/train_logs/{run}_steps.csv, {run}_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          TrainerCallback)
from trl.experimental.ppo import PPOConfig, PPOTrainer  # noqa: E402
from trl.experimental.ppo import ppo_trainer as _ppo  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_rubric import auto_rubric  # noqa: E402
from gold_check import load_meta  # noqa: E402
from prompts import PROMPTS  # noqa: E402
from train_sft import LORA_TARGETS  # noqa: E402
from train_utils import MODELS, ROOT, AdaptiveKL, StepLogger, guard_vram, run_guarded  # noqa: E402

_batch_generation = _ppo.batch_generation


def _batch_generation_eval(model, *a, **kw):
    """Generate rollouts in eval mode. With gradient checkpointing on, transformers drops the KV cache in train mode,
    which makes 500-token generation from ~2k-token prompts recompute the whole context at every token.
    Dropout is already disabled by PPOTrainer, so eval mode changes nothing else."""
    was_training = model.training
    model.eval()
    try:
        return _batch_generation(model, *a, **kw)
    finally:
        model.train(was_training)


# patch TRL so rollouts are generated in eval mode (keeps the KV cache with gradient checkpointing on)
_ppo.batch_generation = _batch_generation_eval


class RolloutStats(TrainerCallback):
    """Gold metrics of the rollouts PPO actually trains on (the proxy-vs-gold curve), one row per update.

    TRL's rollout loop calls get_reward(reward_model, query + truncated response, pad, context_length) per forward
    batch; the wrapper decodes those responses. on_log then adds the stats to the update's logs - it must come BEFORE
    StepLogger in the callback list, which receives the same dict."""

    def __init__(self, reward_model, tok, prompt_meta: dict, run: str):
        """Wrap TRL's get_reward so every scored rollout is captured; rollouts are also written to
        outputs/ppo_rollouts/{run}.jsonl.
        """
        self.rm, self.tok, self.meta = reward_model, tok, prompt_meta    # meta: tuple(prompt ids) -> (symbol, snapshot)
        self.buf, self.facts = [], {}
        self.path = ROOT / "outputs" / "ppo_rollouts" / f"{run}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._get_reward = _ppo.get_reward
        _ppo.get_reward = self._wrapped

    def _wrapped(self, model, query_responses, pad_token_id, context_length):
        """Call the original get_reward and, for the reward model, decode each response and keep its score."""
        out = self._get_reward(model, query_responses, pad_token_id, context_length)
        if model is self.rm:
            eos = self.tok.eos_token_id
            for ids, sc in zip(query_responses.tolist(), out[1].float().tolist()):
                q = tuple(t for t in ids[:context_length] if t != pad_token_id)    # queries are left-padded
                resp = [t for t in ids[context_length:] if t != pad_token_id]     # TRL pads after the stop token
                has_eos = eos in resp
                n = resp.index(eos) + 1 if has_eos else len(resp)
                self.buf.append({"key": self.meta.get(q), "tokens": n, "eos": has_eos, "rm_score": sc,
                                 "text": self.tok.decode(resp[:n], skip_special_tokens=True).strip()})
        return out

    def on_log(self, args, state, control, logs=None, **kw):
        """Score the update's rollouts with the gold rubric and add rollout/* statistics to the logged row."""
        if not logs or "objective/kl" not in logs or not self.buf:
            return
        rows, self.buf = self.buf, []
        golds = []
        for r in rows:
            if r["key"] is None:
                continue
            if r["key"] not in self.facts:
                self.facts[r["key"]] = load_meta(*r["key"])
            facts, prior_year = self.facts[r["key"]]
            g = auto_rubric(r["text"], facts, r["tokens"], not r["eos"], prior_year)
            r.update(gold=g["gold"], severe=g["severe"], truncated=g["truncated"], gates=g["gates"])
            golds.append(r)
        n = len(rows)
        logs["rollout/tokens_mean"] = sum(r["tokens"] for r in rows) / n
        logs["rollout/eos_rate"] = sum(r["eos"] for r in rows) / n
        logs["rollout/over_500_rate"] = sum(r["tokens"] > 500 for r in rows) / n
        if golds:
            logs["rollout/gold_mean"] = sum(r["gold"] for r in golds) / len(golds)
            logs["rollout/severe_rate"] = sum(r["severe"] for r in golds) / len(golds)
            logs["rollout/truncated_rate"] = sum(r["truncated"] for r in golds) / len(golds)
            logs["rollout/words_mean"] = sum(len(r["text"].split()) for r in golds) / len(golds)
        logs["rollout/n_unmatched"] = n - len(golds)
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in rows:
                sym, snap = r.pop("key") or (None, None)
                fh.write(json.dumps({"step": state.global_step, "symbol": sym, "snapshot": snap, **r},
                                    ensure_ascii=False) + "\n")


class SaveAdapter(TrainerCallback):
    """Save the policy LoRA every N updates to {output_dir}/step_{N} (checkpoints for proxy-vs-gold curves).
    PPOTrainer.save_model / _save_checkpoint fail with transformers 5.x (missing Trainer.__init__ attributes)."""

    def __init__(self, every: int, output_dir: str):
        """Save every `every` updates into output_dir/step_N."""
        self.every, self.output_dir = every, Path(output_dir)

    def on_step_end(self, args, state, control, model=None, **kw):
        """Save the policy LoRA when the update number is a multiple of `every`."""
        if state.global_step % self.every == 0:
            model.policy.save_pretrained(self.output_dir / f"step_{state.global_step}")


def prompt_ids(tok, splits, fmt="markdown", prompt_id="guided_v2"):
    """Tokenised PPO prompts (guided_v2 on markdown cards) for every company of the splits x (T1, T2)."""
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as fh:
        syms = [r["symbol"] for r in csv.DictReader(fh) if r["split"] in splits]
    rows = []
    for sym in syms:
        for snap in ("T1", "T2"):
            card = ROOT / "data" / "cards" / f"{sym}_{snap}_{fmt}.txt"
            if card.exists():
                msgs = PROMPTS[prompt_id](card.read_text(encoding="utf-8"), fmt)
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)
                rows.append({"input_ids": ids["input_ids"], "symbol": sym, "snapshot": snap})
    return rows


def main():
    """Load policy (SFT merged + new LoRA), frozen reward model and value model, then run PPO and save the adapter."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sft_adapter", help="LoRA from train_sft.py, merged into the base before PPO")
    ap.add_argument("--reward_model", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", nargs="+", default=["C_ppo"])
    ap.add_argument("--longest", type=int, help="smoke test: use only the N longest prompts (from A/B/C splits)")
    ap.add_argument("--steps", type=int, default=150, help="PPO updates (each = batch_size x grad_accum rollouts)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="micro-batch for the PPO forward/backward (2 OOMs at ~2.4k tokens on 16 GB)")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--rollout_batch", type=int, default=4, help="sequences per generation / scoring forward")
    ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--response_length", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--missing_eos_penalty", type=float, default=1.0)
    ap.add_argument("--kl_coef", type=float, default=0.05, help="initial value")
    ap.add_argument("--kl_target", type=float, default=6.0, help="adaptive KL target (nats per response); 0 = fixed")
    ap.add_argument("--kl_horizon", type=int, default=10000)
    ap.add_argument("--value_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--save_every", type=int, default=0, help="save the policy adapter every N updates")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    args = ap.parse_args()
    guard_vram()

    # ---- policy: base + merged SFT LoRA, neutral sampling settings, new LoRA for PPO
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.sft_adapter:
        policy = PeftModel.from_pretrained(policy, args.sft_adapter).merge_and_unload()
    # Qwen ships top_k=20, top_p=0.8, repetition_penalty=1.1, and transformers fills generation settings that TRL leaves
    # at library defaults (e.g. top_p=1.0) from it. TRL takes rollout logprobs from the PROCESSED generation scores,
    # so these filters bias every logprob vs the reference: the first smoke test showed KL ~95 nats at update 1, when
    # policy == reference. Plain temperature sampling keeps rollout logprobs = true policy logprobs.
    policy.generation_config.update(top_k=0, top_p=1.0, repetition_penalty=1.0, temperature=1.0, do_sample=True)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")

    def load_rm(dtype):
        """Load the reward model checkpoint as a scalar-output classifier (used for both reward and value model)."""
        m = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, dtype=dtype,
                                                               attn_implementation="sdpa")
        m.config.pad_token_id = tok.pad_token_id
        return m
    # ---- reward model (frozen) and value model (trained), both from the reward-model checkpoint
    reward_model = load_rm(torch.bfloat16).eval().requires_grad_(False)
    value_model = load_rm(getattr(torch, args.value_dtype))
    if not args.no_grad_ckpt:
        # without it the policy backward at ~2.4k tokens needs >20 GB of activations (first smoke test spilled)
        for m in (policy, value_model):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- prompts
    if args.longest:
        rows = prompt_ids(tok, ["A_train", "A_val", "B_train", "B_test", "C_ppo", "C_eval"])
        rows = sorted(rows, key=lambda r: -len(r["input_ids"]))[:args.longest]
    else:
        rows = prompt_ids(tok, args.split)
    train = Dataset.from_list(rows).remove_columns(["symbol", "snapshot"])
    eval_ds = train.select(range(min(4, len(train))))  # PPOTrainer builds an eval loader even if unused

    # ---- PPO configuration, logging callbacks and training
    rollouts = args.batch_size * args.grad_accum
    cfg = PPOConfig(
        output_dir=str(MODELS / args.run),
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        num_mini_batches=1, num_ppo_epochs=args.ppo_epochs,
        total_episodes=args.steps * rollouts, local_rollout_forward_batch_size=args.rollout_batch,
        learning_rate=args.lr, response_length=args.response_length, temperature=args.temperature,
        stop_token="eos", missing_eos_penalty=args.missing_eos_penalty,
        kl_coef=args.kl_coef, kl_estimator="k1",
        bf16=True, logging_steps=1, num_sample_generations=0,
        save_strategy="no",  # TRL's own saving crashes with transformers 5.x; SaveAdapter below saves instead
        report_to="none", seed=args.seed,
    )
    settings = {k: v for k, v in vars(args).items()} | {"n_prompts": len(train), "rollouts_per_update": rollouts}
    logger = StepLogger(args.run, "ppo", settings)
    stats = RolloutStats(reward_model, tok, {tuple(r["input_ids"]): (r["symbol"], r["snapshot"]) for r in rows}, args.run)
    callbacks = [stats, logger] + ([AdaptiveKL(args.kl_target, args.kl_horizon)] if args.kl_target > 0 else [])
    if args.save_every:
        callbacks.append(SaveAdapter(args.save_every, cfg.output_dir))
    trainer = PPOTrainer(args=cfg, processing_class=tok, model=policy, ref_model=None, reward_model=reward_model,
                         value_model=value_model, train_dataset=train, eval_dataset=eval_ds,
                         peft_config=peft_config, callbacks=callbacks)
    summary = run_guarded(trainer, logger)
    trainer.accelerator.unwrap_model(trainer.model).policy.save_pretrained(cfg.output_dir)
    print(f"peak reserved {summary['peak_reserved_gb']} GB | {summary['mean_step_seconds_after_first']} s/update "
          f"-> {cfg.output_dir}")


if __name__ == "__main__":
    main()
