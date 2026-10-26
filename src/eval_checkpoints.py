"""Evaluate base, SFT and every PPO checkpoint on held-out prompts (greedy).

Checkpoints: base (Qwen2.5-1.5B-Instruct), sft (base + --sft_adapter merged), and each {ppo_dir}/step_N
(base + SFT merged + PPO LoRA merged; the PPO LoRA was trained on the SFT-merged model), plus the final adapter in
{ppo_dir}/ if it is not already one of the step_N folders.
Prompts: guided_v2 on markdown cards of --split (default C_eval + B_test = 35 companies x T1/T2 = 70 prompts).
Each output is scored with one or more reward models (--reward_model, first = primary = the RM the run trained on)
and with the gated gold (auto_rubric; PPO never sees it).

    python src/eval_checkpoints.py --run ppo_v1 --sft_adapter models/sft_v1 --reward_model models/rm_v2
    python src/eval_checkpoints.py --run ppo_v1 --only base sft --limit 4      # smoke test
    # ablation trained on rm_v1, scored by both RMs; base/SFT generations copied from ppo_v1 (greedy, identical)
    python src/eval_checkpoints.py --run ppo_rmv1 --reward_model models/rm_v1 models/rm_v2 --reuse_from ppo_v1

Outputs (resumable: a checkpoint whose jsonl already has every prompt is not regenerated, only re-scored):
  outputs/eval_runs/{run}/{checkpoint}.jsonl            one line per prompt (text, tokens, reward, gold, flags)
  outputs/train_logs/{run}_checkpoints.csv              one row per checkpoint (rewritten each time)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_rubric import auto_rubric  # noqa: E402
from gold_check import load_meta  # noqa: E402
from prompts import PROMPTS  # noqa: E402
from train_rm import score  # noqa: E402
from train_utils import LOGS, MODELS, ROOT  # noqa: E402


def rel(path) -> str:
    """Path relative to the repository root (forward slashes), so saved outputs carry no local directory names."""
    p = Path(path)
    return (p.relative_to(ROOT) if p.is_absolute() and p.is_relative_to(ROOT) else p).as_posix()


def prompts_for(splits, fmt, prompt_id):
    """Evaluation prompts: the guided_v2 prompt on each company's card, for every company of the given splits x (T1, T2)."""
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as fh:
        rows = [(r["split"], r["symbol"]) for r in csv.DictReader(fh) if r["split"] in splits]
    out = []
    for split, sym in sorted(rows):
        for snap in ("T1", "T2"):
            card = ROOT / "data" / "cards" / f"{sym}_{snap}_{fmt}.txt"
            if card.exists():
                out.append({"split": split, "symbol": sym, "snapshot": snap,
                            "messages": PROMPTS[prompt_id](card.read_text(encoding="utf-8"), fmt)})
    return out


def checkpoints(args):
    """List (name, adapters) to evaluate: base, sft, every saved step_N of the PPO run, and its final adapter if new."""
    ppo = MODELS / args.run
    cks = [("base", []), ("sft", [args.sft_adapter])]
    steps = sorted((p for p in ppo.glob("step_*") if (p / "adapter_config.json").exists()),
                   key=lambda p: int(p.name.split("_")[1])) if ppo.exists() else []
    cks += [(p.name, [args.sft_adapter, str(p)]) for p in steps]
    summary = LOGS / f"{args.run}_summary.json"
    done = json.loads(summary.read_text(encoding="utf-8")).get("steps_done") if summary.exists() else None
    if (ppo / "adapter_config.json").exists() and not (ppo / f"step_{done}").exists():
        cks.append(("final", [args.sft_adapter, str(ppo)]))
    return [c for c in cks if not args.only or c[0] in args.only]


def load_policy(model_name, adapters):
    """Load the base model and merge the given LoRA adapters in order (SFT first, then PPO)."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    for a in adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    return model.eval()


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size, repetition_penalty):
    """Greedy generation for all prompts (batched, prompts sorted by length); returns text, token count and whether the
    cap was hit, in the original prompt order.
    """
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                  else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
              pad_token_id=tok.pad_token_id)
    if repetition_penalty is not None:
        kw["repetition_penalty"] = repetition_penalty
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
        for i, ids in zip(idx, out.tolist()):
            n = next((j for j, t in enumerate(ids) if t in eos_ids), len(ids))
            res[i] = {"output": tok.decode(ids[:n], skip_special_tokens=True).strip(), "output_tokens": n,
                      "hit_max_tokens": n >= max_new_tokens}
        print(f"    {min(b + batch_size, len(order))}/{len(order)}", flush=True)
    return res


def summarise(name, rows):
    """One summary row per checkpoint: mean rewards, gold, severe / over-500 / truncation rates and length."""
    n = len(rows)
    mean = lambda k: round(sum(float(r[k]) for r in rows) / n, 4)  # noqa: E731
    rms = [k for k in rows[0] if k.startswith("reward_")]
    out = {"checkpoint": name, "step": 0 if name in ("base", "sft") else
           int(m.group(1)) if (m := re.match(r"step_(\d+)", name)) else None, "n": n,
           "rm_reward_mean": mean("reward"), **{f"{k}_mean": mean(k) for k in rms},
           "gold_mean": mean("gold"), "composite_mean": mean("composite"),
           "severe_rate": mean("severe"), "over_500_rate": mean("over_500"), "truncated_rate": mean("truncated"),
           "hit_max_rate": mean("hit_max_tokens"), "tokens_mean": mean("output_tokens"), "words_mean": mean("words"),
           "stuffed_rate": round(sum("stuffed" in r["gates"] for r in rows) / n, 4),
           "non_answer_rate": round(sum("non_answer" in r["gates"] for r in rows) / n, 4)}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        out[f"gold_mean_{split}"] = round(sum(r["gold"] for r in rs) / len(rs), 4)
        for k in rms:
            out[f"{k}_mean_{split}"] = round(sum(r[k] for r in rs) / len(rs), 4)
    return out


def main():
    """Generate (or reuse) greedy outputs for every checkpoint, score them with the reward model(s) and gold, and write
    per-prompt outputs and the per-checkpoint table.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_v1", help="PPO run name: adapters in models/{run}/step_N")
    ap.add_argument("--sft_adapter", default="models/sft_v1")
    ap.add_argument("--reward_model", nargs="+", default=["models/rm_v2"],
                    help="RM(s) to score with; the first is the primary (rm_reward_mean / reward)")
    ap.add_argument("--reuse_from", help="copy base/sft generations from outputs/eval_runs/{this run} if missing here")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", nargs="+", default=["C_eval", "B_test"])
    ap.add_argument("--format", default="markdown")
    ap.add_argument("--prompt", default="guided_v2")
    ap.add_argument("--only", nargs="*", help="checkpoint names to evaluate (base, sft, step_25, final, ...)")
    ap.add_argument("--limit", type=int, help="first N prompts only (smoke test; written to a separate folder)")
    ap.add_argument("--max_new_tokens", type=int, default=600, help="above 500 on purpose, to measure overruns")
    ap.add_argument("--repetition_penalty", type=float,
                    help="default: the model's generation_config (Qwen 1.1), as in run_base_inference.py")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--rm_max_length", type=int, default=2560)
    args = ap.parse_args()

    # ---- prompts, output folder and reward models
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    prompts = prompts_for(args.split, args.format, args.prompt)[:args.limit]
    tag = args.run + (f"_limit{args.limit}" if args.limit else "")
    out_dir = ROOT / "outputs" / "eval_runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {(p["symbol"], p["snapshot"]): load_meta(p["symbol"], p["snapshot"]) for p in prompts}
    rms = {}
    for path in args.reward_model:
        rm_tok = AutoTokenizer.from_pretrained(path)
        rm = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.bfloat16,
                                                                attn_implementation="sdpa").cuda().eval()
        rm.config.pad_token_id = rm_tok.pad_token_id
        rms[Path(path).name] = (rm, rm_tok)

    # ---- every checkpoint: generate (or reuse) outputs, then score them with the reward model(s) and gold
    cks = checkpoints(args)
    print(f"{len(prompts)} prompts x {len(cks)} checkpoints: {[c[0] for c in cks]}")
    table = []
    for name, adapters in cks:
        path = out_dir / f"{name}.jsonl"
        src = path
        if not path.exists() and args.reuse_from and name in ("base", "sft"):
            src = ROOT / "outputs" / "eval_runs" / args.reuse_from / f"{name}.jsonl"
        old = [json.loads(line) for line in open(src, encoding="utf-8")] if src.exists() else []
        if len(old) == len(prompts):
            gens = [{k: g[k] for k in ("output", "output_tokens", "hit_max_tokens")} for g in old]
            print(f"{name}: reusing {src.relative_to(ROOT)}")
        else:
            t0 = time.time()
            print(f"{name}: generating ({' + '.join(adapters) or 'no adapter'})")
            model = load_policy(args.model, adapters)
            gens = generate(model, tok, prompts, args.max_new_tokens, args.batch_size, args.repetition_penalty)
            del model
            torch.cuda.empty_cache()
            print(f"  {time.time() - t0:.0f} s")
        convs = [p["messages"] + [{"role": "assistant", "content": g["output"]}] for p, g in zip(prompts, gens)]
        rewards = {k: score(m, t, convs, args.rm_max_length) for k, (m, t) in rms.items()}
        primary = next(iter(rms))
        rows = []
        for i, (p, g) in enumerate(zip(prompts, gens)):
            facts, prior_year = meta[(p["symbol"], p["snapshot"])]
            r = auto_rubric(g["output"], facts, g["output_tokens"], g["hit_max_tokens"], prior_year)
            rows.append({"checkpoint": name, "adapters": [rel(a) for a in adapters], "split": p["split"], "symbol": p["symbol"],
                         "snapshot": p["snapshot"], **g, "reward": round(rewards[primary][i], 4),
                         **{f"reward_{k}": round(v[i], 4) for k, v in rewards.items()}, "gold": r["gold"],
                         "composite": r["composite"], "severe": r["severe"], "over_500": r["over_500"],
                         "truncated": r["truncated"], "gates": r["gates"], "words": r["words"],
                         "n_unsupported": r["n_unsupported"], "dir_errors": len(r.get("direction_errors") or [])})
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        table.append(summarise(name, rows))
        t = table[-1]
        print("  " + " | ".join(f"{k[:-5]} {t[k]:+.3f}" for k in t if k.startswith("reward_") and k.endswith("_mean"))
              + f" | gold {t['gold_mean']:.3f} | severe {t['severe_rate']:.2f} | "
              f">500 {t['over_500_rate']:.2f} | trunc {t['truncated_rate']:.2f} | tokens {t['tokens_mean']:.0f}")

    # ---- one row per checkpoint
    LOGS.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS / f"{tag}_checkpoints.csv"
    cols = list(dict.fromkeys(k for t in table for k in t))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"-> {csv_path}\n-> {out_dir}")


if __name__ == "__main__":
    main()
