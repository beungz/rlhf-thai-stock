"""Bradley-Terry reward model (Qwen2.5-0.5B-Instruct, full fine-tune, scalar head) with margins.

Input: jsonl with {"chosen": [...], "rejected": [...], "margin": float, "item_id": ...} (aggregate_labels.py --mode rm).
Loss = -log sigmoid(r_chosen - r_rejected - margin)  (+ small reward-centering term).
The saved model is also the initial value model for PPO (same tokenizer as the Qwen policy).

Tokenization: chat template, then everything after the final <|im_end|> is dropped (the template adds "\\n"), so the
reward is read at <|im_end|> - the same token PPO reads it at (TRL truncates the response after the stop token).

After training (or with --no_train for an existing RM) it evaluates:
  test    pairwise accuracy on --test (default data/rm/B_test.jsonl), overall and by label margin, plus a
          calibration table: p = sigmoid(|r_chosen - r_rejected|) = the RM's confidence in the response it prefers,
          5 equal-width bins on [0.5, 1], empirical accuracy per bin (and ECE)
  candidates  every --candidates generation for the --test prompts (gen_B_sft on B_test: 30 prompts x 4):
          Spearman of reward vs response tokens, vs gated gold (auto_rubric) and vs mean judge overall
  probes  mean reward per probe type on --probes (default data/rm/probes_B_test.jsonl, src/make_probes.py),
          how often `correct` scores above each probe type on the same prompt, and reward-vs-gold rank correlation
and writes everything to outputs/train_logs/{run}.json.

    # 10-step test on 25% of the prompts
    python src/train_rm.py --run rm_test --train data/rm/B_train.jsonl --test_limit 40 --train_frac 0.25 --max_steps 10
    # full run + learning curve
    python src/train_rm.py --run rm_v1 --train data/rm/B_train.jsonl --eval data/rm/B_val.jsonl --test data/rm/B_test.jsonl --epochs 2
    python src/train_rm.py --run rm_lc25 --train ... --train_frac 0.25        (0.5, 0.75; same --seed -> nested subsets)
    # re-evaluate a trained RM only
    python src/train_rm.py --run rm_v1_probes2 --model models/rm_v1 --no_train

Model -> models/{run}/ ; logs -> outputs/train_logs/{run}_steps.csv, {run}_summary.json, {run}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_rubric import auto_rubric  # noqa: E402
from gold_check import load_meta  # noqa: E402
from prompts import PROMPTS  # noqa: E402
from train_utils import LOGS, MODELS, ROOT, StepLogger, guard_vram, run_guarded, versions  # noqa: E402


def rel(path) -> str:
    """Path relative to the repository root (forward slashes), so logs carry no local directory names."""
    p = Path(path).resolve()
    return (p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).as_posix()


# ---- tokenisation, data subsets and scoring
def encode(tok, messages):
    """Chat-template token ids ending at the final <|im_end|> (drops the template's trailing newline)."""
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
    last = max(i for i, t in enumerate(ids) if t == tok.eos_token_id)
    return ids[:last + 1]


def select_frac(ds, frac, unit, seed):
    """Nested subsets: the same seed gives 25% c 50% c 75% c 100%. unit=prompt keeps whole prompts (all their pairs)."""
    if frac >= 1:
        return ds
    if unit == "prompt" and "item_id" in ds.column_names:
        items = sorted(set(ds["item_id"]))
        random.Random(seed).shuffle(items)
        keep = set(items[:max(1, round(frac * len(items)))])
        return ds.filter(lambda ex: ex["item_id"] in keep)
    return ds.shuffle(seed=seed).select(range(max(1, round(frac * len(ds)))))


@torch.no_grad()
def score(model, tok, conversations, max_length):
    """Scalar reward for each conversation, one at a time (no padding, so the head reads the last token)."""
    out = []
    for msgs in conversations:
        ids = encode(tok, msgs)[-max_length:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(input_ids=torch.tensor([ids], device=model.device)).logits[0, 0].float().item())
    return out


def spearman(a, b):
    """Spearman rank correlation of two equal-length lists (average ranks for ties); None if undefined."""
    def ranks(x):
        """Rank positions of `x`, giving tied values their average rank."""
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):                       # average ranks for ties
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(cov / var, 3) if var else None


# ---- evaluations after training: test pairs, calibration, candidates, probes
def eval_test(model, tok, path, limit, max_length):
    """Pairwise accuracy of the RM on held-out pairs (overall, by label margin) plus a calibration table."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")][:limit]
    rc = score(model, tok, [r["chosen"] for r in rows], max_length)
    rr = score(model, tok, [r["rejected"] for r in rows], max_length)
    win = [1.0 if c > r else 0.5 if c == r else 0.0 for c, r in zip(rc, rr)]
    res = {"file": rel(path), "n_pairs": len(rows), "accuracy": round(sum(win) / len(win), 4),
           "mean_reward_gap": round(sum(c - r for c, r in zip(rc, rr)) / len(rows), 4)}
    if rows and "margin" in rows[0]:
        for name, cond in (("margin_lt_1", lambda m: m < 1), ("margin_ge_1", lambda m: m >= 1)):
            w = [x for x, r in zip(win, rows) if cond(r["margin"])]
            res[f"accuracy_{name}"] = {"n": len(w), "accuracy": round(sum(w) / len(w), 4) if w else None}
    res["calibration"] = calibration(rc, rr, win)
    return res


def calibration(rc, rr, win, n_bins=5):
    """Bin p = sigmoid(|dr|) (confidence in the RM's preferred response, 0.5-1) and compare with how often that
    preference agrees with the label. Ties (p = 0.5) count as half right, as in the accuracy."""
    conf = [1 / (1 + math.exp(-abs(c - r))) for c, r in zip(rc, rr)]
    bins, ece = [], 0.0
    for b in range(n_bins):
        lo, hi = 0.5 + 0.5 * b / n_bins, 0.5 + 0.5 * (b + 1) / n_bins
        idx = [i for i, p in enumerate(conf) if lo <= p < hi or (b == n_bins - 1 and p == hi)]
        acc = sum(win[i] for i in idx) / len(idx) if idx else None
        mp = sum(conf[i] for i in idx) / len(idx) if idx else None
        if idx:
            ece += len(idx) / len(conf) * abs(acc - mp)
        bins.append({"bin": f"[{lo:.1f},{hi:.1f})" if b < n_bins - 1 else f"[{lo:.1f},{hi:.1f}]", "n": len(idx),
                     "mean_confidence": None if mp is None else round(mp, 4),
                     "accuracy": None if acc is None else round(acc, 4)})
    return {"bins": bins, "ece": round(ece, 4) if conf else None}


def eval_candidates(model, tok, gen_path, test_path, max_length):
    """Reward of every generated candidate for the test prompts vs its length, gated gold and judge overall."""
    items = {json.loads(line)["item_id"].rsplit("_", 1)[0] for line in open(test_path, encoding="utf-8")}
    judge = {}
    jpath = ROOT / "data" / "rm" / "candidates_B.csv"
    if jpath.exists():
        with open(jpath, encoding="utf-8") as fh:
            judge = {(r["item_id"].rsplit("_", 1)[0], r["cand_id"]): float(r["overall"]) for r in csv.DictReader(fh)
                     if r["overall"]}
    rows, meta = [], {}
    for line in open(gen_path, encoding="utf-8"):
        g = json.loads(line)
        k = (g["symbol"], g["snapshot"])
        if f"{k[0]}_{k[1]}" not in items:
            continue
        if k not in meta:
            meta[k] = load_meta(*k)
        card = (ROOT / "data" / "cards" / f"{k[0]}_{k[1]}_{g['format']}.txt").read_text(encoding="utf-8")
        cand_id = f"{g['model']}|{g['prompt_id']}|{g['format']}|{g['decoding']}{g['sample_idx']}"
        r = auto_rubric(g["output"], meta[k][0], g["output_tokens"], g["hit_max_tokens"], meta[k][1])
        rows.append({"item_id": f"{k[0]}_{k[1]}", "cand_id": cand_id, "tokens": g["output_tokens"],
                     "gold": r["gold"], "severe": r["severe"], "judge_overall": judge.get((f"{k[0]}_{k[1]}", cand_id)),
                     "messages": PROMPTS[g["prompt_id"]](card, g["format"]) + [{"role": "assistant", "content": g["output"]}]})
    rewards = score(model, tok, [r.pop("messages") for r in rows], max_length)
    for r, x in zip(rows, rewards):
        r["reward"] = round(x, 4)
    rw = [r["reward"] for r in rows]
    jr = [r for r in rows if r["judge_overall"] is not None]
    return {"file": rel(gen_path), "n": len(rows), "n_prompts": len({r["item_id"] for r in rows}),
            "spearman_reward_vs_tokens": spearman(rw, [r["tokens"] for r in rows]),
            "spearman_reward_vs_gold": spearman(rw, [r["gold"] for r in rows]),
            "spearman_reward_vs_judge": spearman([r["reward"] for r in jr], [r["judge_overall"] for r in jr])
            if len(jr) > 2 else None, "n_with_judge": len(jr),
            "spearman_gold_vs_tokens": spearman([r["gold"] for r in rows], [r["tokens"] for r in rows]),
            "mean_reward_severe": round(sum(r["reward"] for r in rows if r["severe"]) / max(1, sum(r["severe"] for r in rows)), 4),
            "mean_reward_not_severe": round(sum(r["reward"] for r in rows if not r["severe"])
                                            / max(1, sum(not r["severe"] for r in rows)), 4),
            "per_candidate": rows}


def eval_probes(model, tok, path, max_length):
    """Score the probe suite: mean reward and gold per probe type, how often `correct` outscores each probe on the same
    prompt, and the reward-vs-gold rank correlation.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rewards = score(model, tok, [r["messages"] for r in rows], max_length)
    for r, x in zip(rows, rewards):
        r["reward"] = x
    correct = {r["item_id"]: r["reward"] for r in rows if r["probe"] == "correct"}
    by = defaultdict(list)
    for r in rows:
        by[r["probe"]].append(r)
    table = {}
    for p, rs in by.items():
        xs = [r["reward"] for r in rs]
        mean = sum(xs) / len(xs)
        paired = [r for r in rs if r["item_id"] in correct]
        table[p] = {"n": len(rs), "mean_reward": round(mean, 4),
                    "std_reward": round((sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5, 4),
                    "mean_gold": round(sum(r["gold"] for r in rs) / len(rs), 4),
                    "correct_beats_pct": None if p == "correct" else
                    round(100 * sum(correct[r["item_id"]] > r["reward"] for r in paired) / len(paired), 1)}
    return {"file": rel(path), "n": len(rows), "by_probe": table,
            "spearman_reward_vs_gold": spearman([r["reward"] for r in rows], [r["gold"] for r in rows]),
            "per_probe_rewards": [{"item_id": r["item_id"], "probe": r["probe"], "reward": round(r["reward"], 4),
                                   "gold": r["gold"]} for r in rows]}


def main():
    """Train the reward model (unless --no_train), then evaluate it on test pairs, probes and candidates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", nargs="+", help="one or more pair files (concatenated)")
    ap.add_argument("--eval", nargs="+", help="validation pair file(s) (concatenated), evaluated every epoch")
    ap.add_argument("--test", default="data/rm/B_test.jsonl", help="held-out pairs for final pairwise accuracy")
    ap.add_argument("--test_limit", type=int)
    ap.add_argument("--probes", default="data/rm/probes_B_test.jsonl")
    ap.add_argument("--candidates", default="outputs/base_runs/gen_B_sft.jsonl",
                    help="generations whose --test prompts are scored for reward-vs-length/gold correlations")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no_train", action="store_true", help="only evaluate --model (a trained RM)")
    ap.add_argument("--limit", type=int, help="use only the first N training pairs (smoke tests)")
    ap.add_argument("--train_frac", "--frac", dest="train_frac", type=float, default=1.0,
                    help="fraction of training data (learning curve); nested across fractions for a fixed --seed")
    ap.add_argument("--frac_unit", default="prompt", choices=["prompt", "pair"],
                    help="subsample whole prompts (default; labelling cost is per prompt) or individual pairs")
    ap.add_argument("--no_margin", action="store_true", help="plain Bradley-Terry loss")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=2, help="pairs per device step")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--center", type=float, default=0.01, help="center_rewards_coefficient")
    ap.add_argument("--max_length", type=int, default=2560,
                    help="TRL silently DROPS pairs longer than this; longest prompt+summary is ~2.4k tokens")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="master weights; float32 + bf16 autocast is the numerically safe default")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not args.no_train and not args.train:
        ap.error("--train is required unless --no_train")
    guard_vram()

    # ---- model: base (or trained RM) with a scalar head
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    model.config.pad_token_id = tok.pad_token_id

    report = {"run": args.run, "model": args.model, "settings": vars(args), "versions": versions()}
    # ---- training: pairs (optionally a nested fraction of prompts), tokenisation, RewardTrainer
    if not args.no_train:
        def load_pairs(paths):
            """Concatenate pair files, keeping only the columns training needs (synthetic files carry extra ones)."""
            keep = ("chosen", "rejected", "margin", "item_id")
            rows = [{k: r[k] for k in keep if k in r} for path in paths
                    for r in map(json.loads, open(path, encoding="utf-8")) if r]
            return Dataset.from_list(rows)
        ds = DatasetDict({"train": load_pairs(args.train)} | ({"eval": load_pairs(args.eval)} if args.eval else {}))
        n_all = len(ds["train"])
        ds["train"] = select_frac(ds["train"], args.train_frac, args.frac_unit, args.seed)
        if args.limit:
            ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
        n_prompts = len(set(ds["train"]["item_id"])) if "item_id" in ds["train"].column_names else None

        def tokenize(ex):
            """Token ids of the chosen and rejected conversations (ending at <|im_end|>) and the margin."""
            out = {"chosen_ids": encode(tok, ex["chosen"]), "rejected_ids": encode(tok, ex["rejected"])}
            if not args.no_margin:
                out["margin"] = float(ex["margin"])
            return out
        for split in list(ds):
            ds[split] = ds[split].map(tokenize, remove_columns=ds[split].column_names)

        cfg = RewardConfig(
            output_dir=str(MODELS / args.run),
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,  # transformers 5: float < 1 = ratio
            max_length=args.max_length, center_rewards_coefficient=args.center,
            gradient_checkpointing=not args.no_grad_ckpt, bf16=True,
            logging_steps=1, eval_strategy="epoch" if args.eval else "no",
            save_strategy="no", report_to="none", seed=args.seed,
        )
        n_in = len(ds["train"])
        settings = vars(args) | {"n_train_pairs_all": n_all, "n_train_pairs": n_in, "n_train_prompts": n_prompts}
        logger = StepLogger(args.run, "rm", settings)
        trainer = RewardTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds.get("eval"),
                                processing_class=tok, callbacks=[logger])
        n_kept = len(trainer.train_dataset)
        if n_kept < n_in:
            print(f"WARNING: {n_in - n_kept} pairs dropped for exceeding max_length={args.max_length}")
        logger.settings["n_train_after_length_filter"] = n_kept
        summary = run_guarded(trainer, logger)
        trainer.save_model(cfg.output_dir)
        tok.save_pretrained(cfg.output_dir)
        hist = trainer.state.log_history
        report.update({
            "n_train_pairs_all": n_all, "n_train_pairs": n_kept, "n_train_prompts": n_prompts,
            "train_curve": [{k: h.get(k) for k in ("step", "epoch", "loss", "accuracy", "margin", "mean_reward")}
                            for h in hist if "loss" in h],
            "eval_curve": [{k: v for k, v in h.items() if k in ("step", "epoch") or k.startswith("eval_")}
                           for h in hist if "eval_loss" in h],
            "peak_reserved_gb": summary["peak_reserved_gb"], "mean_step_seconds": summary["mean_step_seconds_after_first"],
            "total_seconds": summary["total_seconds"],
        })
        print(f"peak reserved {summary['peak_reserved_gb']} GB | {summary['mean_step_seconds_after_first']} s/step "
              f"-> {cfg.output_dir}")

    # ---- evaluation, report file and a short printout
    model.cuda().eval()
    for key, path, fn in (("test", args.test, lambda p: eval_test(model, tok, p, args.test_limit, args.max_length)),
                          ("probes", args.probes, lambda p: eval_probes(model, tok, p, args.max_length)),
                          ("candidates", args.candidates,
                           lambda p: eval_candidates(model, tok, p, ROOT / args.test, args.max_length))):
        p = ROOT / path
        if p.exists():
            report[key] = fn(p)
        else:
            print(f"NOTE: {path} not found -> no {key} evaluation")
            report[key] = None
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{args.run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["test"]:
        t = report["test"]
        print(f"test pairwise accuracy {t['accuracy']:.3f} on {t['n_pairs']} pairs (mean gap {t['mean_reward_gap']:+.3f})")
        print(f"calibration (ECE {t['calibration']['ece']}): " + "  ".join(
            f"{b['bin']} n={b['n']} acc={b['accuracy']}" for b in t["calibration"]["bins"]))
    if report["candidates"]:
        c = report["candidates"]
        print(f"candidates ({c['n']}): spearman reward vs tokens {c['spearman_reward_vs_tokens']}, vs gold "
              f"{c['spearman_reward_vs_gold']}, vs judge {c['spearman_reward_vs_judge']}")
    if report["probes"]:
        print(f"{'probe':22}{'reward':>9}{'gold':>7}{'correct>':>10}")
        for p, s in report["probes"]["by_probe"].items():
            print(f"{p:22}{s['mean_reward']:9.3f}{s['mean_gold']:7.3f}"
                  f"{'' if s['correct_beats_pct'] is None else str(s['correct_beats_pct']) + '%':>10}")
        print(f"spearman(reward, gold) over probes: {report['probes']['spearman_reward_vs_gold']}")
    print(f"-> {LOGS / (args.run + '.json')}")


if __name__ == "__main__":
    main()
