"""Generate summaries with base (instruct) models - or a fine-tuned LoRA adapter - on the data cards (needs a CUDA GPU).

Examples (from the repository root):
    # Experiment 1 - which card format works best (greedy decoding, one prompt)
    python src/run_base_inference.py --run exp1_formats --models qwen1.5b gemma1b --formats text markdown json rawjson --prompts guided

    # Experiment 2 - which prompt works best (best format from exp 1, plus 2 sampled outputs each)
    python src/run_base_inference.py --run exp2_prompts --models qwen1.5b gemma1b --formats markdown --prompts minimal guided concise --samples 2

    # A fine-tuned policy: a PEFT LoRA (e.g. from train_sft.py) is merged into the base model at load time.
    # Records get model = "{model}+{adapter folder name}" (e.g. qwen1.5b+sft_v1) and an "adapter" field.
    python src/run_base_inference.py --run gen_B_sft --adapter models/sft_v1 --split B_train B_test --formats markdown --prompts guided_v2 --samples 3 --temperature 0.8

Then score:
    python src/gold_check.py outputs/base_runs/exp1_formats.jsonl

Output: outputs/base_runs/{run}.jsonl (one line per generation; re-running skips what is already done).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "data" / "cards"
OUT = ROOT / "outputs" / "base_runs"

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "gemma1b": "google/gemma-3-1b-it",          # gated: accept the licence on Hugging Face, then `hf auth login`
    "gemma270m": "google/gemma-3-270m-it",
}


def load(model_key: str, dtype, adapter: str | None = None):
    """Load a model by short key (or Hugging Face id) with its tokenizer (left padding); merge a LoRA adapter if given."""
    name = MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return name, tok, model


def main():
    """Generate greedy and/or sampled summaries for every (model, prompt, format, company, snapshot) not done yet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name of the output file")
    ap.add_argument("--models", nargs="+", default=["qwen1.5b"])
    ap.add_argument("--formats", nargs="+", default=["text", "markdown", "json", "rawjson"])
    ap.add_argument("--prompts", nargs="+", default=["guided"])
    ap.add_argument("--snapshots", nargs="+", default=["T1", "T2"])
    ap.add_argument("--symbols", nargs="*", help="default: every company with cards")
    ap.add_argument("--split", nargs="*", help="take companies from data/splits.csv, e.g. --split A_train A_val")
    ap.add_argument("--samples", type=int, default=0, help="sampled outputs per prompt, in addition to one greedy")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=600, help="above 500 on purpose, to measure overruns")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adapter", help="PEFT LoRA folder (e.g. models/sft_v1), merged into the base model; one --models only")
    args = ap.parse_args()
    if args.adapter and len(args.models) != 1:
        sys.exit("--adapter needs exactly one --models (the adapter's base model)")

    # ---- device, output file and the generations already done (resume)
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available - check that PyTorch was installed with CUDA (see README).")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.run}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                done.add((g["model"], g["prompt_id"], g["format"], g["symbol"], g["snapshot"], g["decoding"], g["sample_idx"]))

    # ---- companies and decodings to run
    if args.split:
        import csv
        with open(ROOT / "data" / "splits.csv", encoding="utf-8") as fh:
            symbols = sorted(r["symbol"] for r in csv.DictReader(fh) if r["split"] in args.split)
        print(f"{len(symbols)} companies from splits {args.split}")
    else:
        symbols = args.symbols or sorted({p.name.split("_")[0] for p in CARDS.glob("*_T*_*.txt")})
    decodings = ([] if args.no_greedy else [("greedy", 0)]) + [("sample", i) for i in range(args.samples)]

    # ---- per model: build the prompt list, batch it, generate and append one line per output
    for base_key in args.models:
        name, tok, model = load(base_key, dtype, args.adapter)
        mkey = f"{base_key}+{Path(args.adapter).name}" if args.adapter else base_key
        print(f"\nLoaded {name}{' + ' + args.adapter if args.adapter else ''}  "
              f"(GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB)")
        jobs = []
        for pid in args.prompts:
            for fmt in args.formats:
                for sym in symbols:
                    for snap in args.snapshots:
                        card_path = CARDS / f"{sym}_{snap}_{fmt}.txt"
                        if not card_path.exists():
                            continue
                        msgs = PROMPTS[pid](card_path.read_text(encoding="utf-8"), fmt)
                        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        for dec, idx in decodings:
                            key = (mkey, pid, fmt, sym, snap, dec, idx)
                            if key not in done:
                                jobs.append((key, prompt_text))
        # batch by decoding type, then by similar prompt length
        jobs.sort(key=lambda j: (j[0][5], len(j[1])))
        batches = []
        for dec in ("greedy", "sample"):
            group = [j for j in jobs if j[0][5] == dec]
            batches += [group[i:i + args.batch_size] for i in range(0, len(group), args.batch_size)]
        gc_eos = model.generation_config.eos_token_id
        eos_ids = set(gc_eos if isinstance(gc_eos, list) else [gc_eos]) | {tok.eos_token_id}
        print(f"{len(jobs)} generations to run")
        t_all, n_done = time.time(), 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for bi, batch in enumerate(batches):
                dec = batch[0][0][5]
                enc = tok([j[1] for j in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                if dec == "sample":
                    torch.manual_seed(args.seed + bi)
                    gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
                else:
                    gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**enc, **gen_kwargs)
                secs = time.time() - t0
                new = out[:, enc["input_ids"].shape[1]:]
                for (key, _), ids, mask in zip(batch, new, enc["attention_mask"]):
                    ids = ids.tolist()
                    n_out = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
                    text = tok.decode(ids[:n_out], skip_special_tokens=True).strip()
                    rec = dict(zip(["model", "prompt_id", "format", "symbol", "snapshot", "decoding", "sample_idx"], key))
                    rec.update(model_name=name, adapter=args.adapter, prompt_tokens=int(mask.sum()), output_tokens=n_out,
                               hit_max_tokens=n_out >= args.max_new_tokens, gen_seconds=round(secs / len(batch), 2),
                               temperature=args.temperature if dec == "sample" else 0.0, output=text)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += len(batch)
                print(f"  {n_done}/{len(jobs)}  ({secs:.1f}s this batch, {(time.time() - t_all) / 60:.1f} min total)")
        del model
        torch.cuda.empty_cache()
    print(f"\nDone -> {out_path}\nScore it with:  python src/gold_check.py {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
