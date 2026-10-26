"""Turn model generations into judging items (one data card + K candidate summaries each).

    python src/make_judge_items.py outputs/base_runs/exp1_formats.jsonl --out data/judge_items/pilot.jsonl --k 4

Candidates for an item are all generations for the same company + snapshot (+ prompt, by default),
across models / formats / samples. If there are more than K, K are drawn at random (fixed seed).
The judge always sees the plain-text card, whatever format the policy model was given.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "data" / "cards"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("generations", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--group-by-prompt", action="store_true", help="only compare outputs of the same prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", help="keep only these models, e.g. qwen1.5b")
    ap.add_argument("--prompts", nargs="*", help="keep only these prompt ids")
    ap.add_argument("--formats", nargs="*", help="keep only these card formats")
    args = ap.parse_args()
    # ---- group generations by company x snapshot (x prompt), applying the filters
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for path in args.generations:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            if not g["output"].strip():
                continue
            if (args.models and g["model"] not in args.models) or (args.prompts and g["prompt_id"] not in args.prompts) \
                    or (args.formats and g["format"] not in args.formats):
                continue
            key = (g["symbol"], g["snapshot"]) + ((g["prompt_id"],) if args.group_by_prompt else ())
            groups[key].append(g)
    # ---- one judging item per group (or per chunk of K candidates), with the plain-text card
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for key, gens in sorted(groups.items()):
            if len(gens) < 2:
                continue
            rng.shuffle(gens)
            for start in range(0, len(gens) - 1, args.k):
                chunk = gens[start:start + args.k]
                if len(chunk) < 2:
                    break
                item = {"item_id": "_".join(key) + f"_{start // args.k}", "symbol": key[0], "snapshot": key[1],
                        "card": (CARDS / f"{key[0]}_{key[1]}_text.txt").read_text(encoding="utf-8"),
                        "candidates": [{"cand_id": f"{g['model']}|{g['prompt_id']}|{g['format']}|{g['decoding']}{g['sample_idx']}",
                                        "text": g["output"]} for g in chunk]}
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} judging items -> {args.out}")
