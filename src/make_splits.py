"""Assign companies to dataset splits (company level, stratified by market x industry, fixed seed).

    python src/make_splits.py                  # uses data/keep_symbols.txt -> data/splits.csv

Both snapshots (T1, T2) of a company always go to the same split.
Default shares (of the kept companies, ~200):
  A_train 27%  A_val 3%    SFT on the judges' best candidate; val = early stopping
  B_train 37.5% B_test 7.5% reward model (4 candidates ranked -> 6 pairs per prompt); test = RM accuracy, human gold
  C_ppo 15%   C_eval 10%  PPO prompts (no labels needed); final evaluation of base vs SFT vs PPO
The 12 pilot companies used for prompt/format tuning (data/pilot_symbols.txt) are kept OUT of every split and listed
as 'pilot'. The 106 'reserve' companies in data/splits.csv were collected later and appended to the file with split
'reserve' (they were used as extra PPO prompts); they are not produced by this script.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SHARES = {"A_train": 0.27, "A_val": 0.03, "B_train": 0.375, "B_test": 0.075, "C_ppo": 0.15, "C_eval": 0.10}
# pilot companies: used only to choose the prompt, card format and judges (never in A / B / C)
PILOT = (ROOT / "data" / "pilot_symbols.txt").read_text(encoding="utf-8").split()


def label_sequence(n: int) -> list[str]:
    """n labels in the target proportions, spread evenly along the sequence (systematic allocation)."""
    counts = {k: int(v * n) for k, v in SHARES.items()}
    rest = n - sum(counts.values())
    for k in sorted(SHARES, key=lambda k: -(SHARES[k] * n - counts[k]))[:rest]:
        counts[k] += 1
    slots = [((i + 0.5) / c, k) for k, c in counts.items() for i in range(c)]
    return [k for _, k in sorted(slots)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=str(ROOT / "data" / "keep_symbols.txt"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    # ---- strata = market x industry, from each company's overview feed
    rng = random.Random(args.seed)
    syms = [s.strip() for s in Path(args.keep).read_text(encoding="utf-8").split() if s.strip()]
    strata = defaultdict(list)
    meta = {}
    for s in syms:
        ov = json.loads((RAW / f"{s}.json").read_text(encoding="utf-8")).get("overview", {})
        meta[s] = (ov.get("market"), ov.get("industry"))
        strata[meta[s]].append(s)
    # order: strata one after another, shuffled inside -> systematic labels give each stratum every split
    ordered = []
    for k in sorted(strata, key=str):
        g = sorted(strata[k])
        rng.shuffle(g)
        ordered += g
    # ---- systematic allocation with a random start, so every stratum is spread over the splits
    labels = label_sequence(len(ordered))
    offset = rng.randrange(len(labels))
    labels = labels[offset:] + labels[:offset]
    rows = [{"symbol": s, "split": lab, "market": meta[s][0], "industry": meta[s][1]} for s, lab in zip(ordered, labels)]
    rows += [{"symbol": s, "split": "pilot", "market": "", "industry": ""} for s in PILOT]
    # ---- write data/splits.csv and print the split sizes
    with open(ROOT / "data" / "splits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "split", "market", "industry"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["symbol"])))
    print(f"{len(ordered)} companies -> data/splits.csv")
    c = Counter(r["split"] for r in rows)
    for k in list(SHARES) + ["pilot"]:
        ind = Counter(r["industry"] for r in rows if r["split"] == k)
        print(f"  {k:<8} {c[k]:>3} companies  {dict(sorted(ind.items()))}")
