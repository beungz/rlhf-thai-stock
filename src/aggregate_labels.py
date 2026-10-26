"""Turn judge (and human) rubric judgments into training data.

    python src/aggregate_labels.py --items data/judge_items/A.jsonl --judgments outputs/judgments/A_sonnet.jsonl outputs/judgments/A_gpt.jsonl --mode sft
    python src/aggregate_labels.py --items data/judge_items/B.jsonl --judgments outputs/judgments/B_sonnet.jsonl outputs/judgments/B_gpt.jsonl --mode rm

Per candidate, scores are averaged over every judge and both orders (fwd/rev), then combined with the
automatic rubric (src/auto_rubric.py). Uses whatever judgments exist; reports coverage.

--mode sft : one target per prompt = the best candidate, ONLY if it passes the gate
             (no severe error by judges' majority or automatic checks; judge fidelity/units/direction/attribution >= 4;
              automatic number precision >= 0.95; not cut off). Prompts with no passing candidate are skipped.
             Writes data/sft/{split}.jsonl  ({"messages": [system, user, assistant]})
--mode rm  : all candidate pairs within a prompt with |mean overall difference| >= --min-margin.
             Writes data/rm/{split}.jsonl  ({"chosen": [...], "rejected": [...], "margin": float, ...})
             (TRL RewardTrainer conversational format with an optional margin column.)
Also writes a per-candidate table: data/{sft|rm}/candidates_{stem}.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_rubric import auto_rubric  # noqa: E402
from gold_check import load_meta  # noqa: E402
from prompts import PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "data" / "cards"
DIMS = ["fidelity", "units", "direction", "attribution", "invented", "coverage"]
PERSONAS = ["value_investor", "fund_manager", "retail_investor"]


def load_judgments(paths):
    """Read judgment files into {item_id: [records]} (records tagged with their source file).

    Returns (by_item, stats) where stats maps each file stem to (valid, invalid) counts.
    """
    by = defaultdict(list)          # item_id -> list of records
    stats = {}
    for p in paths:
        ok = bad = 0
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or "overall" not in r:
                bad += 1
                continue
            r["_source"] = Path(p).stem
            by[r["item_id"]].append(r)
            ok += 1
        stats[Path(p).stem] = (ok, bad)
    return by, stats


def mean(xs):
    """Mean of the numeric values in `xs`, or None if there are none."""
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def candidate_table(item, recs):
    """One row per candidate of a judging item: judge scores averaged over all judgments, plus automatic checks."""
    facts, prior_year = load_meta(item["symbol"], item["snapshot"])
    rows = []
    for c in item["candidates"]:
        cid = c["cand_id"]
        got = [r for r in recs if cid in r["overall"]]
        a = auto_rubric(c["text"], facts, prior_year=prior_year)
        row = {"item_id": item["item_id"], "symbol": item["symbol"], "snapshot": item["snapshot"], "cand_id": cid,
               "n_judgments": len(got), "judges": ",".join(sorted({r["_source"] for r in got})),
               "overall": mean([r["overall"][cid] for r in got]),
               "severe_votes": mean([float(bool(r["scores"].get(cid, {}).get("severe_error"))) for r in got]),
               "auto_gold": a["gold"], "auto_severe": a["severe"], "auto_precision": a["number_precision"] or 0.0,
               "auto_truncated": a["truncated"], "auto_dir_errors": len(a["direction_errors"]),
               "auto_scale_errors": len(a["scale_errors"]), "words": a["words"],
               "errors": " | ".join(sorted({e for r in got for e in r["scores"].get(cid, {}).get("errors", [])}))[:600]}
        for d in DIMS:
            row[d] = mean([r["scores"].get(cid, {}).get(d) for r in got])
        for p in PERSONAS:
            row["use_" + p] = mean([(r.get("usefulness") or {}).get(p, {}).get(cid) for r in got])
        row["usefulness"] = mean([row["use_" + p] for p in PERSONAS])
        row["text"] = c["text"]
        rows.append(row)
    return rows


def passes_gate(r, min_dim=4.0, min_precision=0.95):
    """SFT quality gate: no severe error (judges' majority or automatic), not cut off, number precision >= min_precision,
    and judge fidelity / units / direction / attribution all >= min_dim.
    """
    if r["overall"] is None:
        return False
    if (r["severe_votes"] or 0) >= 0.5 or r["auto_severe"] or r["auto_truncated"]:
        return False
    if r["auto_precision"] < min_precision:
        return False
    return all((r[d] or 0) >= min_dim for d in ("fidelity", "units", "direction", "attribution"))


def rank_key(r):
    """Sort key for choosing the best candidate: overall, then fidelity, then usefulness, then automatic gold."""
    return ((r["overall"] or 0), (r["fidelity"] or 0), (r["usefulness"] or 0), r["auto_gold"])


def prompt_messages(cand_id, symbol, snapshot):
    # cand_id = model|prompt_id|format|decoding
    """Rebuild the chat prompt a candidate was generated from (its cand_id is model|prompt_id|format|decoding)."""
    _, pid, fmt, _ = cand_id.split("|")
    card = (CARDS / f"{symbol}_{snapshot}_{fmt}.txt").read_text(encoding="utf-8")
    return PROMPTS[pid](card, fmt)


def main():
    """Build SFT examples (--mode sft) or reward-model preference pairs (--mode rm) from the judgments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--mode", choices=["sft", "rm"], required=True)
    ap.add_argument("--min-dim", type=float, default=4.0, help="SFT gate: minimum judge score on dims 1-4")
    ap.add_argument("--min-margin", type=float, default=0.5, help="RM: minimum mean-overall difference for a pair")
    ap.add_argument("--min-judgments", type=int, default=2, help="ignore items with fewer judgments than this")
    ap.add_argument("--corrections", help="SFT: outputs/corrections/{run}.jsonl from correct_targets.py; corrected texts are "
                                          "used as targets if they pass the automatic check")
    args = ap.parse_args()

    # ---- load judging items, judgments and (optionally) editor corrections
    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    by, stats = load_judgments(args.judgments)
    for k, (ok, bad) in stats.items():
        print(f"  {k}: {ok} valid judgments, {bad} errors")
    split_of = {r["symbol"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8"))}

    corrections, rejected_corr = None, []
    if args.corrections:
        corrections = {}
        for l in Path(args.corrections).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("revised") and not r.get("error"):
                    corrections[r["item_id"]] = r
    # ---- one pass over the items: SFT target or RM pairs per prompt
    out_dir = ROOT / "data" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, outputs = [], defaultdict(list)
    covered = skipped = 0
    for it in items:
        recs = by.get(it["item_id"], [])
        if len(recs) < args.min_judgments:
            continue
        covered += 1
        rows = candidate_table(it, recs)
        all_rows += rows
        split = split_of.get(it["symbol"], "unknown")
        msgs = prompt_messages(it["candidates"][0]["cand_id"], it["symbol"], it["snapshot"])
        if args.mode == "sft" and corrections is not None:
            # SFT with corrections: use the editor's revised text if it passes the automatic re-check
            c = corrections.get(it["item_id"])
            if not c:
                skipped += 1
                continue
            facts, py = load_meta(it["symbol"], it["snapshot"])
            a = auto_rubric(c["revised"], facts, prior_year=py)
            if a["severe"] or (a["number_precision"] or 0) < 0.95 or not a["ends_cleanly_ok"] or a["words"] > 330:
                skipped += 1
                rejected_corr.append((it["item_id"], a["severe"], a["number_precision"], a["words"],
                                      [e["item"] for e in a["direction_errors"]], a["scale_errors"]))
                continue
            msgs = prompt_messages(c["cand_id"], it["symbol"], it["snapshot"])
            outputs[split].append({"messages": msgs + [{"role": "assistant", "content": c["revised"]}],
                                   "item_id": it["item_id"], "cand_id": c["cand_id"], "source": c.get("source"),
                                   "auto_gold": a["gold"]})
            continue
        # SFT without corrections: best candidate among those passing the gate
        if args.mode == "sft":
            ok = [r for r in rows if passes_gate(r, args.min_dim)]
            if not ok:
                skipped += 1
                continue
            best = max(ok, key=rank_key)
            msgs = prompt_messages(best["cand_id"], it["symbol"], it["snapshot"])
            outputs[split].append({"messages": msgs + [{"role": "assistant", "content": best["text"]}],
                                   "item_id": it["item_id"], "cand_id": best["cand_id"], "overall": best["overall"],
                                   "auto_gold": best["auto_gold"]})
        # RM: every pair of candidates whose mean overall scores differ by >= min_margin
        else:
            for a, b in combinations(rows, 2):
                if a["overall"] is None or b["overall"] is None:
                    continue
                diff = a["overall"] - b["overall"]
                if abs(diff) < args.min_margin:
                    continue
                ch, rj = (a, b) if diff > 0 else (b, a)
                outputs[split].append({
                    "chosen": msgs + [{"role": "assistant", "content": ch["text"]}],
                    "rejected": msgs + [{"role": "assistant", "content": rj["text"]}],
                    "margin": round(abs(diff), 3), "item_id": it["item_id"],
                    "chosen_id": ch["cand_id"], "rejected_id": rj["cand_id"],
                    "chosen_severe": bool((ch["severe_votes"] or 0) >= 0.5), "rejected_severe": bool((rj["severe_votes"] or 0) >= 0.5),
                    "chosen_auto_gold": ch["auto_gold"], "rejected_auto_gold": rj["auto_gold"]})

    # ---- write the candidate table and the per-split output files, then print a summary
    stem = Path(args.items).stem
    with open(out_dir / f"candidates_{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        cols = [c for c in (all_rows[0].keys() if all_rows else []) if c != "text"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    for split, recs in sorted(outputs.items()):
        path = out_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        print(f"  wrote {len(recs)} {'examples' if args.mode == 'sft' else 'pairs'} -> {path.relative_to(ROOT)}")
    if rejected_corr:
        print(f"corrections rejected by the automatic check: {len(rejected_corr)}")
        for x in rejected_corr[:10]:
            print("   ", x)
    if corrections is not None:
        src = defaultdict(int)
        for recs in outputs.values():
            for r in recs:
                src[r.get("source")] += 1
        print("SFT targets by source:", dict(src))
    print(f"items: {len(items)} | with >= {args.min_judgments} judgments: {covered}"
          + (f" | no candidate passed the gate: {skipped}" if args.mode == "sft" else ""))
    if args.mode == "rm" and outputs:
        ms = [r["margin"] for recs in outputs.values() for r in recs]
        sev = sum(r["rejected_severe"] and not r["chosen_severe"] for recs in outputs.values() for r in recs)
        agree = sum(r["chosen_auto_gold"] >= r["rejected_auto_gold"] for recs in outputs.values() for r in recs)
        print(f"pairs: {len(ms)} | mean margin {sum(ms)/len(ms):.2f} | chosen=clean & rejected=severe: {sev} | "
              f"automatic gold agrees with chosen: {agree}/{len(ms)}")


if __name__ == "__main__":
    main()
