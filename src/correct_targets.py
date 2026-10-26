"""Make SFT targets by minimally correcting each prompt's best candidate ("judge-corrected demonstrations").

    python src/correct_targets.py --items data/judge_items/A.jsonl \
        --judgments outputs/judgments/A_sonnet.jsonl outputs/judgments/A_gpt.jsonl \
        --run A_fix --backend claude --model sonnet --items-per-call 3

For every item: pick the best candidate (judges' mean overall, then fidelity, then usefulness), collect every error
the judges listed for it, and ask the editor model to fix exactly those errors (and anything else the DATA does not
support) while keeping the candidate's structure and wording. Candidates that already pass the SFT gate are kept
unchanged. Output: outputs/corrections/{run}.jsonl  (resumable).
Then build the SFT files with:  python src/aggregate_labels.py ... --mode sft --corrections outputs/corrections/{run}.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402  (reuses the CLI/API backends)
from aggregate_labels import candidate_table, load_judgments, passes_gate, rank_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "corrections"

SYSTEM = ("You are a meticulous financial editor. You correct investor summaries so that every statement is supported "
          "by the source data, changing as little as possible. You reply with JSON only.")

RULES = """Edit each DRAFT so that it is fully correct, changing as little as possible:
- Fix every problem in REVIEWER ERRORS, and anything else the DATA does not support.
- Keep the draft's structure, headings, order and wording wherever they are correct. Do not rewrite correct sentences.
- Numbers: copy them exactly as in the DATA, with their unit (amounts are THB million unless the DATA says otherwise;
  never rescale, e.g. 571,134.5 stays "THB 571,134.5 million"). Say which period each figure refers to.
- Directions: "increased / decreased" must match the DATA (current vs prior).
- Remove invented causes, peer or sector comparisons the DATA does not state, forecasts, verdicts on ratios
  ("healthy", "manageable", "attractive") and any buy/sell advice. Do not add new facts.
- If a required part is missing (business, revenue and net profit with growth, financial position, valuation, dividends),
  add one short sentence using only DATA figures.
- At most 300 words. No greeting or closing remarks."""


def build_prompt(batch):
    """Editor prompt for a batch of drafts: the rules, then for each item its data card, the draft and the judges' errors.
    Returns (system, user).
    """
    blocks = []
    for b in batch:
        errs = "\n".join(f"- {e}" for e in b["errors"]) or "- (none listed; still check every statement)"
        blocks.append(f"=== ITEM {b['item_id']} ===\nDATA:\n{b['card'].strip()}\n\nDRAFT:\n{b['text'].strip()}\n\n"
                      f"REVIEWER ERRORS:\n{errs}\n")
    user = (f"{RULES}\n\nReply with JSON only: {{\"items\": [{{\"item_id\": \"<id>\", \"revised\": \"<corrected summary>\"}}]}}\n\n"
            + "\n".join(blocks))
    return SYSTEM, user


def main():
    """Pick each prompt's best candidate, keep it if it passes the SFT gate, otherwise ask the editor model to fix the
    errors the judges listed; append the results to outputs/corrections/{run}.jsonl (resumable).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--backend", required=True, choices=["claude", "codex", "cursor", "openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--items-per-call", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--cursor-cmd", default="cursor-agent")
    ap.add_argument("--base-url"); ap.add_argument("--api-key-env")
    ap.add_argument("--ollama-url", default="http://localhost:11434"); ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--think", default=None)
    args = ap.parse_args()
    args.items_per_call_judge = args.items_per_call

    # ---- load items, judgments and the corrections already done (resume)
    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    by, _ = load_judgments(args.judgments)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.run}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["item_id"] for l in out_path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not json.loads(l).get("error")}

    # ---- choose each prompt's best candidate; keep it if it already passes the gate, else queue it for editing
    todo, kept = [], 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for it in items:
            if it["item_id"] in done or len(by.get(it["item_id"], [])) < 2:
                continue
            rows = candidate_table(it, by[it["item_id"]])
            best = max(rows, key=rank_key)
            if passes_gate(best):
                fh.write(json.dumps({"item_id": it["item_id"], "cand_id": best["cand_id"], "source": "original",
                                     "revised": best["text"]}, ensure_ascii=False) + "\n")
                kept += 1
                continue
            errs = sorted({e for r in by[it["item_id"]] for e in r["scores"].get(best["cand_id"], {}).get("errors", [])})
            todo.append({"item_id": it["item_id"], "cand_id": best["cand_id"], "card": it["card"],
                         "text": best["text"], "errors": errs})
    if args.limit:
        todo = todo[:args.limit]
    print(f"{kept} best candidates already pass the gate (kept unchanged); {len(todo)} to correct "
          f"in {-(-len(todo) // args.items_per_call)} calls [{args.backend} {args.model or ''}]")

    if args.backend == "ollama":
        print("note: ollama backend uses the judge JSON schema; use a CLI/API backend for corrections")
    fn = {"claude": J.call_claude, "codex": J.call_codex, "cursor": J.call_cursor, "openai": J.call_openai}[args.backend]
    # ---- send the queued drafts to the editor model in batches
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for ci in range(0, len(todo), args.items_per_call):
            batch = todo[ci:ci + args.items_per_call]
            system, user = build_prompt(batch)
            parsed, err = None, None
            for attempt in range(args.retries + 1):
                try:
                    raw, meta = fn(system, user, args, 0)
                    parsed = J.extract_json(raw)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    print(f"  call {ci // args.items_per_call + 1}: attempt {attempt + 1} failed - {err[:200]}")
                    if J.is_limit(err):
                        print("  Looks like a usage/rate limit. Re-run later to resume.")
                        return
                    time.sleep(15 * (attempt + 1))
            got = {str(x.get("item_id")): x.get("revised", "") for x in (parsed or {}).get("items", [])}
            for b in batch:
                rec = {"item_id": b["item_id"], "cand_id": b["cand_id"], "source": "corrected",
                       "editor": f"{args.backend}:{args.model or 'default'}",
                       "n_errors_listed": len(b["errors"]), "original": b["text"]}
                if got.get(b["item_id"], "").strip():
                    rec["revised"] = got[b["item_id"]].strip()
                else:
                    rec["error"] = err or "missing in reply"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  call {ci // args.items_per_call + 1}/{-(-len(todo) // args.items_per_call)} done "
                  f"[{(time.time() - t0) / 60:.1f} min]")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
