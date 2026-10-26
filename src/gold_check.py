"""Gold checker: are the numbers in a summary supported by the source data card?

Library use:
    from gold_check import check_summary
    result = check_summary(summary_text, facts)        # facts = data/facts/{SYM}_{SNAP}.json["facts"]

CLI (score a generation file with ALL automatic rubric checks - see auto_rubric.py - and print a table):
    python src/gold_check.py outputs/base_runs/run1.jsonl                 # grouped by model,format,prompt_id
    python src/gold_check.py outputs/base_runs/run1.jsonl model,snapshot  # custom grouping

A number in the summary is SUPPORTED when it equals some fact in the card after
  - unit scaling   ("1.55 trillion baht", "THB 1,549 billion", "1,549,118 million" -> THB million)
  - rounding       (within half a unit of the last digit the summary shows: "1.5 trillion" ok for 1,549,118)
  - sign           ("fell 9.5%" matches a fact of -9.5)
  - unit class     ("%" mentions only match % facts, "x"/"times" only match multiples, money only money)
Years, dates, small counts (<=12 without a unit) and list numbering are ignored.
Numbers that match nothing are UNSUPPORTED: invented, mis-copied, or computed by the model.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS_DIR = ROOT / "data" / "facts"

# ---- patterns: numbers with optional currency and scale, scale words, headline facts for coverage
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
PATTERN = re.compile(
    rf"(?P<cur>(?:THB|Bt\.?|฿)\s*)?(?P<neg>[-−–])?(?P<num>{NUM})"
    rf"\s*(?P<unit>%|percent|per cent|percentage points|pp|x\b|times|trillion|tn\b|billion|bn\b|b\b|million|mn\b|m\b|mil\b)?"
    rf"\s*(?P<cur2>baht|THB)?",
    re.IGNORECASE,
)
SCALE = {"trillion": 1e6, "tn": 1e6, "billion": 1e3, "bn": 1e3, "b": 1e3,
         "million": 1.0, "mn": 1.0, "m": 1.0, "mil": 1.0}
KEY_FACTS = {  # coverage: does the summary mention these headline numbers?
    "net_profit": r"^income\.Net profit.*\.current$",
    "net_profit_yoy": r"^income\.Net profit.*\.yoy$",
    "top_line": r"^income\.(Revenue from operations|Net interest income|Insurance revenue)\.current$",
    "roe": r"^ratio\.ROE \(%\)\.current$",
    "pe": r"^market\.pe_x$",
    "dividend_yield": r"^market\.dividend_yield_pct$",
}
ADVICE = re.compile(r"\b(strong buy|we recommend|you should (buy|sell)|buy rating|sell rating|"
                    r"recommend (buying|selling)|is a (buy|sell)\b|target price)\b", re.I)
THAI = re.compile(r"[฀-๿]")


# ---- number extraction and matching
def _decimals(s: str) -> int:
    """Number of decimal places written in a number string."""
    return len(s.split(".")[1]) if "." in s else 0


def extract_numbers(text: str) -> list[dict]:
    """Find every number in a summary with its unit class (pct, mult, money, any), scale word and position.

    Years, dates, period labels (6M, FY2025), small counts and list numbering are skipped.
    """
    out = []
    for m in PATTERN.finditer(text):
        raw, start = m.group("num"), m.start("num")
        prev = text[max(0, start - 1):start]
        if prev and (prev.isalpha() or prev in "._/"):      # FY2025, Q2, 6M, file.v2, 30/06
            continue
        nxt = text[m.end("num"):m.end("num") + 1]
        if nxt in "/" or (nxt.isalpha() and not m.group("unit")):  # 6M, 1H26, 30/06
            continue
        unit = (m.group("unit") or "").lower()
        has_money = bool(m.group("cur") or m.group("cur2"))
        if unit in ("m", "b") and not has_money:          # "6M 2026", "1H" period labels, not money
            continue
        val = float(raw.replace(",", ""))
        ctx = text[max(0, start - 12):m.end() + 12].lower()
        is_int = "." not in raw and "," not in raw
        if is_int and not unit and not has_money:
            if 1900 <= val <= 2100:                      # years
                continue
            if val <= 12:                                 # small counts, list numbering
                continue
            if re.search(rf"\b{int(val)}\s+{MONTHS}|{MONTHS}\s+{int(val)}\b", ctx):   # dates
                continue
        # list item markers such as "1." at line start
        line_start = text.rfind("\n", 0, start) + 1
        if text[line_start:start].strip() in ("", "-", "*", "#", "##", "###") and re.match(r"\d+\.\s", text[start:start + 4]) \
                and is_int and val <= 20:
            continue
        neg = bool(m.group("neg"))
        if unit in ("%", "percent", "per cent", "percentage points", "pp"):
            cls = "pct"
        elif unit in ("x", "times"):
            cls = "mult"
        elif unit in SCALE or has_money:
            cls = "money"
        else:
            cls = "any"
        out.append({"text": m.group(0).strip(), "value": -val if neg else val, "abs": val,
                    "start": m.start(), "end": m.end(),
                    "decimals": _decimals(raw), "scale": SCALE.get(unit, 1.0) if unit in SCALE else None,
                    "cls": cls})
    return out


def _fact_cls(unit: str) -> str:
    """Unit class of a fact (pct, mult, money, text, other), comparable with extract_numbers' classes."""
    return {"%": "pct", "x": "mult", "THB m": "money", "THB": "money", "text": "text"}.get(unit, "other")


def _matches(num: dict, fact: dict) -> bool:
    """True if a summary number equals a fact after unit scaling, rounding to the digits shown, and sign."""
    fc = _fact_cls(fact["unit"])
    if num["cls"] == "pct" and fc != "pct":
        return False
    if num["cls"] == "mult" and fc != "mult":
        return False
    if num["cls"] == "money" and fc != "money":
        return False
    if fc == "text" and num["cls"] != "any":
        return False
    fv = float(fact["value"])
    # candidate interpretations of the summary number in the fact's unit
    cands = []
    if fact["unit"] == "THB m":
        scale = num["scale"] if num["scale"] is not None else 1.0
        cands.append((num["abs"] * scale, 0.5 * 10 ** (-num["decimals"]) * scale))
        if num["scale"] is None and num["cls"] == "money":          # "THB 5,734,000,000" written in baht
            cands.append((num["abs"] / 1e6, 0.5 * 10 ** (-num["decimals"]) / 1e6))
    elif fact["unit"] == "THB" and num["scale"] not in (None, 1.0):
        return False
    else:
        cands.append((num["abs"], 0.5 * 10 ** (-num["decimals"])))
    for v, tol in cands:
        tol = max(tol, 1e-9)
        if abs(v - abs(fv)) <= tol + 1e-9 or (abs(fv) > 0 and abs(v - abs(fv)) / abs(fv) < 0.0005):
            return True
    return False


def check_summary(text: str, facts: list[dict]) -> dict:
    """Number precision of a summary against the fact list: supported / unsupported numbers, key-fact coverage,
    length, 4-gram repetition, Thai characters and whether the text ends cleanly.
    """
    nums = extract_numbers(text)
    supported, unsupported = [], []
    matched_keys = set()
    for n in nums:
        hits = [f["key"] for f in facts if _matches(n, f)]
        (supported if hits else unsupported).append(n["text"])
        matched_keys.update(hits)
    coverage = {}
    for name, pat in KEY_FACTS.items():
        keys = [f["key"] for f in facts if re.search(pat, f["key"])]
        if keys:
            coverage[name] = any(k in matched_keys for k in keys)
    words = text.split()
    grams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    rep = 1 - len(set(grams)) / len(grams) if grams else 0.0
    return {
        "n_numbers": len(nums),
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "number_precision": (len(supported) / len(nums)) if nums else None,
        "unsupported": unsupported,
        "key_fact_coverage": (sum(coverage.values()) / len(coverage)) if coverage else None,
        "coverage_detail": coverage,
        "words": len(words),
        "repeat_4gram_rate": round(rep, 3),
        "thai_chars": len(THAI.findall(text)),
        "advice_flag": bool(ADVICE.search(text)),
        "ends_cleanly": text.rstrip()[-1:] in ".!?)|*" if text.strip() else False,
    }


# ---- scoring generation files
def load_facts(symbol: str, snap: str) -> list[dict]:
    """Fact list of one company snapshot (data/facts/{SYMBOL}_{SNAP}.json)."""
    return json.loads((FACTS_DIR / f"{symbol}_{snap}.json").read_text(encoding="utf-8"))["facts"]


def load_meta(symbol: str, snap: str) -> tuple[list[dict], str | None]:
    """Fact list plus the prior period's year (used to recognise prior-period wording)."""
    d = json.loads((FACTS_DIR / f"{symbol}_{snap}.json").read_text(encoding="utf-8"))
    m = re.search(r"(20\d\d)", (d.get("periods") or {}).get("prior") or "")
    return d["facts"], (m.group(1) if m else None)


def score_file(path: Path) -> list[dict]:
    """Score every generation in a jsonl file with the automatic rubric (one row per generation)."""
    from auto_rubric import auto_rubric   # local import: auto_rubric imports this module
    rows = []
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        k = (g["symbol"], g["snapshot"])
        if k not in cache:
            cache[k] = load_meta(*k)
        facts, prior_year = cache[k]
        r = auto_rubric(g["output"], facts, g.get("output_tokens"), g.get("hit_max_tokens"), prior_year)
        rows.append({**{x: g.get(x) for x in ("model", "prompt_id", "format", "symbol", "snapshot", "decoding",
                                              "sample_idx", "prompt_tokens", "output_tokens", "hit_max_tokens")},
                     **{x: r[x] for x in ("gold", "severe", "composite", "score_fidelity", "score_units", "score_dir_attr",
                                          "score_invented", "score_coverage", "score_discipline", "n_numbers",
                                          "n_unsupported", "number_precision", "key_fact_coverage", "words",
                                          "repeat_4gram_rate", "thai_chars", "filler", "advice", "over_500", "truncated")},
                     "n_direction_errors": len(r["direction_errors"]), "n_scale_errors": len(r["scale_errors"]),
                     "n_missing_scale": len(r["missing_scale"]), "n_attribution_flags": len(r["attribution_flags"]),
                     "invented_strong_per100w": r["invented"]["per_100w_strong"],
                     "unsupported": " | ".join(r["unsupported"]),
                     "direction_detail": " | ".join(f"{e['item']}: said {e['said']}, data {e['truth']}" for e in r["direction_errors"]),
                     "scale_detail": " | ".join(r["scale_errors"]),
                     "attribution_detail": " | ".join(f"{a['type']}: {a['text']}" for a in r["attribution_flags"])})
    return rows


def summarise(rows: list[dict], by=("model", "format", "prompt_id")) -> list[dict]:
    """Mean scores per group of generations (default grouping: model, format, prompt)."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[b] for b in by)].append(r)
    out = []
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    for k, rs in sorted(groups.items()):
        prec = [r["number_precision"] for r in rs if r["number_precision"] is not None]
        out.append({**dict(zip(by, k)), "n": len(rs),
                    "gold": round(mean([r["gold"] for r in rs]), 3),
                    "severe_pct": round(100 * mean([r["severe"] for r in rs])),
                    "num_precision": round(mean(prec), 3) if prec else None,
                    "dir_err": round(mean([r["n_direction_errors"] for r in rs]), 2),
                    "scale_err": round(mean([r["n_scale_errors"] for r in rs]), 2),
                    "attr_flags": round(mean([r["n_attribution_flags"] for r in rs]), 2),
                    "invented100w": round(mean([r["invented_strong_per100w"] for r in rs]), 2),
                    "coverage": round(mean([r["score_coverage"] for r in rs]), 2),
                    "out_tokens": round(mean([r["output_tokens"] or 0 for r in rs])),
                    "over_500_pct": round(100 * mean([r["over_500"] for r in rs])),
                    "trunc_pct": round(100 * mean([r["truncated"] for r in rs])),
                    "filler_pct": round(100 * mean([r["filler"] for r in rs])),
                    "advice_pct": round(100 * mean([r["advice"] for r in rs])),
                    "repeat4": round(mean([r["repeat_4gram_rate"] for r in rs]), 3)})
    return out


def print_table(rows: list[dict]):
    """Print the summary rows as an aligned text table."""
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))


# ---- command line: score a generation file, write {run}_scored.csv and {run}_summary.csv, print the summary
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    rows = score_file(path)
    detail = path.with_name(path.stem + "_scored.csv")
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("model", "format", "prompt_id")
    summ = summarise(rows, by)
    with open(path.with_name(path.stem + "_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print_table(summ)
    print(f"\nPer-summary detail: {detail}")
