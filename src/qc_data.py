"""Quality-check every company in data/raw and decide which ones to keep.

    python src/qc_data.py            # writes data/qc_report.csv and data/keep_symbols.txt

A company is dropped when any snapshot (T1, T2) cannot be built properly:
  not listed / suspended, no statements, no prior period to compare with, no market data,
  no net profit line, or Thai text left in the card.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cards import RAW, THAI, build_snapshot, render_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# pilot companies (prompt / format / judge selection) are checked but not written to keep_symbols.txt
PILOT = set((ROOT / "data" / "pilot_symbols.txt").read_text(encoding="utf-8").split())


def check(d: dict) -> tuple[dict, list[str]]:
    """Quality-check one company: returns (qc_report row, list of problems); keep = no problems."""
    sym = d["_meta"]["symbol"]
    ov, info = d.get("overview") or {}, d.get("info") or {}
    row = {"symbol": sym, "pilot": sym in PILOT, "stratum": d["_meta"].get("stratum", ""),
           "market": ov.get("market"), "industry": ov.get("industry"), "status": ov.get("status"),
           "sign": info.get("sign", ""), "fetch_errors": ";".join(f"{k}:{v}" for k, v in d["_meta"].get("errors", {}).items())}
    problems = []
    if ov.get("status") != "Listed":
        problems.append(f"status={ov.get('status')}")
    if info.get("sign") in ("SP", "H", "NP", "NR", "ST"):
        problems.append(f"trading sign={info.get('sign')}")
    inc = (d.get("income_statement") or [{}])
    if inc and inc[0].get("accountFormId") == "5":
        problems.append("property fund / REIT (trust, not an operating company): excluded by design")
    for snap in ("T1", "T2"):
        try:
            s = build_snapshot(d, snap)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{snap}: error {type(e).__name__}: {e}")
            continue
        if s is None:
            problems.append(f"{snap}: no usable period")
            continue
        row[f"{snap}_period"] = s["periods"]["current"]
        row["layout"] = s["company"]["statement_layout"]
        if not s["periods"]["prior"]:
            problems.append(f"{snap}: no prior period")
        if not any(r["item"].startswith("Net profit") and r["current"] is not None for r in s["income"]):
            problems.append(f"{snap}: no net profit")
        if not s["market"]:
            problems.append(f"{snap}: no market data")
        card = render_markdown(s)
        row[f"{snap}_tokens_approx"] = round(len(card) / 3.2)
        if THAI.search(card):
            problems.append(f"{snap}: Thai text in card")
        if not s["company"]["business"]:
            problems.append(f"{snap}: no business description")
    row["problems"] = " | ".join(problems)
    row["keep"] = not problems
    return row, problems


# ---- command line: check every raw file, write the report and the list of kept companies
if __name__ == "__main__":
    rows = []
    for f in sorted(RAW.glob("*.json")):
        row, _ = check(json.loads(f.read_text(encoding="utf-8")))
        rows.append(row)
    cols = ["symbol", "keep", "pilot", "market", "industry", "layout", "status", "sign", "T1_period", "T2_period",
            "T1_tokens_approx", "T2_tokens_approx", "problems", "fetch_errors", "stratum"]
    with open(ROOT / "data" / "qc_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    keep = [r["symbol"] for r in rows if r["keep"] and not r["pilot"]]
    (ROOT / "data" / "keep_symbols.txt").write_text("\n".join(keep), encoding="utf-8")
    bad = [r for r in rows if not r["keep"]]
    print(f"{len(rows)} companies checked: {len(keep)} kept (excluding pilot), {len(bad)} dropped")
    for r in bad:
        print(f"  drop {r['symbol']:<8} {r['problems']}")
    from collections import Counter
    print("layouts kept:", Counter(r.get("layout") for r in rows if r["keep"]))
