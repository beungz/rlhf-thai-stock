"""Compare rubric judgments: each judge's order consistency, judge-vs-judge agreement, judge-vs-automatic checks.

    python src/analyze_judges.py data/judge_items/pilot.jsonl outputs/judgments/rubric_sonnet.jsonl outputs/judgments/rubric_gpt.jsonl [labeling/results/*.jsonl]

Human results from the labelling form use the same format, so they are compared the same way.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_rubric import auto_rubric  # noqa: E402
from gold_check import load_meta  # noqa: E402

DIMS = ["fidelity", "units", "direction", "attribution", "invented", "coverage"]
AUTO_FOR = {"fidelity": "score_fidelity", "units": "score_units", "direction": "score_dir_attr",
            "attribution": "score_dir_attr", "invented": "score_invented", "coverage": "score_coverage"}


def spearman(x, y):
    """Spearman rank correlation of two equal-length lists (average ranks for ties); NaN if fewer than 3 values."""
    if len(x) < 3:
        return float("nan")
    def ranks(v):
        """Rank positions of `v`, giving tied values their average rank."""
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def pairs(rank):
    """All ordered (better, worse) pairs implied by a ranking list, best first."""
    return {(a, b) for i, a in enumerate(rank) for b in rank[i + 1:]}


def load(path):
    """Read one judgment file into {item_id: {order: record}}; returns (records, n_errors, n_lines).

    Records with an error or without a ranking are counted as errors and skipped.
    """
    by, errors, n = {}, 0, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if r.get("error") or "ranking" not in r:
            errors += 1
            continue
        by.setdefault(r["item_id"], {})[r["order"]] = r
    return by, errors, n


def mean_over_orders(recs):
    """Average a judge's fwd/rev (or single human) judgments per candidate."""
    out = {}
    for r in recs.values():
        for c, v in r["overall"].items():
            out.setdefault(c, {"overall": [], "severe": [], **{d: [] for d in DIMS}})
            out[c]["overall"].append(v)
            sc = r["scores"].get(c, {})
            out[c]["severe"].append(bool(sc.get("severe_error")))
            for d in DIMS:
                if isinstance(sc.get(d), (int, float)):
                    out[c][d].append(sc[d])
    return {c: {k: (sum(v) / len(v) if v else float("nan")) for k, v in d.items()} for c, d in out.items()}


def main():
    """Print order consistency, judge-vs-judge agreement and judge-vs-automatic agreement for the given runs."""
    items_path, runs = sys.argv[1], sys.argv[2:]
    items = {json.loads(l)["item_id"]: json.loads(l) for l in Path(items_path).read_text(encoding="utf-8").splitlines() if l.strip()}
    # ---- load every judgment file (AI judges and/or human form results)
    J = {}
    for p in runs:
        name = Path(p).stem
        J[name], err, n = load(p)
        print(f"{name}: {n} records, {err} errors/invalid")

    # automatic rubric for every candidate
    auto = {}
    for iid, it in items.items():
        facts, py = load_meta(it["symbol"], it["snapshot"])
        for c in it["candidates"]:
            auto[(iid, c["cand_id"])] = auto_rubric(c["text"], facts, prior_year=py)

    print("\n== Order consistency (forward vs reversed) ==")
    for name, by in J.items():
        ag, top, ovd = [], [], []
        for d in by.values():
            if "fwd" in d and "rev" in d:
                a, b = d["fwd"], d["rev"]
                ag.append(len(pairs(a["ranking"]) & pairs(b["ranking"])) / max(len(pairs(a["ranking"])), 1))
                top.append(a["ranking"][0] == b["ranking"][0])
                ovd += [abs(a["overall"][c] - b["overall"].get(c, a["overall"][c])) for c in a["overall"]]
        if ag:
            print(f"  {name:16} pair agreement {sum(ag)/len(ag):.2f} | same top pick {sum(top)/len(top):.2f} | "
                  f"mean |overall fwd-rev| {sum(ovd)/len(ovd):.2f}  (n items {len(ag)})")

    print("\n== Judge vs judge (averaged over orders) ==")
    M = {name: {iid: mean_over_orders(d) for iid, d in by.items()} for name, by in J.items()}
    for a, b in combinations(J, 2):
        ag = n = 0
        xs, ys, sev_a, sev_b = [], [], [], []
        for iid in M[a]:
            if iid not in M[b]:
                continue
            ma, mb = M[a][iid], M[b][iid]
            cands = [c for c in ma if c in mb]
            for c1, c2 in combinations(cands, 2):
                da, db = ma[c1]["overall"] - ma[c2]["overall"], mb[c1]["overall"] - mb[c2]["overall"]
                if da and db:
                    n += 1
                    ag += (da > 0) == (db > 0)
            for c in cands:
                xs.append(ma[c]["overall"]); ys.append(mb[c]["overall"])
                sev_a.append(ma[c]["severe"] >= 0.5); sev_b.append(mb[c]["severe"] >= 0.5)
        sev_agree = sum(x == y for x, y in zip(sev_a, sev_b)) / max(len(sev_a), 1)
        print(f"  {a} vs {b}: pairwise preference agreement {ag}/{n} = {ag/max(n,1):.2f} | overall-score Spearman "
              f"{spearman(xs, ys):.2f} | severe-flag agreement {sev_agree:.2f} (severe rate {sum(sev_a)/max(len(sev_a),1):.2f} vs {sum(sev_b)/max(len(sev_b),1):.2f})")

    print("\n== Judge vs automatic checks (per candidate; Spearman, severe-flag agreement) ==")
    for name in J:
        row = []
        for d in DIMS:
            xs, ys = [], []
            for iid, m in M[name].items():
                for c, v in m.items():
                    if (iid, c) in auto and v[d] == v[d]:
                        xs.append(v[d]); ys.append(auto[(iid, c)][AUTO_FOR[d]])
            row.append(f"{d} {spearman(xs, ys):.2f}")
        xs, ys, sa, sj = [], [], [], []
        for iid, m in M[name].items():
            for c, v in m.items():
                if (iid, c) in auto:
                    xs.append(v["overall"]); ys.append(auto[(iid, c)]["gold"])
                    sa.append(auto[(iid, c)]["severe"]); sj.append(v["severe"] >= 0.5)
        both = sum(a and b for a, b in zip(sa, sj))
        print(f"  {name:16} overall~gold {spearman(xs, ys):.2f} | " + " | ".join(row))
        print(f"  {'':16} severe: auto flags {sum(sa)}, judge flags {sum(sj)}, both {both} "
              f"(judge catches {both}/{sum(sa)} of auto-flagged)")


if __name__ == "__main__":
    main()
