"""Automatic rubric checks (gold metrics that PPO never sees). Extends gold_check.py.

Dimensions (same names as the judge rubric):
  1 fidelity     share of numbers that exist in the data (gold_check)
  2 units        x1000 scale slips; amounts written without their scale ("THB 27,211.5" for THB 27,211.5 million)
  3 direction    "rose/fell/increase/decrease" next to a metric vs the real sign of (current - prior)
  4 attribution  numbers that only match a prior-period or sector/market value, with no wording that says so
  5 invented     causal / evaluative / peer-comparison / forecast phrases (rough proxy, per 100 words)
  6 coverage     6 headline facts + 5 required sections
  7 discipline   <= 500 tokens, not cut off, no chat filler, no advice

Gold composite (pre-registered before any PPO run - do not tune after seeing results):
  0.30 fidelity + 0.20 direction&attribution + 0.15 units + 0.15 coverage + 0.10 invented + 0.10 discipline
Severe error (any of): direction error on the top line or net profit, any x1000 scale slip, a sector/market
figure presented as the company's. gold = composite, capped at 0.4 when a severe error is present.
Validity gates (added before any PPO run): non-answer -> 0, no headline fact -> <= 0.4, truncated -> <= 0.6,
keyword stuffing -> <= 0.5, over 500 tokens -> x max(0.5, 1 - (tokens - 500) / 400).

All checks are rule-based (regular expressions and number matching against the fact list) - no language model is used.
"""
from __future__ import annotations

import re

from gold_check import _matches, check_summary, extract_numbers

# ---- metric aliases: fact item label -> regex for how a summary refers to it (most specific first) ---
METRIC_ALIASES = [
    ("Total revenues", r"total revenues?"),
    ("Revenue from operations", r"revenues?(?! growth rate)|sales|top[- ]line"),
    ("Net interest income", r"net interest income|\bNII\b"),
    ("Net fee and service income", r"(net )?fee(s)?( and service)?( income)?"),
    ("Insurance revenue", r"insurance revenue"),
    ("Insurance service result", r"insurance service result"),
    ("Net investment result", r"(net )?investment result"),
    ("Expected credit losses", r"(expected )?credit loss(es)?|provisions?"),
    ("Profit before tax", r"profit before tax|pre-?tax profit|\bEBT\b"),
    ("EPS (THB per share)", r"\bEPS\b|earnings per share"),
    ("Net profit (owners of the parent)", r"(net profit|net income|net earnings|bottom[- ]line|profit attributable)(?! margin)"),
    ("EBITDA", r"\bEBITDA\b"),
    ("EBIT", r"\bEBIT\b"),
    ("Operating cash flow", r"operating cash ?flows?|cash ?flows? from operations|cash from operations"),
    ("Investing cash flow", r"investing cash ?flows?"),
    ("Financing cash flow", r"financing cash ?flows?"),
    ("Net cash flow", r"net cash ?flows?"),
    ("Total assets", r"total assets"),
    ("Total liabilities", r"total liabilities|(?<!current )(?<!non-current )\bliabilities"),
    ("Shareholders' equity", r"shareholders'? equity|(?<!on )(?<!return on )\bequity\b(?! ratio)"),
    ("Deposits", r"deposits"),
    ("Net loans and accrued interest", r"(net )?loans"),
]
# ---- word lists used by the checks
HEADLINE = {"Revenue from operations", "Total revenues", "Net interest income", "Insurance revenue",
            "Net profit (owners of the parent)"}
UP = r"increas\w*|rose|rise[sn]?|rising|grew|grow(s|ing|n)?\b|growth (?:of|by|to|at)\b|up\b|higher|improv\w*|surg\w*|jump\w*|gained|expand\w*|climb\w*|rebound\w*|turnaround|recover\w*"
DOWN = r"decreas\w*|declin\w*|fell|fall(s|ing)?|dropp?\w*|down\b|lower|contract\w*|shr[ai]nk\w*|reduc\w*|weaken\w*|slump\w*|dip\w*|deteriorat\w*"
BREAK_WORDS = r"\b(?:suggest\w*|indicat\w*|reflect\w*|associated|due to|driven|despite|because|which|maintain\w*|remain\w*|margin|ratio|costs?|expenses?|profitab\w*|liquidity|leverage|solvency|D/E|P/E|P/BV|ROE|ROA|yield|price|turnover)\b|,\s*and\b"
PRIOR_CUES = r"prior|previous|last year|a year (earlier|ago)|year[- ]ago|year[- ]earlier|compared|versus|\bvs\b|from|up from|down from|than"
SECTOR_CUES = r"sector|industry|SET index|\bSET\b|market|benchmark|index|peers"
INVENTED = {
    "causal": r"driven by|due to|thanks to|because|as a result of|reflect(s|ing|ed)?|attributable to|owing to|on the back of|fu?elled by|supported by|led by",
    "evaluative": r"healthy|robust|solid|attractive|compelling|undervalued|overvalued|cheap|expensive|impressive|concerning|efficient|favou?rable|conservative|strong|weak",
    "comparative": r"peers|industry average|competitors|top performers|among the (top|best|largest|leading)|market leader|outperform\w*",
    "forecast": r"will likely|expected to|poised|potential for|future growth|outlook|going forward|positions? (it|the company) (well|for)",
}
FILLER = re.compile(r"^\s*(okay|ok|sure|certainly|here('|’)s|here is|absolutely)\b|(let me know|would you like|do you want me|hope this helps)",
                    re.I | re.M)
ADVICE = re.compile(r"\b(strong buy|we recommend|you should (buy|sell|consider)|investors should (buy|sell|consider)|"
                    r"buy rating|sell rating|recommend (buying|selling)|is a (buy|sell)\b|target price)\b", re.I)
SECTIONS = {"business": r"business|operates|engaged in|provides",
            "performance": r"revenue|net profit|net income|performance|earnings",
            "position": r"assets|liabilit|equity|cash flow|balance sheet|D/E|current ratio",
            "valuation": r"P/E|P/BV|valuation|dividend yield|market cap",
            "takeaway": r"takeaway|overall|in summary|conclusion"}


# ---- individual checks
def _clauses(text):
    """Split into clauses with their start offsets."""
    out, start = [], 0
    for m in re.finditer(r"\.(?=\s|$)|[;\n]|\bwhile\b|\bbut\b|\bwhereas\b|\bhowever\b", text):  # not decimal points
        if m.end() > start:
            out.append((start, text[start:m.start()]))
        start = m.end()
    if start < len(text):
        out.append((start, text[start:]))
    return [(s, c) for s, c in out if c.strip()]


def _fact_map(facts):
    """Index statement facts as {item label: {'current': value, 'prior': value}} for the direction check."""
    fm = {}
    for f in facts:
        k = f["key"]
        for sec in ("income.", "balance.", "cash_flow."):
            if k.startswith(sec):
                item, per = k[len(sec):].rsplit(".", 1)
                fm.setdefault(item, {})[per] = f["value"]
    return fm


def direction_errors(text, facts):
    """Find rise/fall words that contradict the data (current vs prior value of the metric they refer to).

    Each clause is scanned for metric mentions (METRIC_ALIASES) and direction words (UP / DOWN). A direction word is
    paired with the metric it modifies ("rise in revenue", "higher revenue") or with the closest metric before it
    ("revenue rose"), unless a break word shows it belongs to another phrase. Returns one error per clause at most:
    {item, said, truth, headline, text}; headline=True for the top line and net profit (these count as severe).
    """
    fm = _fact_map(facts)
    errors = []
    for _, clause in _clauses(text):
        if re.search(r"dividend policy|policy (is|to)", clause, re.I):   # policy text describes rules, not changes
            continue
        # metric mentions (most specific alias wins; mask matched spans so 'revenues' isn't counted twice)
        masked, mentions = clause, []
        for item, pat in METRIC_ALIASES:
            if item not in fm:
                continue
            for m in re.finditer(pat, masked, re.I):
                mentions.append((m.start(), m.end(), item))
                masked = masked[:m.start()] + "#" * (m.end() - m.start()) + masked[m.end():]
        if not mentions:
            continue
        for m in re.finditer(rf"\b({UP})|\b({DOWN})", clause, re.I):
            said = "up" if m.group(1) else "down"
            word = m.group(0).lower()
            after = clause[m.end():]
            nxt = (re.match(r"\s*([\w%()+\-.,]+)", after) or [None, ""])[1].lower()
            if re.match(r"[\d(+\-−]", nxt):        # "decreased 2.0%" -> a verb with its amount, not an adjective
                nxt = "by"
            # (a) "rise in X" / "decline of X"  (b) adjective before a noun: "higher costs", "declining revenue"
            modifies_next = nxt in ("in", "of") or (
                re.fullmatch(r"\w+ing|higher|lower|increased|decreased|reduced|improved|weaker|stronger", word)
                and nxt not in ("by", "to", "from", "slightly", "significantly", "sharply", "marginally", "substantially",
                                "strongly", "modestly", "notably", ",", "and", "year-on-year", "yoy", "compared"))
            if modifies_next:
                ahead = [x for x in mentions if 0 <= x[0] - m.end() <= 30]
                if not ahead:
                    continue            # the word describes something we don't track (debt, costs, volumes, ...)
                near = min(ahead, key=lambda x: x[0])
            else:
                before = [x for x in mentions if 0 <= m.start() - x[1] <= 60]
                if not before:
                    continue
                near = max(before, key=lambda x: x[1])
                between = clause[near[1]:m.start()]
                if re.search(BREAK_WORDS, between, re.I):   # direction word belongs to another phrase
                    continue
            item = near[2]
            cur, pri = fm[item].get("current"), fm[item].get("prior")
            if cur is None or pri is None or abs(cur - pri) <= 0.0005 * max(abs(pri), 1e-9):
                continue
            truth = "up" if cur > pri else "down"
            if said != truth:
                errors.append({"item": item, "said": said, "truth": truth, "headline": item in HEADLINE,
                               "text": clause.strip()[:160]})
                break  # one error per clause is enough
    return errors


def attribution_flags(text, facts, prior_year=None):
    """Flag numbers presented without the right context: a value that only matches a sector / SET-index figure with no
    sector wording nearby (sector_as_company, severe), or one that only matches a prior-period value with no prior-period
    wording nearby (prior_as_current).
    """
    flags = []
    for n in extract_numbers(text):
        hits = [f for f in facts if _matches(n, f)]
        if not hits:
            continue
        ctx = text[max(0, n["start"] - 90): n["end"] + 40]
        keys = [h["key"] for h in hits]
        if all(k.startswith(("market.sector_", "market.set_index")) for k in keys):
            if not re.search(SECTOR_CUES, ctx, re.I):
                flags.append({"type": "sector_as_company", "text": n["text"], "context": ctx.strip()[:140]})
            continue
        if all(h.get("period") == "prior" for h in hits):
            cue = re.search(PRIOR_CUES, ctx, re.I) or (prior_year and prior_year in ctx)
            if not cue:
                flags.append({"type": "prior_as_current", "text": n["text"], "context": ctx.strip()[:140]})
    return flags


def unit_checks(text, facts):
    """Return (scale_errors, missing_scale).

    scale_errors: numbers that match no fact as written but match a THB-million fact after multiplying or dividing by
    1,000 or 1,000,000 (e.g. "THB 571.1 million" for THB 571,134.5 million; severe).
    missing_scale: correct THB-million amounts of 100 or more written without "million" / "billion" after them.
    """
    scale_errors, missing_scale = [], []
    for n in extract_numbers(text):
        hits = [f for f in facts if _matches(n, f)]
        if hits:
            if n["scale"] is None and n["abs"] >= 100 and any(f["unit"] == "THB m" for f in hits) \
                    and not any(f["unit"] != "THB m" for f in hits):
                after = text[n["end"]: n["end"] + 25].lower()
                if not re.search(r"million|billion|trillion|\bmn\b|\bbn\b", after):
                    missing_scale.append(n["text"])
            continue
        if n["cls"] in ("pct", "mult"):
            continue
        for mult in (1e3, 1e-3, 1e6, 1e-6):
            m = dict(n, abs=n["abs"] * mult)
            if any(_matches(m, f) for f in facts if f["unit"] == "THB m"):
                scale_errors.append(n["text"])
                break
    return scale_errors, missing_scale


def invented_counts(text):
    """Count invented-claim phrases by family (causal, evaluative, comparative, forecast) by keyword matching."""
    words = max(len(text.split()), 1)
    c = {k: len(re.findall(rf"\b(?:{p})\b", text, re.I)) for k, p in INVENTED.items()}
    c["per_100w_strong"] = round(100 * (c["causal"] + c["comparative"] + c["forecast"]) / words, 2)
    return c


# ---- gold score: component scores, severe cap and validity gates
META_CLAIMS = re.compile(r"\b(accura(?:te|cy)|data[- ]supported|verified|factual|precise(?:ly)?|copied exactly|"
                         r"consistent with the data|correct units|no rescaling|no (?:forecasts?|recommendations?|invented))\b", re.I)


def auto_rubric(text, facts, output_tokens=None, hit_max_tokens=None, prior_year=None):
    """Score one summary against the company's facts; returns every check, the component scores and the gold score.

    output_tokens / hit_max_tokens come from generation (tokens are estimated from characters when not given);
    prior_year helps recognise prior-period wording.
    """
    # run every check
    base = check_summary(text, facts)
    dir_err = direction_errors(text, facts)
    attr = attribution_flags(text, facts, prior_year)
    scale_err, missing_scale = unit_checks(text, facts)
    inv = invented_counts(text)
    filler = bool(FILLER.search(text))
    advice = bool(ADVICE.search(text))
    over = output_tokens is not None and output_tokens > 500
    truncated = bool(hit_max_tokens) or not base["ends_cleanly"]
    sections = {k: bool(re.search(p, text, re.I)) for k, p in SECTIONS.items()}

    # component scores, each 0-1
    fidelity = base["number_precision"] if base["number_precision"] is not None else 0.0
    units = max(0.0, 1 - 0.34 * len(scale_err) - 0.10 * len(missing_scale))
    dir_attr = max(0.0, 1 - 0.34 * len(dir_err) - 0.20 * len(attr))
    invented = max(0.0, 1 - 0.10 * (inv["causal"] + inv["comparative"] + inv["forecast"]) - 0.03 * inv["evaluative"])
    coverage = 0.5 * (base["key_fact_coverage"] or 0) + 0.5 * (sum(sections.values()) / len(sections))
    discipline = max(0.0, 1 - 0.25 * (over + truncated + filler + advice))
    # weighted composite (pre-registered weights) and the severe-error cap
    composite = (0.30 * fidelity + 0.20 * dir_attr + 0.15 * units + 0.15 * coverage
                 + 0.10 * invented + 0.10 * discipline)
    severe = (any(e["headline"] for e in dir_err) or bool(scale_err)
              or any(a["type"] == "sector_as_company" for a in attr))
    gold = min(composite, 0.4) if severe else composite
    # Validity gates (added 25 Oct, BEFORE any PPO run; component weights unchanged). The probe suite showed that the
    # weighted sum alone rewards degenerate outputs (empty 0.53, template 0.63, keyword-stuffed 0.95, truncated 0.90).
    tok = output_tokens if output_tokens is not None else round(0.28 * len(text))   # ~0.28 Qwen tokens / char
    meta_hits = len(META_CLAIMS.findall(text))
    stuffed = meta_hits >= 2 or (base["repeat_4gram_rate"] or 0) > 0.15
    gates = []
    if (base["n_numbers"] or 0) == 0 or base["words"] < 30:
        gold = 0.0; gates.append("non_answer")
    elif not base["key_fact_coverage"]:          # none of the headline figures (revenue, profit, assets...) at all
        gold = min(gold, 0.4); gates.append("no_key_facts")
    if truncated:
        gold = min(gold, 0.6); gates.append("truncated")
    if stuffed:
        gold = min(gold, 0.5); gates.append("stuffed")
    if tok > 500:
        gold *= max(0.5, 1 - (tok - 500) / 400); gates.append("over_500")
    return {
        **{k: base[k] for k in ("n_numbers", "n_unsupported", "number_precision", "key_fact_coverage", "words",
                                "repeat_4gram_rate", "thai_chars")},
        "unsupported": base["unsupported"],
        "scale_errors": scale_err, "missing_scale": missing_scale,
        "direction_errors": dir_err, "attribution_flags": attr,
        "invented": inv, "filler": filler, "advice": advice, "over_500": over, "truncated": truncated,
        "sections": sections, "ends_cleanly_ok": bool(base["ends_cleanly"]),
        "score_fidelity": round(fidelity, 3), "score_units": round(units, 3), "score_dir_attr": round(dir_attr, 3),
        "score_invented": round(invented, 3), "score_coverage": round(coverage, 3),
        "score_discipline": round(discipline, 3),
        "composite": round(composite, 3), "severe": severe,
        "gold": round(gold, 3), "gates": gates, "meta_claims": meta_hits, "est_tokens": tok,
    }
