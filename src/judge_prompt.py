"""Rubric judge prompt: score each candidate summary on the shared rubric, then rank.

Same dimensions as the automatic checks (src/auto_rubric.py) and the human labelling form, so
judge scores, human scores and gold metrics can be compared dimension by dimension.
Facts dimensions are scored once per candidate (persona-independent); personas only rate usefulness.
"""

# ---- personas (usefulness only), rubric dimensions, system prompt and the rubric text shown to every judge
PERSONAS = {
    "value_investor": "a VALUE INVESTOR: valuation (P/E, P/BV, dividend yield), balance-sheet strength, cash flow, durable earnings",
    "fund_manager": "a FUND MANAGER: growth and what the data shows about it, risk, ratios (ROE, margins, leverage), quick to act on",
    "retail_investor": "a RETAIL INVESTOR: plain language, key takeaway first, no unexplained jargon",
}
DIMENSIONS = ["fidelity", "units", "direction", "attribution", "invented", "coverage"]

SYSTEM = ("You are an expert, strict evaluator of investor summaries. You check every number and claim against "
          "the source data. You reply with JSON only.")

RUBRIC = """Score every summary on each dimension from 1 (worst) to 5 (best):

fidelity - are the numbers the ones in the DATA (rounding is fine)?
  5 all correct | 4 one small slip on a minor figure | 3 one wrong number | 2 two or three wrong | 1 four or more wrong, or revenue / main income line / net profit wrong
units - correct unit and scale on every amount?
  5 all correct | 4 unit missing on a minor figure | 3 unit or scale missing on a key figure ("THB 27,211.5" without "million") | 1 any amount off by 1,000x or more (e.g. "THB 571.1 million" for THB 571,134.5 million)
direction - every "increased / decreased / grew / fell" and every number the summary calculates is right?
  5 all correct | 3 one wrong direction or calculation on a minor item | 1 wrong direction on revenue or net profit, or a loss-to-profit turnaround described as a decline
attribution - every figure tied to the right period and the right entity?
  5 all correct | 3 one figure from the wrong period | 1 a sector or market figure presented as the company's, or the prior period presented as current for a headline item
invented - only what the DATA supports?
  5 nothing added | 4 mild generic framing | 3 one unsupported cause, peer comparison or forecast | 2 several | 1 invented causes throughout, or wrong verdicts on ratios (e.g. calling D/E 6.7 "manageable" or P/BV 19 "good value")
coverage - business, top line and profit with growth, financial position, valuation, dividends?
  5 all present | 4 one missing | 3 two missing | 2 three missing | 1 mostly missing

severe_error = true ONLY for these factual failures:
  - a wrong revenue / main income line / net profit figure,
  - a wrong direction (rise vs fall) for revenue / main income line / net profit, or a loss-to-profit turnaround called a decline,
  - an amount off by 1,000x or more,
  - a sector or market figure presented as the company's own.
NOT severe (score them under "invented", "direction", "attribution" and in overall instead): invented causes, peer comparisons,
forecasts, wrong verdicts on ratios ("healthy", "manageable"), advice, wrong direction on minor items. List every mistake in "errors"
either way. Most summaries should have severe_error = false; reserve it for the failures listed above.

usefulness (1-5) for each reader:
{personas}

overall (1-5): your holistic quality judgement, using the whole scale: 5 = accurate and useful with at most trivial issues,
4 = good with minor issues, 3 = usable but with clear weaknesses, 2 = misleading in places, 1 = unreliable.
It MUST be 1 or 2 when severe_error is true.
Buy/sell advice, greetings or chat filler, and running past 500 tokens lower the overall score. Length alone is not a merit.
ranking: all summaries best to worst by overall (break ties by fidelity, then average usefulness). No ties."""


# ---- prompt building, reply validation and letter -> candidate mapping
def _item_block(item: dict, letters: list[str], order: list[int]) -> str:
    """Text block for one judging item: its data card, then each candidate under its letter in the given order."""
    parts = [f"=== ITEM {item['item_id']} ===", "DATA:", item["card"].strip(), ""]
    for letter, idx in zip(letters, order):
        parts += [f"--- SUMMARY {letter} ---", item["candidates"][idx]["text"].strip(), ""]
    return "\n".join(parts)


def build_prompt(items: list[dict], orders: list[list[int]]) -> tuple[str, str, list[dict]]:
    """Return (system, user, letter_maps). One or more items per call."""
    blocks, maps = [], []
    for item, order in zip(items, orders):
        letters = [chr(65 + i) for i in range(len(order))]
        blocks.append(_item_block(item, letters, order))
        maps.append({letter: item["candidates"][idx]["cand_id"] for letter, idx in zip(letters, order)})
    personas = "\n".join(f"  {k}: {v}" for k, v in PERSONAS.items())
    L = [chr(65 + i) for i in range(len(orders[0]))]
    one = "{" + ", ".join(f'"{d}": 1-5' for d in DIMENSIONS) + ', "severe_error": false, "errors": ["<short>"]}'
    shape = (
        '{"items": [{"item_id": "<id exactly as given>",\n'
        f'  "scores": {{"{L[0]}": {one}, "{L[1]}": ..., ...}},\n'
        f'  "usefulness": {{"value_investor": {{"{L[0]}": 1-5, ...}}, "fund_manager": {{...}}, "retail_investor": {{...}}}},\n'
        f'  "overall": {{"{L[0]}": 1-5, ...}},\n'
        f'  "ranking": [{", ".join(chr(34) + x + chr(34) for x in L)}],\n'
        '  "reason": "<one sentence>"}]}'
    )
    user = f"""You will evaluate summaries written about Thai listed companies. For each ITEM, read the DATA, then
evaluate every summary.

{RUBRIC.format(personas=personas)}

Reply with JSON only, exactly in this shape (one entry per ITEM, in the same order; "errors" lists the specific
mistakes found, empty list if none):
{shape}

{chr(10).join(blocks)}"""
    return SYSTEM, user, maps


def validate(entry: dict, letters: list[str]) -> list[str]:
    """Return a list of problems; empty means the entry is complete and well-formed."""
    if not isinstance(entry, dict):
        return ["entry is not an object"]
    probs = []
    rk = entry.get("ranking")
    if not isinstance(rk, list) or sorted(rk) != sorted(letters):
        probs.append(f"bad ranking {rk}")
    sc = entry.get("scores") or {}
    for l in letters:
        c = sc.get(l)
        if not isinstance(c, dict):
            probs.append(f"scores.{l} missing")
            continue
        for d in DIMENSIONS:
            if not isinstance(c.get(d), (int, float)) or not 1 <= c[d] <= 5:
                probs.append(f"scores.{l}.{d}={c.get(d)}")
    ov = entry.get("overall") or {}
    if any(not isinstance(ov.get(l), (int, float)) for l in letters):
        probs.append(f"bad overall {ov}")
    us = entry.get("usefulness") or {}
    for p in PERSONAS:
        if any(not isinstance((us.get(p) or {}).get(l), (int, float)) for l in letters):
            probs.append(f"bad usefulness.{p}")
    return probs


def remap(entry: dict, lmap: dict) -> dict:
    """Translate letters to candidate ids everywhere in a validated entry."""
    return {
        "ranking": [lmap[x] for x in entry["ranking"]],
        "overall": {lmap[k]: v for k, v in entry["overall"].items() if k in lmap},
        "scores": {lmap[k]: v for k, v in entry["scores"].items() if k in lmap},
        "usefulness": {p: {lmap[k]: v for k, v in (entry["usefulness"].get(p) or {}).items() if k in lmap}
                       for p in PERSONAS},
        "reason": entry.get("reason", ""),
    }


def json_schema(n_cands: int) -> dict:
    """JSON schema of a judge reply for n_cands candidates (used to constrain local models served by Ollama)."""
    letters = [chr(65 + i) for i in range(n_cands)]
    score = {"type": "integer", "minimum": 1, "maximum": 5}
    per_cand = {"type": "object", "properties": {**{d: score for d in DIMENSIONS},
                                                 "severe_error": {"type": "boolean"},
                                                 "errors": {"type": "array", "items": {"type": "string"}}},
                "required": DIMENSIONS + ["severe_error", "errors"]}
    by_letter = lambda v: {"type": "object", "properties": {l: v for l in letters}, "required": letters}
    item = {"type": "object", "properties": {
        "item_id": {"type": "string"},
        "scores": by_letter(per_cand),
        "usefulness": {"type": "object", "properties": {p: by_letter(score) for p in PERSONAS}, "required": list(PERSONAS)},
        "overall": by_letter(score),
        "ranking": {"type": "array", "items": {"type": "string", "enum": letters}},
        "reason": {"type": "string"}},
        "required": ["item_id", "scores", "usefulness", "overall", "ranking", "reason"]}
    return {"type": "object", "properties": {"items": {"type": "array", "items": item}}, "required": ["items"]}
