"""Prompt variants for the stock-summary task. Each returns a chat `messages` list.

Add a variant by writing a function and registering it in PROMPTS.
The card text is produced by build_cards.py (text / markdown / markdown_bn / json / rawjson).
"""

# ---- one sentence per card format telling the model how the data is laid out and which unit it uses
FORMAT_HINT = {
    "text": "The company data is given as labelled lines.",
    "markdown": "The company data is given as Markdown sections and tables.",
    "json": "The company data is given as JSON. Amounts are in THB million unless the field name says otherwise.",
    "markdown_bn": "The company data is given as Markdown sections and tables. Check the unit in each table header "
                   "(THB billion or THB million).",
    "rawjson": "The company data is given as raw JSON from the stock exchange. Statement amounts are in THB thousands "
               "(see the 'divider' field); market capitalisation is in THB.",
}


def minimal(card: str, fmt: str) -> list[dict]:
    """One-line instruction plus the card (baseline prompt)."""
    return [{"role": "user", "content": f"Summarise this company for an investor in at most 500 tokens.\n\n{card}"}]


def guided(card: str, fmt: str) -> list[dict]:
    """Guided prompt: required structure, copy numbers with units, state periods, no causes/advice."""
    system = ("You are a careful equity research assistant. You write short, factual company summaries for "
              "investors, using only the data you are given.")
    user = f"""{FORMAT_HINT[fmt]}

Write an investor summary of this company in at most 300 words (it must stay under 500 tokens).
Structure it as:
1. Business: one or two sentences on what the company does.
2. Performance: revenue (or its main income line) and net profit for the current period versus the prior period, with growth rates.
3. Financial position: balance sheet strength, cash flow and key ratios.
4. Valuation and shares: price performance, P/E, P/BV, dividend yield and dividends.
5. Key takeaway: one or two sentences.

Rules:
- Use only numbers that appear in the data, copied exactly, with their units (THB million, %, x).
- Say which period each figure refers to.
- Do not invent facts, forecasts or news, and do not give buy or sell recommendations.

DATA:
{card}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def guided_v2(card: str, fmt: str) -> list[dict]:
    """guided + fixes for errors seen in exp1: unit slips, invented causes, misread ratios, chat filler."""
    system = ("You are a careful equity research assistant. You write short, factual company summaries for "
              "investors, using only the data you are given.")
    user = f"""{FORMAT_HINT[fmt]}

Write an investor summary of this company in at most 300 words (it must stay under 500 tokens).
Structure it as:
1. Business: one or two sentences on what the company does.
2. Performance: revenue (or its main income line) and net profit for the current period versus the prior period, with growth rates.
3. Financial position: balance sheet, cash flow and key ratios.
4. Valuation and shares: price performance, P/E, P/BV, dividend yield and dividends.
5. Key takeaway: one or two sentences.

Rules:
- Copy numbers exactly as they appear in the data, keeping the unit next to them. Never rescale a number:
  "1,234,567.8" in a THB million table is THB 1,234,567.8 million (about THB 1.23 trillion), not THB 1,234.6 million.
- Say which period each figure refers to.
- Do not explain causes, compare with peers or judge a ratio as good or bad unless the data says so.
- Do not invent facts, forecasts or news, and do not give buy or sell recommendations.
- Start directly with the summary. No greeting, no introduction, no closing offer.

DATA:
{card}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def concise(card: str, fmt: str) -> list[dict]:
    """Shorter guided prompt asking for a compact summary (format experiment only)."""
    system = "You summarise company data accurately for investors."
    user = f"""{FORMAT_HINT[fmt]}

In at most 150 words, summarise the five most important facts an investor should know about this company.
Use bullet points. Copy every number exactly as it appears in the data, with its unit and period.
Do not add information that is not in the data.

DATA:
{card}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---- registry used by every script (--prompts / prompt_id); guided_v2 is the prompt used throughout training
PROMPTS = {"minimal": minimal, "guided": guided, "guided_v2": guided_v2, "concise": concise}
