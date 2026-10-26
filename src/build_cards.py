"""Build English data cards + gold fact tables from raw SET JSON feeds.

Usage:
    python src/build_cards.py                      # all raw files, all formats
    python src/build_cards.py --symbols PTT KBANK  # subset
    python src/build_cards.py --raw data/live --out data/new   # companies fetched with src/fetch_set_pages.py

Input : data/raw/{SYMBOL}.json   (one file per company, 13 records from the SET quote pages; --raw for another folder)
Output: data/cards/{SYMBOL}_{SNAP}_{FORMAT}.txt   the text the model will read
        data/facts/{SYMBOL}_{SNAP}.json           every number in the card (gold checker)
        data/cards/_card_stats.csv                 size of every card

Snapshots
    T1 = annual view: latest full fiscal year ending on/before 2025-12-31 vs the year before,
         market data at calendar year-end 2025, dividends recorded up to that date.
    T2 = latest view: latest reported period vs same period a year earlier,
         market data as of the latest close.

Formats (the "JSON conversion strategies" to compare)
    text     - plain "Label: value" lines
    markdown - section headings + tables
    json     - compact normalised JSON (millions of THB, rounded)
    rawjson  - trimmed raw SET feeds (original units: thousands of THB, unrounded)

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CARDS = ROOT / "data" / "cards"
FACTS = ROOT / "data" / "facts"

T1_CUTOFF = "2025-12-31"
FORMATS = ["text", "markdown", "markdown_bn", "json", "rawjson"]
THAI = re.compile(r"[฀-๿]")

# --------------------------------------------------------------------------------------
# Statement line items to keep, per SET account layout (accountFormId)
#   6 = general companies, 1 = banks, 4 = insurers. Unknown layouts fall back to totals.
# Each entry: raw accountName (whitespace-normalised) -> display label
# --------------------------------------------------------------------------------------
INCOME_ITEMS = {
    "6": {
        "Revenue From Operations": "Revenue from operations",
        "Total Revenues": "Total revenues",
        "EBITDA": "EBITDA",
        "EBIT": "EBIT",
        "Net Profit : Owners Of The Parent": "Net profit (owners of the parent)",
        "EPS (B.)": "EPS (THB per share)",
    },
    "1": {
        "Net Interest Income": "Net interest income",
        "Net Fees And Service Expenses": "Net fee and service income",
        "(Reversal Of) Expected Credit Losses": "Expected credit losses",
        "Net Income before Tax": "Profit before tax",
        "Net Profit : Owners Of The Parent": "Net profit (owners of the parent)",
        "EPS (B.)": "EPS (THB per share)",
    },
    "4": {
        "Insurance Revenue": "Insurance revenue",
        "Insurance Service Result": "Insurance service result",
        "Net Investment Result": "Net investment result",
        "EBT": "Profit before tax",
        "Net Profit : Owners Of The Parent": "Net profit (owners of the parent)",
        "EPS (B.)": "EPS (THB per share)",
    },
    "2": {  # finance / leasing companies
        "Revenue From Operations": "Revenue from operations",
        "Total Revenue": "Total revenues",
        "(Reversal Of) Expected Credit Losses": "Expected credit losses",
        "EBIT": "EBIT",
        "Net Profit : Owners Of The Parent": "Net profit (owners of the parent)",
        "EPS (B.)": "EPS (THB per share)",
    },
    "3": {  # securities companies
        "Revenue From Operations": "Revenue from operations",
        "Total Revenues": "Total revenues",
        "Expected Credit Losses": "Expected credit losses",
        "EBT": "Profit before tax",
        "Net Profit : Owners Of The Parent": "Net profit (owners of the parent)",
        "EPS (B.)": "EPS (THB per share)",
    },
}
BALANCE_ITEMS = {
    "6": {
        "Cash And Cash Equivalents": "Cash and cash equivalents",
        "Current Assets": "Current assets",
        "Total Assets": "Total assets",
        "Current Liabilities": "Current liabilities",
        "Total Liabilities": "Total liabilities",
        "Shareholders' Equity": "Shareholders' equity",
    },
    "1": {
        "Net Loans and Accured Int.": "Net loans and accrued interest",
        "Total Assets": "Total assets",
        "Deposits": "Deposits",
        "Total Liabilities": "Total liabilities",
        "Shareholders' Equity": "Shareholders' equity",
    },
    "4": {
        "Cash And Cash Equivalents": "Cash and cash equivalents",
        "Total Assets": "Total assets",
        "Insurance Liabilities": "Insurance liabilities",
        "Total Liabilities": "Total liabilities",
        "Shareholders' Equity": "Shareholders' equity",
    },
    "2": {
        "Net Loans and Accured Int.": "Net loans and accrued interest",
        "Total Assets": "Total assets",
        "Borrowings & Deposits": "Borrowings and deposits",
        "Total Liabilities": "Total liabilities",
        "Shareholders' Equity": "Shareholders' equity",
    },
    "3": {
        "Cash And Cash Equivalents": "Cash and cash equivalents",
        "Securities And Derivatives Business Receivables": "Securities business receivables",
        "Total Assets": "Total assets",
        "Total Liabilities": "Total liabilities",
        "Shareholders' Equity": "Shareholders' equity",
    },
}
CASHFLOW_ITEMS = {
    "Operating Cash Flow": "Operating cash flow",
    "Investing Cash Flow": "Investing cash flow",
    "Financing Cash Flow": "Financing cash flow",
    "Net Cash Flow": "Net cash flow",
}
LAYOUT_NAME = {"6": "general company", "1": "bank", "4": "insurer", "2": "finance company",
               "3": "securities company", "5": "property fund / REIT"}

# Thai shareholder names that are public bodies -> English. Every other Thai name is labelled by its type only
# (see shareholder_name), so no individual's or company's name is translated or reproduced.
THAI_NAME_MAP = {
    "กระทรวงการคลัง": "Ministry of Finance",
    "กองทุนรวม วายุภักษ์หนึ่ง": "Vayupak Fund 1",
    "สำนักงานประกันสังคม": "Social Security Office",
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def norm(s: str) -> str:
    """Collapse whitespace (and zero-width spaces) in a feed string."""
    return re.sub(r"\s+", " ", (s or "").replace("​", "")).strip()


def shareholder_name(raw: str) -> str:
    """English display name for a shareholder: English names are kept, a few public bodies are translated,
    and any other Thai name is replaced by its type (individual, bank, fund, government body, company).
    """
    n = norm(raw)
    if not THAI.search(n):
        return n
    if n in THAI_NAME_MAP:
        return THAI_NAME_MAP[n]
    if n.startswith(("นาย", "นาง", "น.ส.")):
        return "Individual shareholder (Thai name)"
    if n.startswith("ธนาคาร"):
        return "Thai bank (Thai name)"
    if n.startswith("กองทุน"):
        return "Thai fund (Thai name)"
    if n.startswith(("กระทรวง", "สำนักงาน")):
        return "Thai government body (Thai name)"
    if n.startswith("บริษัท"):
        return "Thai company (Thai name)"
    return "Shareholder (Thai name)"


def d10(s: str | None) -> str | None:
    """First 10 characters of an ISO timestamp (the date), or None."""
    return s[:10] if s else None


def fmt_date(s: str | None) -> str:
    """ISO date -> '31 Dec 2025' ('n/a' if missing)."""
    return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%d %b %Y") if s else "n/a"


def months_between(b: str, e: str) -> int:
    """Length of a reporting period in months, from its begin and end dates."""
    b1, e1 = datetime.strptime(b[:10], "%Y-%m-%d"), datetime.strptime(e[:10], "%Y-%m-%d")
    return round(((e1 - b1).days + 1) / 30.44)


def period_label(entry: dict) -> str:
    """Readable period label: 'FY2025 (12 months to 31 Dec 2025)' or '6M 2026 (6 months to 30 Jun 2026)'."""
    m = months_between(entry["beginDate"], entry["endDate"])
    end = datetime.strptime(entry["endDate"][:10], "%Y-%m-%d")
    if m >= 12:
        return f"FY{end.year} (12 months to {end.strftime('%d %b %Y')})"
    return f"{m}M {end.year} ({m} months to {end.strftime('%d %b %Y')})"


def pct_change(cur, prior):
    # % change is only meaningful when both values are positive (not for losses / cash outflows)
    """Percentage change from prior to current; None when either value is missing or not positive
    (a % change of a loss or a cash outflow is misleading).
    """
    if cur is None or prior is None or prior <= 0 or cur < 0:
        return None
    return (cur - prior) / abs(prior) * 100


def r(x, nd=2):
    """Round to `nd` decimals, keeping None."""
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------------------
# Period selection
# --------------------------------------------------------------------------------------
def pick_periods(stmts: list[dict], snap: str):
    """Return (current, prior) entries from a statement feed for a snapshot."""
    if not stmts:
        return None, None
    if snap == "T2":
        cur = stmts[0]
    else:
        fy = [s for s in stmts if months_between(s["beginDate"], s["endDate"]) >= 12
              and s["endDate"][:10] <= T1_CUTOFF]
        if not fy:
            return None, None
        cur = max(fy, key=lambda s: s["endDate"])
    prior = next((s for s in stmts if s["quarter"] == cur["quarter"] and s["year"] == cur["year"] - 1), None)
    return cur, prior


def find_same(entries: list[dict], ref: dict | None):
    """Entry of another feed (balance sheet, cash flow, ratios) for the same quarter and year as `ref`."""
    if not ref or not entries:
        return None
    return next((e for e in entries if e.get("quarter") == ref["quarter"] and e.get("year") == ref["year"]), None)


def accounts_to_millions(entry: dict | None, wanted: dict | None) -> dict:
    """{label: value in THB million} (EPS kept in THB). wanted=None -> totals only."""
    out = {}
    if not entry:
        return out
    for a in entry.get("accounts", []):
        name = norm(a["accountName"])
        if wanted is not None:
            if name not in wanted:
                continue
            label = wanted[name]
        else:
            if a.get("level") != -1:
                continue
            label = name
        amt = a.get("amount")
        if amt is None:
            continue
        if name.startswith("EPS"):
            out[label] = round(amt, 2)
        else:
            # divider=1000 means the feed is in THB thousands -> divide by 1000 for THB million
            out[label] = round(amt / 1000.0, 1)
    return out


# --------------------------------------------------------------------------------------
# Build the canonical snapshot record
# --------------------------------------------------------------------------------------
def build_snapshot(d: dict, snap: str) -> dict | None:
    """Build the normalised record of one company for one snapshot (T1 or T2), or None if no usable period.

    The record holds the company profile, current vs prior statements (THB million), ratios, market data,
    dividends and top shareholders; every card format and the fact list are rendered from it.
    """
    prof, ov, info = d.get("profile", {}), d.get("overview", {}), d.get("info", {})
    inc_cur, inc_pri = pick_periods(d.get("income_statement", []), snap)
    if not inc_cur:
        return None
    form = inc_cur.get("accountFormId", "?")
    bal_cur, bal_pri = find_same(d.get("balance_sheet", []), inc_cur), find_same(d.get("balance_sheet", []), inc_pri)
    cf_cur, cf_pri = find_same(d.get("cash_flow", []), inc_cur), find_same(d.get("cash_flow", []), inc_pri)
    rt_cur, rt_pri = find_same(d.get("financial_ratio", []), inc_cur), find_same(d.get("financial_ratio", []), inc_pri)

    def rows(cur_map, pri_map, unit):
        """Statement rows {item, current, prior, yoy_pct, unit} in the feed's order."""
        out = []
        for label in cur_map.keys() | pri_map.keys():
            c, p = cur_map.get(label), pri_map.get(label)
            out.append({"item": label, "current": c, "prior": p,
                        "yoy_pct": r(pct_change(c, p), 1), "unit": unit})
        order = list(cur_map.keys()) + [k for k in pri_map if k not in cur_map]
        return sorted(out, key=lambda x: order.index(x["item"]))

    inc_w, bal_w = INCOME_ITEMS.get(form), BALANCE_ITEMS.get(form)
    inc_c, inc_p = accounts_to_millions(inc_cur, inc_w), accounts_to_millions(inc_pri, inc_w)
    income = rows(inc_c, inc_p, "THB m")
    for row in income:
        if row["item"].startswith("EPS"):
            row["unit"] = "THB"
    balance = rows(accounts_to_millions(bal_cur, bal_w), accounts_to_millions(bal_pri, bal_w), "THB m")
    cash = rows(accounts_to_millions(cf_cur, CASHFLOW_ITEMS), accounts_to_millions(cf_pri, CASHFLOW_ITEMS), "THB m")

    def ratio_map(e):
        """{ratio name: value} for one financial-ratio entry."""
        return {norm(x["accountName"]): r(x["value"]) for x in (e or {}).get("data", []) if x.get("value") is not None}
    rc, rp = ratio_map(rt_cur), ratio_map(rt_pri)
    ratios = [{"item": k, "current": rc.get(k), "prior": rp.get(k)} for k in list(rc) + [k for k in rp if k not in rc]]

    # ---------------- market data ----------------
    trading = d.get("highlight_trading", []) or []
    if snap == "T2":
        tr = next((t for t in trading if t.get("period") == "YTD"), None)
        pp = (d.get("price_performance") or {}).get("stock", {})
        sec = (d.get("price_performance") or {}).get("sector", {})
        mkt = (d.get("price_performance") or {}).get("market", {})
        asof = d10((tr or {}).get("date")) or d10(info.get("statisticsAsOf"))
        market = {
            "as_of": asof,
            "last_price_thb": r(info.get("last")),
            "price_change_ytd_pct": r(pp.get("ytdPercentChange")),
            "price_change_1m_pct": r(pp.get("oneMonthPercentChange")),
            "price_change_3m_pct": r(pp.get("threeMonthPercentChange")),
            "price_change_6m_pct": r(pp.get("sixMonthPercentChange")),
            "high_52w_thb": r(info.get("high52Weeks")),
            "low_52w_thb": r(info.get("low52Weeks")),
            "market_cap_thb_m": r((info.get("marketCap") or 0) / 1e6, 1) if info.get("marketCap") else None,
            "pe_x": r(info.get("peRatio")),
            "pbv_x": r(info.get("pbRatio")),
            "dividend_yield_pct": r(info.get("dividendYield")),
            "beta": r((tr or {}).get("beta")),
            "sector_ytd_pct": r(sec.get("ytdPercentChange")) if sec.get("symbol") else None,
            "sector_pe_x": r(sec.get("peRatio")) if sec.get("symbol") else None,
            "set_index_ytd_pct": r(mkt.get("ytdPercentChange")),
        }
    else:
        year = str(int(inc_cur["endDate"][:4]))
        tr = next((t for t in trading if t.get("period") == year), None)
        if tr is None and trading:
            # fiscal year ends mid-year: use the calendar year-end on/before cutoff
            cands = [t for t in trading if t.get("period", "").isdigit() and t["date"][:10] <= T1_CUTOFF]
            tr = max(cands, key=lambda t: t["date"]) if cands else None
        asof = d10((tr or {}).get("date")) or T1_CUTOFF
        market = {} if not tr else {
            "as_of": asof,
            "year_end_close_thb": r(tr.get("close")),
            "price_change_year_pct": r(tr.get("percentChange")),
            "year_high_thb": r(tr.get("high")),
            "year_low_thb": r(tr.get("low")),
            "market_cap_thb_m": r(tr["marketCap"] / 1e6, 1) if tr.get("marketCap") else None,
            "pe_x": r(tr.get("pe")),
            "pbv_x": r(tr.get("pbv")),
            "dividend_yield_pct": r(tr.get("dividendYield")),
            "beta": r(tr.get("beta")),
        }
    market = {k: v for k, v in market.items() if v is not None}

    # ---------------- dividends ----------------
    cutoff = market.get("as_of") or "9999"
    divs = []
    for x in d.get("dividend_xd", []) or []:
        rec = d10(x.get("recordDate") or x.get("bookCloseDate") or x.get("xdate"))
        if x.get("dividendType") != "Cash Dividend" or x.get("dividend") is None or not rec or rec > cutoff:
            continue
        divs.append({"record_date": rec, "dps_thb": r(x["dividend"], 4),
                     "period_from": d10(x.get("beginOperation")), "period_to": d10(x.get("endOperation")),
                     "payment_date": d10(x.get("paymentDate"))})
    divs = sorted(divs, key=lambda x: x["record_date"], reverse=True)[:4]

    # ---------------- shareholders ----------------
    sh = d.get("shareholder") or {}
    top = [{"name": shareholder_name(x["name"]), "pct": r(x.get("percentOfShare"))}
           for x in (sh.get("majorShareholders") or [])[:5]]
    holders = {
        "as_of": d10(sh.get("bookCloseDate")),
        "total_shareholders": sh.get("totalShareholder"),
        "free_float_pct": r((sh.get("freeFloat") or {}).get("percentFreeFloat")),
        "top5": top,
    }

    def clip(text, n):
        """Shorten long profile text to at most `n` characters, ending at a full sentence where possible."""
        text = norm(text)
        if len(text) <= n:
            return text
        cut = text[:n]
        return cut[: cut.rfind(". ") + 1] if ". " in cut else cut.rstrip() + "..."

    return {
        "symbol": d["_meta"]["symbol"],
        "snapshot": snap,
        "company": {
            "name": prof.get("name") or ov.get("name"),
            "market": ov.get("market"),
            "industry": prof.get("industryName"),
            "sector": prof.get("sectorName"),
            "statement_layout": LAYOUT_NAME.get(form, f"form {form}"),
            "business": clip(prof.get("businessType", ""), 700),
            "dividend_policy": clip(prof.get("dividendPolicy", ""), 350),
            "set_esg_rating": prof.get("setesgRating") or None,
            "cg_score": prof.get("cgScore"),
            "indices": ov.get("indices") or [],
        },
        "periods": {
            "current": period_label(inc_cur),
            "prior": period_label(inc_pri) if inc_pri else None,
            "statement_type": inc_cur.get("fsTypeDescription"),
            "audit_status": inc_cur.get("status"),
        },
        "income": income,
        "balance": balance,
        "cash_flow": cash,
        "ratios": ratios,
        "market": market,
        "dividends": divs,
        "shareholders": holders,
        "_raw_refs": {"inc": [inc_cur, inc_pri], "bal": [bal_cur, bal_pri], "cf": [cf_cur, cf_pri],
                      "ratio": [rt_cur, rt_pri], "trading": tr},
    }


# --------------------------------------------------------------------------------------
# Gold facts: every number shown in the normalised card, with unit
# --------------------------------------------------------------------------------------
MARKET_UNITS = {
    "_thb_m": "THB m", "_thb": "THB", "_pct": "%", "_x": "x", "beta": "x",
}


def unit_for(key: str) -> str:
    """Unit of a market-data field from its key suffix (THB, %, x, THB m, ...)."""
    for suf, u in MARKET_UNITS.items():
        if key.endswith(suf):
            return u
    return ""


def facts_for(s: dict) -> list[dict]:
    """List every number shown in a card as a fact {key, label, value, unit, period}; the gold checker matches
    the numbers in a summary against this list.
    """
    f = []
    def add(key, label, value, unit, period=None):
        """Append one fact if its value is present."""
        if value is not None:
            f.append({"key": key, "label": label, "value": value, "unit": unit, "period": period})
    for sec in ("income", "balance", "cash_flow"):
        for row in s[sec]:
            add(f"{sec}.{row['item']}.current", row["item"], row["current"], row["unit"], "current")
            add(f"{sec}.{row['item']}.prior", row["item"], row["prior"], row["unit"], "prior")
            add(f"{sec}.{row['item']}.yoy", row["item"] + " YoY change", row["yoy_pct"], "%")
    for row in s["ratios"]:
        u = "%" if "(%)" in row["item"] else "x"
        add(f"ratio.{row['item']}.current", row["item"], row["current"], u, "current")
        add(f"ratio.{row['item']}.prior", row["item"], row["prior"], u, "prior")
    for k, v in s["market"].items():
        if k != "as_of":
            add(f"market.{k}", k, v, unit_for(k))
    for i, dv in enumerate(s["dividends"]):
        add(f"dividend.{i}.dps", "Dividend per share", dv["dps_thb"], "THB")
    h = s["shareholders"]
    add("holders.total", "Number of shareholders", h.get("total_shareholders"), "count")
    add("holders.free_float", "Free float", h.get("free_float_pct"), "%")
    for i, t in enumerate(h["top5"]):
        add(f"holders.top{i+1}", t["name"], t["pct"], "%")
    add("company.cg_score", "CG score", s["company"].get("cg_score"), "score")
    # numbers quoted inside the business description / dividend policy text
    for fld in ("business", "dividend_policy"):
        for i, m in enumerate(re.finditer(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|percent)?",
                                          s["company"].get(fld) or "")):
            v = float(m.group(1).replace(",", ""))
            add(f"text.{fld}.{i}", f"number in {fld} text", v, "%" if m.group(2) else "text")
    return f


# --------------------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------------------
def n(v, nd=None):
    """Format a number with thousands separators (1 decimal for large values, 2 otherwise); 'n/a' for None."""
    if v is None:
        return "n/a"
    if isinstance(v, (int,)) and nd is None:
        return f"{v:,}"
    if nd is None:
        nd = 1 if abs(v) >= 1000 else 2
    return f"{v:,.{nd}f}"


def yoy(v):
    """Format a year-on-year change as '+12.3%' ('n/a' if missing)."""
    return "n/a" if v is None else f"{v:+.1f}%"


MARKET_LABELS = {
    "last_price_thb": "Last price (THB)", "year_end_close_thb": "Year-end closing price (THB)",
    "price_change_ytd_pct": "Price change year-to-date (%)", "price_change_year_pct": "Price change over the year (%)",
    "price_change_1m_pct": "Price change 1 month (%)", "price_change_3m_pct": "Price change 3 months (%)",
    "price_change_6m_pct": "Price change 6 months (%)",
    "high_52w_thb": "52-week high (THB)", "low_52w_thb": "52-week low (THB)",
    "year_high_thb": "High during the year (THB)", "year_low_thb": "Low during the year (THB)",
    "market_cap_thb_m": "Market capitalisation (THB million)", "pe_x": "P/E (x)", "pbv_x": "P/BV (x)",
    "dividend_yield_pct": "Dividend yield (%)", "beta": "Beta",
    "sector_ytd_pct": "Sector price change year-to-date (%)", "sector_pe_x": "Sector P/E (x)",
    "set_index_ytd_pct": "SET index change year-to-date (%)",
}


def div_period(dv):
    """Describe the results period a dividend was paid for."""
    if dv.get("period_from") and dv.get("period_to"):
        return f"for results {fmt_date(dv['period_from'])} to {fmt_date(dv['period_to'])}"
    return "period not stated"


def header_lines(s):
    """Company header shared by the text and markdown cards (name, market, business, periods, units)."""
    c, p = s["company"], s["periods"]
    lines = [
        f"Company: {c['name']} ({s['symbol']})",
        f"Market: {c['market']} | Industry: {c['industry']} | Sector: {c['sector']}",
        f"Business: {c['business']}",
    ]
    if c["dividend_policy"]:
        lines.append(f"Dividend policy: {c['dividend_policy']}")
    extra = []
    if c.get("set_esg_rating"):
        extra.append(f"SET ESG rating: {c['set_esg_rating']}")
    if c.get("cg_score"):
        extra.append(f"CG score: {c['cg_score']} of 5")
    if c.get("indices"):
        extra.append("Index member: " + ", ".join(c["indices"]))
    if extra:
        lines.append(" | ".join(extra))
    lines.append(f"Reporting periods compared: {p['current']} vs {p['prior'] or 'n/a'} "
                 f"({p['statement_type']} statements, {p['audit_status']})")
    lines.append("Units: amounts in THB million unless stated; EPS and dividends in THB per share.")
    return lines


def render_text(s):
    """Plain-text card: labelled lines, current vs prior with YoY change (the format the judges see)."""
    L = header_lines(s)
    for title, sec in (("Income statement", "income"), ("Balance sheet", "balance"), ("Cash flow", "cash_flow")):
        if s[sec]:
            L.append(f"\n{title} (current vs prior, YoY change):")
            for row in s[sec]:
                nd = 1 if row["unit"] == "THB m" else 2
                L.append(f"- {row['item']}: {n(row['current'], nd)} vs {n(row['prior'], nd)} ({yoy(row['yoy_pct'])})")
    if s["ratios"]:
        L.append("\nFinancial ratios (current vs prior):")
        for row in s["ratios"]:
            L.append(f"- {row['item']}: {n(row['current'])} vs {n(row['prior'])}")
    if s["market"]:
        L.append(f"\nMarket data (as of {fmt_date(s['market'].get('as_of'))}):")
        for k, v in s["market"].items():
            if k != "as_of":
                L.append(f"- {MARKET_LABELS.get(k, k)}: {n(v, 1 if k.endswith('_thb_m') else 2)}")
    if s["dividends"]:
        L.append("\nRecent cash dividends:")
        for dv in s["dividends"]:
            L.append(f"- THB {dv['dps_thb']:g} per share, {div_period(dv)}, paid {fmt_date(dv['payment_date'])}")
    h = s["shareholders"]
    L.append(f"\nShareholders (as of {fmt_date(h['as_of'])}): {n(h['total_shareholders'])} holders, free float {n(h['free_float_pct'])}%")
    for t in h["top5"]:
        L.append(f"- {t['name']}: {n(t['pct'])}%")
    return "\n".join(L)


def money_scale(s):
    """THB billion when the company's largest statement amount is >= THB 100 billion, else THB million."""
    vals = [abs(r_[k]) for sec in ("income", "balance", "cash_flow") for r_ in s[sec]
            if r_["unit"] == "THB m" for k in ("current", "prior") if r_[k] is not None]
    return ("THB billion", 1000.0) if vals and max(vals) >= 100_000 else ("THB million", 1.0)


def render_markdown(s, auto_scale=False):
    """auto_scale=False: all amounts in THB million (format 'markdown').
    auto_scale=True : large companies shown in THB billion, unit repeated in every table header ('markdown_bn')."""
    unit, div = money_scale(s) if auto_scale else ("THB million", 1.0)
    def money(v):
        """Format an amount in the card's money unit."""
        return n(None if v is None else v / div, 2 if div > 1 else 1)
    head = header_lines(s)
    if auto_scale:
        head = [h if not h.startswith("Units:") else
                f"Units: all amounts in {unit} (as written in every table header); EPS and dividends in THB per share."
                for h in head]
    L = ["## " + head[0].replace("Company: ", "")]
    L += ["- " + x for x in head[1:]]
    for title, sec in (("Income statement", "income"), ("Balance sheet", "balance"), ("Cash flow", "cash_flow")):
        if s[sec]:
            col = f"Item ({unit})" if auto_scale else "Item"
            L += [f"\n### {title}", f"| {col} | Current | Prior | YoY |", "|---|---:|---:|---:|"]
            for r_ in s[sec]:
                if r_["unit"] == "THB m":
                    c, p = money(r_["current"]), money(r_["prior"])
                else:
                    c, p = n(r_["current"], 2), n(r_["prior"], 2)
                L.append(f"| {r_['item']} | {c} | {p} | {yoy(r_['yoy_pct'])} |")
    if s["ratios"]:
        L += ["\n### Financial ratios", "| Ratio | Current | Prior |", "|---|---:|---:|"]
        L += [f"| {r_['item']} | {n(r_['current'])} | {n(r_['prior'])} |" for r_ in s["ratios"]]
    if s["market"]:
        L += [f"\n### Market data (as of {fmt_date(s['market'].get('as_of'))})", "| Metric | Value |", "|---|---:|"]
        for k, v in s["market"].items():
            if k == "as_of":
                continue
            if k.endswith("_thb_m") and auto_scale:
                L.append(f"| Market capitalisation ({unit}) | {money(v)} |")
            else:
                L.append(f"| {MARKET_LABELS.get(k, k)} | {n(v, 1 if k.endswith('_thb_m') else 2)} |")
    if s["dividends"]:
        L += ["\n### Recent cash dividends", "| Period covered | DPS (THB) | Paid |", "|---|---:|---|"]
        L += [f"| {div_period(dv)} | {dv['dps_thb']:g} | {fmt_date(dv['payment_date'])} |"
              for dv in s["dividends"]]
    h = s["shareholders"]
    L += [f"\n### Shareholders (as of {fmt_date(h['as_of'])})",
          f"- Number of shareholders: {n(h['total_shareholders'])}; free float: {n(h['free_float_pct'])}%"]
    L += [f"- {t['name']}: {n(t['pct'])}%" for t in h["top5"]]
    return "\n".join(L)


def render_json(s):
    """Compact normalised JSON card (format experiment only)."""
    out = {k: v for k, v in s.items() if not k.startswith("_")}
    out["units"] = "THB million unless the field name says otherwise; EPS and dividends in THB per share"
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def render_rawjson(s, d):
    """Trimmed original feeds: original units (THB thousands), unrounded, no derived fields."""
    keep_acc = lambda e: None if not e else {
        "period": f"{e['quarter']} {e['year']}", "beginDate": d10(e["beginDate"]), "endDate": d10(e["endDate"]),
        "accounts": [{"accountName": norm(a["accountName"]), "amount": a["amount"], "divider": a.get("divider")}
                     for a in e.get("accounts", [])]}
    refs = s["_raw_refs"]
    prof = d.get("profile", {})
    out = {
        "profile": {k: prof.get(k) for k in ("symbol", "name", "market", "industryName", "sectorName",
                                              "businessType", "dividendPolicy", "setesgRating", "cgScore")},
        "income_statement": [keep_acc(e) for e in refs["inc"] if e],
        "balance_sheet": [keep_acc(e) for e in refs["bal"] if e],
        "cash_flow": [keep_acc(e) for e in refs["cf"] if e],
        "financial_ratio": [{"period": f"{e['quarter']} {e['year']}", "data": e["data"]} for e in refs["ratio"] if e],
        "market": refs["trading"] if s["snapshot"] == "T1" else {
            "info": {k: d["info"].get(k) for k in ("last", "high52Weeks", "low52Weeks", "marketCap", "peRatio",
                                                   "pbRatio", "dividendYield", "statisticsAsOf")},
            "price_performance": d.get("price_performance")},
        "dividends": s["dividends"],
        "shareholders": s["shareholders"],
    }
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# Command line: build every card and fact file
# --------------------------------------------------------------------------------------
def main():
    """Build cards and fact files for every company in the raw folder (or --symbols), both snapshots."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--raw", default=str(RAW), help="folder of raw company files (default data/raw)")
    ap.add_argument("--out", default=None, help="output folder holding cards/ and facts/ (default data)")
    args = ap.parse_args()
    raw = Path(args.raw)
    cards, facts_dir = (CARDS, FACTS) if args.out is None else (Path(args.out) / "cards", Path(args.out) / "facts")
    cards.mkdir(parents=True, exist_ok=True)
    facts_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*.json"))
    if args.symbols:
        wanted = {x.upper() for x in args.symbols}
        files = [f for f in files if f.stem.upper() in wanted]
    if not files:
        raise SystemExit(f"No raw files found in {raw}" + (f" for {args.symbols}" if args.symbols else ""))
    stats = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for snap in ("T1", "T2"):
            s = build_snapshot(d, snap)
            if s is None:
                print(f"{f.stem} {snap}: no usable periods, skipped")
                continue
            facts = facts_for(s)
            (facts_dir / f"{f.stem}_{snap}.json").write_text(json.dumps(
                {"symbol": f.stem, "snapshot": snap, "periods": s["periods"], "facts": facts},
                ensure_ascii=False, indent=1), encoding="utf-8")
            renders = {"text": render_text(s), "markdown": render_markdown(s), "markdown_bn": render_markdown(s, auto_scale=True),
                       "json": render_json(s), "rawjson": render_rawjson(s, d)}
            for fmt, txt in renders.items():
                (cards / f"{f.stem}_{snap}_{fmt}.txt").write_text(txt, encoding="utf-8")
                stats.append({"symbol": f.stem, "snapshot": snap, "format": fmt, "chars": len(txt),
                              "approx_tokens": round(len(txt) / 3.2), "thai_chars": len(THAI.findall(txt)),
                              "layout": s["company"]["statement_layout"], "current": s["periods"]["current"],
                              "prior": s["periods"]["prior"], "n_facts": len(facts)})
    # the stats table describes a full build; a subset into data/cards must not overwrite it
    if stats and not (args.symbols and cards == CARDS):
        with open(cards / "_card_stats.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)
    print(f"Wrote {len(stats)} cards for {len(files)} companies -> {cards}")


if __name__ == "__main__":
    main()
