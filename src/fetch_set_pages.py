"""Extract a company's data from the public SET quote pages into data/live/{SYMBOL}.json (or --out).

    python src/fetch_set_pages.py PTT
    python src/fetch_set_pages.py PTT KBANK AOT --out data/raw --pause 5

How it works: Selenium opens each company's pages in a regular Chrome window, as a reader would:

    /en/market/product/stock/quote/{SYMBOL}/factsheet
    /en/market/product/stock/quote/{SYMBOL}/financial-statement/company-highlights

While a page renders, it loads its own data (financial statements, ratios, prices, shareholders, dividends) as
JSON. The script reads those responses from Chrome's network log, so it sends no requests of its own, and parses
the two text fields that are only in the rendered HTML (business description, dividend policy). The result has
the same structure as the files in data/raw/, so src/build_cards.py and app.py read it unchanged.

Be considerate: one company at a time, a pause between pages, and saved files are reused (--refresh to fetch
again). The script does not hide that it is automated. If the site refuses the page (an "Access denied" page),
it stops and reports it; use the saved data instead.

Needs Google Chrome and `pip install selenium beautifulsoup4` (Selenium 4 downloads a matching chromedriver).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.set.or.th"
PAGES = ["/en/market/product/stock/quote/{S}/factsheet",
         "/en/market/product/stock/quote/{S}/financial-statement/company-highlights"]

# feed name in data/raw/*.json  ->  pattern of the JSON URL the pages load ({S} = symbol)
FEEDS = {
    "overview": r"/api/set/stock/{S}/overview\?lang=en$",
    "info": r"/api/set/stock/{S}/info\?lang=en$",
    "balance_sheet": r"/api/set/factsheet/{S}/financialstatement\?accountType=balance_sheet&lang=en$",
    "income_statement": r"/api/set/factsheet/{S}/financialstatement\?accountType=income_statement&lang=en$",
    "cash_flow": r"/api/set/factsheet/{S}/financialstatement\?accountType=cash_flow&lang=en$",
    "financial_ratio": r"/api/set/factsheet/{S}/financial-ratio\?lang=en$",
    "financial_growth": r"/api/set/factsheet/{S}/financial-growth\?lang=en$",
    "price_performance": r"/api/set/factsheet/{S}/price-performance\?lang=en$",
    "highlight_financial": r"/api/set/stock/{S}/company-highlight/financial-data\?lang=en$",
    "highlight_trading": r"/api/set/stock/{S}/company-highlight/trading-stat\?lang=en$",
    "shareholder": r"/api/set/stock/{S}/shareholder\?lang=en$",
    "dividend_xd": r"/api/set/stock/{S}/corporate-action/historical\?caType=XD&lang=en$",
}
# fields of the "profile" feed: most come from the overview feed, the two long texts from the factsheet HTML
PROFILE_FROM_OVERVIEW = ("symbol", "name", "market", "industry", "industryName", "sector", "sectorName",
                         "cgScore", "setesgRating")
BLOCKED = re.compile(r"Access denied|Error 15|request was blocked", re.I)


# ---- browser
def open_browser(headless: bool = False):
    """Start Chrome through Selenium with the network log enabled (no other changes to the browser)."""
    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,900")
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    return webdriver.Chrome(options=opts)


def load_page(driver, url: str, wait_s: float = 20) -> str:
    """Open a page, wait until its tables have rendered, scroll to the bottom so lazy sections load,
    and return the page HTML. Raises RuntimeError if the site refused the page."""
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    driver.get(url)
    try:
        WebDriverWait(driver, wait_s).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except TimeoutException:
        pass                                             # checked below: blocked page, or a page without tables
    if BLOCKED.search(driver.title or "") or BLOCKED.search(driver.page_source[:5000]):
        raise RuntimeError(f"The site refused {url} (access denied). Stopping; use the saved data instead.")
    last = driver.execute_script("return document.body.scrollHeight")
    for _ in range(8):                                   # scroll until the page stops growing
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.6)
        h = driver.execute_script("return document.body.scrollHeight")
        if h == last:
            break
        last = h
    time.sleep(1.0)                                      # let the last responses finish
    return driver.page_source


def captured_json(driver) -> dict[str, tuple[int, object]]:
    """JSON responses received since the last call, from Chrome's network log: {url: (status, body)}."""
    out = {}
    for entry in driver.get_log("performance"):
        msg = json.loads(entry["message"])["message"]
        if msg.get("method") != "Network.responseReceived":
            continue
        resp = msg["params"]["response"]
        if "/api/set/" not in resp["url"] or "json" not in (resp.get("mimeType") or ""):
            continue
        try:
            body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": msg["params"]["requestId"]})
            out[resp["url"]] = (resp["status"], json.loads(body["body"]))
        except Exception:                                # body no longer available, or not JSON
            out.setdefault(resp["url"], (resp["status"], None))
    return out


# ---- parsing
def profile_texts(html: str) -> dict:
    """Business description and dividend policy from the rendered factsheet page."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    biz = soup.find("div", class_="business-desc")
    if biz:
        out["businessType"] = biz.get_text(" ", strip=True)
    lab = soup.find("label", string=re.compile(r"Dividend Policy", re.I))
    if lab:
        span = lab.find_next_sibling("span")
        out["dividendPolicy"] = (span.get_text(" ", strip=True) if span
                                 else lab.parent.get_text(" ", strip=True).replace(lab.get_text(strip=True), "").strip())
    return out


def assemble(symbol: str, responses: dict, html: str) -> dict:
    """Map the captured responses onto the feed names used in data/raw/ and add the profile record."""
    rec = {"_meta": {"symbol": symbol, "fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "source": "SET quote pages", "errors": {}}}
    for name, pat in FEEDS.items():
        rx = re.compile(re.escape(BASE) + pat.replace("{S}", re.escape(symbol)), re.I)
        hit = next(((st, body) for url, (st, body) in responses.items() if rx.search(url)), None)
        if hit and hit[0] == 200 and hit[1] is not None:
            rec[name] = hit[1]
        else:
            rec["_meta"]["errors"][name] = hit[0] if hit else "not loaded by the pages"
    ov = rec.get("overview") or {}
    rec["profile"] = {k: ov.get(k) for k in PROFILE_FROM_OVERVIEW} | profile_texts(html)
    if not rec["profile"].get("businessType"):
        rec["_meta"]["errors"]["profile"] = "business description not found on the factsheet page"
    return rec


# ---- one company, several companies
def fetch_company(driver, symbol: str, pause: float = 3.0) -> dict:
    """Open the company's pages one after another and return its record (data/raw/ format)."""
    symbol = symbol.upper()
    captured_json(driver)                                # discard log entries from earlier pages
    responses, factsheet_html = {}, ""
    for i, p in enumerate(PAGES):
        html = load_page(driver, BASE + p.format(S=symbol))
        responses.update(captured_json(driver))
        if i == 0:
            factsheet_html = html
        time.sleep(pause)
    return assemble(symbol, responses, factsheet_html)


def fetch_symbols(symbols: list[str], out_dir: Path, pause: float = 3.0, refresh: bool = False,
                  headless: bool = False) -> dict[str, dict]:
    """Fetch several companies with one browser, saving each to out_dir/{SYMBOL}.json; saved files are reused."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results, driver = {}, None
    try:
        for s in [x.upper() for x in symbols]:
            path = out_dir / f"{s}.json"
            if path.exists() and not refresh:
                results[s] = json.loads(path.read_text(encoding="utf-8"))
                print(f"{s}: saved copy {path}")
                continue
            driver = driver or open_browser(headless)
            rec = fetch_company(driver, s, pause)
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            errs = rec["_meta"]["errors"]
            print(f"{s}: {len(FEEDS) + 1 - len(errs)}/{len(FEEDS) + 1} feeds -> {path}" + (f"; missing {errs}" if errs else ""))
            results[s] = rec
    finally:
        if driver:
            driver.quit()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+", help="SET tickers, e.g. PTT KBANK")
    ap.add_argument("--out", default="data/live", help="output folder (data/raw holds the study's companies)")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between pages")
    ap.add_argument("--refresh", action="store_true", help="fetch again even if a saved file exists")
    ap.add_argument("--headless", action="store_true", help="run Chrome without a window")
    a = ap.parse_args()
    fetch_symbols(a.symbols, ROOT / a.out, a.pause, a.refresh, a.headless)
