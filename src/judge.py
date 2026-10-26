"""Run AI judges over judging items in bulk.

Each item = one data card + K candidate summaries. The judge scores every candidate on the rubric in judge_prompt.py
(six 1-5 dimensions, severe-error flag, errors, usefulness per persona, overall 1-5) and ranks them.
Every item is judged in a forward and a reversed candidate order (--orders 2) to measure and cancel position bias.

Backends (pick with --backend):
  ollama   a local model served by Ollama                  --model <ollama model name>
  openai   any OpenAI-compatible chat-completions API      --model <id> --base-url <url> --api-key-env <ENV_VAR_NAME>
  claude   Claude Code CLI (`claude`), its own login        --model sonnet
  codex    Codex CLI (`codex`), its own login               --model <model id, or omit for the CLI default>
  cursor   Cursor CLI (`cursor-agent` / `agent`)            --model <id from `cursor-agent --list-models`>
  dry      print one prompt and exit (no call)

Examples:
  python src/judge.py --items data/judge_items/B.jsonl --run B_sonnet --backend claude --model sonnet --items-per-call 4
  python src/judge.py --items data/judge_items/B.jsonl --run B_gpt --backend codex --items-per-call 4
  python src/judge.py --items data/judge_items/B.jsonl --run look --backend dry

Output: outputs/judgments/{run}.jsonl  (resumable: re-run the same command to continue after a stop or a usage limit)
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompt import PERSONAS, build_prompt, json_schema, remap, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "judgments"


# ------------------------------------------------------------------ backends
def call_ollama(system, user, args, n_cands):
    """Judge through a local Ollama server, with the reply constrained to the judgment JSON schema."""
    body = {"model": args.model, "stream": False, "format": json_schema(n_cands),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": args.num_ctx, "num_predict": 1500 * args.items_per_call}}
    if args.think is not None:
        body["think"] = args.think
    req = urllib.request.Request(f"{args.ollama_url}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        res = json.loads(r.read())
    if res.get("prompt_eval_count", 0) >= args.num_ctx - 50:
        raise RuntimeError(f"prompt ({res['prompt_eval_count']} tokens) filled num_ctx={args.num_ctx}: raise --num-ctx")
    return res["message"]["content"], {"prompt_tokens": res.get("prompt_eval_count"), "output_tokens": res.get("eval_count")}


def call_openai(system, user, args, n_cands):
    """Judge through any OpenAI-compatible chat-completions endpoint (JSON mode, temperature 0)."""
    key = os.environ.get(args.api_key_env or "", "")
    body = {"model": args.model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        res = json.loads(r.read())
    u = res.get("usage", {})
    return res["choices"][0]["message"]["content"], {"prompt_tokens": u.get("prompt_tokens"),
                                                     "output_tokens": u.get("completion_tokens"),
                                                     "model_version": res.get("model")}


def _run_cli(cmd, stdin_text, args, cwd):
    """Run a command-line tool with the prompt on stdin (or none) and return its stdout; raise on a non-zero exit."""
    exe = shutil.which(cmd[0])
    if not exe:
        raise RuntimeError(f"'{cmd[0]}' not found on PATH")
    p = subprocess.run([exe] + cmd[1:], input=stdin_text, capture_output=True, text=True, encoding="utf-8",
                       timeout=args.timeout, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exit {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
    return p.stdout


def call_claude(system, user, args, n_cands):
    # Claude Code headless mode (uses the CLI's own login). Runs in an empty temp folder, tools denied.
    """Judge through the Claude Code CLI in headless mode (single turn, no tools)."""
    cmd = ["claude", "-p", "--output-format", "json", "--system-prompt", system,
           "--permission-mode", "dontAsk", "--max-turns", "1"]
    if args.model:
        cmd += ["--model", args.model]
    with tempfile.TemporaryDirectory() as tmp:
        out = _run_cli(cmd, user, args, tmp)
    res = json.loads(out)
    if res.get("is_error"):
        raise RuntimeError(f"claude error: {str(res.get('result'))[:300]}")
    u = res.get("usage", {})
    return res.get("result", ""), {"prompt_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                                   "cost_usd_estimate": res.get("total_cost_usd")}


def call_codex(system, user, args, n_cands):
    # Codex CLI non-interactive mode (uses the CLI's own login); prompt read from stdin ("-"), read-only sandbox.
    """Judge through the Codex CLI (`codex exec`, read-only sandbox); returns the agent's last message."""
    with tempfile.TemporaryDirectory() as tmp:
        last = Path(tmp) / "last.txt"
        # --ignore-user-config: skip user config and MCP servers (authentication still works); keeps each call light
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--sandbox", "read-only",
               "-o", str(last)]
        if args.model:
            cmd += ["-m", args.model]
        _run_cli(cmd + ["-"], system + "\n\n" + user, args, tmp)
        return last.read_text(encoding="utf-8"), {}


def call_cursor(system, user, args, n_cands):
    # Cursor CLI print mode, "ask" mode (no edits). The prompt is written to a file the agent is told to read,
    # because Windows limits command-line length.
    """Judge through the Cursor CLI in ask mode; the prompt is passed as a file to avoid command-line length limits."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "task.md").write_text(system + "\n\n" + user, encoding="utf-8")
        # --trust: skip the workspace-trust prompt; safe here because the folder is an empty temp dir with only task.md
        cmd = [args.cursor_cmd, "-p", "--trust", "--workspace", tmp, "--mode", "ask", "--output-format", "text"]
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["Read the file task.md in the current folder and do exactly what it says. "
                "Reply with the JSON only, no other text."]
        return _run_cli(cmd, None, args, tmp), {}


def call_dry(system, user, args, n_cands):
    """Print the prompt of the first call and exit (to inspect what the judge would see)."""
    print("=" * 80, "\nSYSTEM:\n", system, "\n\nUSER:\n", user[:6000], "\n...", f"\n[{len(user)} chars]")
    raise SystemExit(0)


BACKENDS = {"ollama": call_ollama, "openai": call_openai, "claude": call_claude, "codex": call_codex,
            "cursor": call_cursor, "dry": call_dry}


# ------------------------------------------------------------------ limits
LIMIT_PAT = re.compile(r"usage limit|spend(ing)? cap|rate[ _-]?limit|limit reached|reached your|hit your|quota|too many requests|"
                       r"\b429\b|out of (credits|usage)|try again (at|in|later)|plan limit", re.I)
TRANSIENT_PAT = re.compile(r"channel closed|error decoding response body|transport|timed out|timeout|connection|"
                           r"ECONNRESET|503|502|500|overloaded", re.I)


def is_limit(err: str) -> bool:
    """True only for real usage/rate limits; transient network/transport errors are retried instead."""
    return bool(LIMIT_PAT.search(err or "")) and not (TRANSIENT_PAT.search(err or "") and not re.search(r"\b429\b|quota|usage limit|limit reached|spend(ing)? cap|hit your", err or "", re.I))


# ------------------------------------------------------------------ parsing
def extract_json(text: str) -> dict:
    """Parse the judge's reply as JSON, tolerating code fences and text around the JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ------------------------------------------------------------------ main
def main():
    """Judge every (item, order) not yet in outputs/judgments/{run}.jsonl and append the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--model")
    ap.add_argument("--orders", type=int, default=2, choices=[1, 2], help="1 = forward only, 2 = forward + reversed")
    ap.add_argument("--reverse-subset", type=float, default=1.0,
                    help="share of items that also get the reversed order (e.g. 0.33 to save calls)")
    ap.add_argument("--items-per-call", type=int, default=1, help="pack several items into one call (fewer calls through the CLIs)")
    ap.add_argument("--limit", type=int, help="judge only the first N items (pilot)")
    ap.add_argument("--sleep", type=float, default=0, help="seconds between calls")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--think", type=lambda s: {"true": True, "false": False}.get(s.lower(), s), default=None,
                    help="Ollama thinking: true/false, or low/medium/high for models that support it")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env")
    ap.add_argument("--cursor-cmd", default="agent")
    args = ap.parse_args()

    # ---- load items and the judgments already done (resume)
    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.run}.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["item_id"], r["order"]))

    # ---- one task per (item, order); group tasks into calls
    rng = random.Random(0)
    fwd_tasks, rev_tasks = [], []   # (item, order_name, order)
    for it in items:
        k = len(it["candidates"])
        fwd = list(range(k))
        rng.shuffle(fwd)                      # random but fixed starting order per item
        fwd_tasks.append((it, "fwd", fwd))
        if args.orders == 2 and rng.random() < args.reverse_subset:
            rev_tasks.append((it, "rev", fwd[::-1]))
    # all forward passes first, then all reversed passes: the two orders of one item never share a call
    tasks = fwd_tasks + rev_tasks
    tasks = [t for t in tasks if (t[0]["item_id"], t[1]) not in done]
    # group into calls with the same number of candidates
    calls, cur = [], []
    for t in tasks:
        if cur and (len(cur) == args.items_per_call or len(cur[0][2]) != len(t[2])):
            calls.append(cur)
            cur = []
        cur.append(t)
    if cur:
        calls.append(cur)
    print(f"{len(items)} items -> {len(tasks)} judgments to do in {len(calls)} calls  [{args.backend} {args.model or ''}]")

    # ---- run the calls: retry transient errors, stop on usage limits, write one record per (item, order)
    fn = BACKENDS[args.backend]
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for ci, call in enumerate(calls):
            its, names, orders = [c[0] for c in call], [c[1] for c in call], [c[2] for c in call]
            # unique id per (item, order) inside a call, so replies can never be mapped to the wrong order
            keyed = [{**it, "item_id": f"{it['item_id']}#{nm}"} for it, nm in zip(its, names)]
            system, user, maps = build_prompt(keyed, orders)
            raw, meta, err, parsed = None, {}, None, None
            for attempt in range(args.retries + 1):
                try:
                    t0 = time.time()
                    raw, meta = fn(system, user, args, len(orders[0]))
                    meta["seconds"] = round(time.time() - t0, 1)
                    parsed = extract_json(raw)
                    err = None
                    break
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    print(f"  call {ci + 1}: attempt {attempt + 1} failed - {err[:200]}")
                    if is_limit(err):
                        print("  Looks like a usage/rate limit. Stop here and re-run later to resume.")
                        return
                    time.sleep(15 * (attempt + 1))
            entries = {str(e.get("item_id")): e for e in (parsed or {}).get("items", [])} if parsed else {}
            if parsed and len(its) == 1 and not entries and "ranking" in parsed:
                entries = {keyed[0]["item_id"]: parsed}          # model skipped the wrapper
            for it, kit, name, order, lmap in zip(its, keyed, names, orders, maps):
                e = entries.get(str(kit["item_id"]))
                rec = {"item_id": it["item_id"], "order": name, "judge": f"{args.backend}:{args.model or 'default'}",
                       "letter_to_cand": lmap, **meta}
                problems = validate(e, list(lmap)) if e else ["missing in reply"]
                if err or problems:
                    rec["error"] = err or "; ".join(problems)
                    rec["raw"] = (raw or "")[:3000]
                else:
                    rec.update(remap(e, lmap))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_err = sum(1 for kit in keyed if not entries.get(str(kit["item_id"])))
            print(f"  call {ci + 1}/{len(calls)} done ({meta.get('seconds', '?')}s){' - ' + str(n_err) + ' missing' if n_err else ''}"
                  f"  [{(time.time() - t_start) / 60:.1f} min]")
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
