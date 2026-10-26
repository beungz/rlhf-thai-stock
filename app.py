"""Streamlit demo of the stock-summary agent.

    streamlit run app.py

1. Type a SET stock symbol and choose a reporting period (T1 = latest full fiscal year, T2 = latest period).
2. The app reads the company's data feeds - the saved copy in data/raw/ (317 companies) or data/live/, or, for any
   other company, extracts them from the public SET quote pages in Chrome (src/fetch_set_pages.py).
3. It builds the same markdown data card the models were trained on and asks one or more models to summarise it:
   base Qwen2.5-1.5B-Instruct, the SFT model, the PPO model, or the over-optimised ablation (ppo_lowkl).
4. Every summary is checked with the automatic rubric (gold score, numbers not found in the data, wrong rise/fall,
   length) and, if available, scored by the reward model.

Running a model needs its weights in models/ (not in the repository; see README). Without them the app shows the
outputs saved during evaluation (outputs/eval_runs/) for the 35 held-out companies, so the demo also works without a GPU.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from auto_rubric import auto_rubric  # noqa: E402
from build_cards import build_snapshot, facts_for, render_markdown, render_text  # noqa: E402
from prompts import PROMPTS  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS = ROOT / "models"
EVAL_RUNS = ROOT / "outputs" / "eval_runs"
LIVE = ROOT / "data" / "live"
# label -> (adapters merged on top of the base model, saved-output run/checkpoint)
AGENTS = {
    "Base (Qwen2.5-1.5B-Instruct)": ([], ("ppo_v1", "base")),
    "SFT (sft_v1)": (["sft_v1"], ("ppo_v1", "sft")),
    "PPO (ppo_v1, step 120)": (["sft_v1", "ppo_v1/step_120"], ("ppo_v1", "step_120")),
    "Over-optimised PPO (ppo_lowkl, step 80)": (["sft_v1", "ppo_lowkl/step_80"], ("ppo_lowkl", "step_80")),
}


# ---------------------------------------------------------------- data
def load_company(symbol: str, source: str, show_browser: bool = False) -> tuple[dict | None, str]:
    """Raw feeds of one company: the saved copy in data/raw/ or data/live/, or extracted from the SET quote pages
    (src/fetch_set_pages.py, saved to data/live/). Returns (data or None, a message for the page).
    """
    if source.startswith("Fetch"):
        from fetch_set_pages import FEEDS, fetch_symbols
        try:
            rec = fetch_symbols([symbol], LIVE, refresh=True, headless=not show_browser)[symbol]
        except Exception as e:                                            # no Chrome, page refused, ...
            hint = "" if show_browser else " Try again with 'Show the browser window' on."
            return None, f"Could not extract {symbol} from the SET pages: {e}{hint}"
        errs = rec["_meta"]["errors"]
        if "income_statement" in errs or "overview" in errs:
            return None, f"The SET pages did not provide the data needed for {symbol} (missing: {', '.join(errs)})."
        n = len(FEEDS) + 1 - len(errs)
        return rec, f"Extracted {symbol} from the SET pages ({n}/{len(FEEDS) + 1} records); saved to data/live/."
    for p in (ROOT / "data" / "raw" / f"{symbol}.json", LIVE / f"{symbol}.json"):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), f"Saved data: {p.relative_to(ROOT)}"
    return None, f"No saved data for {symbol}. Choose 'Fetch from SET pages' to extract it."


# ---------------------------------------------------------------- models
def weights_available(adapters: list[str]) -> bool:
    """True if every adapter folder needed by an agent exists under models/."""
    return all((MODELS / a / "adapter_config.json").exists() for a in adapters)


@st.cache_resource(max_entries=1, show_spinner="Loading model…")
def load_policy(adapters: tuple[str, ...]):
    """Load the base model and merge the agent's LoRA adapters (cached; one model in memory at a time)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    for a in adapters:                               # same order as training: SFT LoRA, then PPO LoRA
        model = PeftModel.from_pretrained(model, str(MODELS / a)).merge_and_unload()
    return tok, model.to(dev).eval()


@st.cache_resource(max_entries=1, show_spinner="Loading reward model…")
def load_rm(name: str):
    """Load a trained reward model from models/ (cached)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)   # the RM was trained with the base model's tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rm = AutoModelForSequenceClassification.from_pretrained(str(MODELS / name), dtype=torch.bfloat16 if dev == "cuda"
                                                            else torch.float32)
    return tok, rm.to(dev).eval()


def generate(adapters, messages, max_new_tokens, slot):
    """Greedy generation streamed into the page; returns (summary, token count)."""
    import threading

    import torch
    from transformers import TextIteratorStreamer
    tok, model = load_policy(tuple(adapters))
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
              streamer=streamer, pad_token_id=tok.eos_token_id)   # greedy, as in evaluation
    th = threading.Thread(target=lambda: model.generate(**kw))
    with torch.no_grad():
        th.start()
        out = ""
        for piece in streamer:
            out += piece
            slot.markdown(out + " ▌")
        th.join()
    n_tok = len(tok(out, add_special_tokens=False)["input_ids"])
    return out.strip(), n_tok


def rm_score(name, messages, answer):
    """Reward-model score of a summary, read at the final <|im_end|> token as in training."""
    import torch
    tok, rm = load_rm(name)
    ids = tok.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True,
                                  return_dict=True)["input_ids"]
    last = max(i for i, t in enumerate(ids) if t == tok.eos_token_id)   # read the reward at <|im_end|>, as in training
    with torch.no_grad():
        return rm(input_ids=torch.tensor([ids[:last + 1][-2560:]], device=rm.device)).logits[0, 0].float().item()


def saved_output(run, ckpt, symbol, snap):
    """Saved evaluation output of one agent for one company and period, or None."""
    p = EVAL_RUNS / run / f"{ckpt}.jsonl"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["symbol"] == symbol and r["snapshot"] == snap:
            return r
    return None


@st.cache_data
def saved_symbols() -> list[str]:
    """Companies that have saved evaluation outputs (usable without model weights)."""
    p = EVAL_RUNS / "ppo_v1" / "step_120.jsonl"
    if not p.exists():
        return []
    return sorted({json.loads(l)["symbol"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()})


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Thai stock summary agent", page_icon="📈", layout="wide")
st.title("Thai stock summary agent")
st.caption("Qwen2.5-1.5B-Instruct aligned with RLHF (SFT -> reward model -> PPO) to summarise a SET-listed company "
           "in at most 500 tokens, using only the figures in its data.")

# ---- sidebar: inputs
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Stock symbol", value="AMC", help="SET ticker, e.g. PTT, KBANK, AMC").strip().upper()
    snap_label = st.radio("Reporting period", ["T1 – latest full fiscal year", "T2 – latest period"], index=1)
    snap = snap_label[:2]
    source = st.radio("Data source", ["Saved copy (data/raw, data/live)", "Fetch from SET pages (Chrome)"],
                      help="Saved copies exist for the 317 companies in data/raw/. For any other company, the app "
                           "opens its SET factsheet and company-highlight pages in Chrome and reads the data they load "
                           "(src/fetch_set_pages.py). A copy is kept in data/live/.")
    show_browser = False
    if source.startswith("Fetch"):
        show_browser = st.checkbox("Show the browser window", value=False,
                                   help="Off: Chrome runs in the background without a window. Turn on if the "
                                        "background fetch fails.")
    agents = st.multiselect("Agents", list(AGENTS), default=["SFT (sft_v1)", "PPO (ppo_v1, step 120)",
                                                         "Over-optimised PPO (ppo_lowkl, step 80)"])
    max_new = st.slider("Max new tokens", 300, 700, 600, 50, help="Above 500 on purpose, so overruns are visible.")
    use_rm = st.checkbox("Score with the reward model (rm_v2)", value=(MODELS / "rm_v2").exists())
    run_btn = st.button("Summarise", type="primary", use_container_width=True)
    if saved_symbols():
        st.caption("Saved outputs (no GPU needed) exist for: " + ", ".join(saved_symbols()))

if not run_btn:
    st.info("Type a stock symbol, choose the period and the agents, then press **Summarise**.")
    st.stop()

# ---- data: raw feeds -> snapshot -> markdown card, fact list and prompt
with st.spinner(f"Loading data for {symbol}…"):
    data, msg = load_company(symbol, source, show_browser)
st.caption(msg)
if data is None:
    st.error(msg)
    st.stop()
s = build_snapshot(data, snap)
if s is None:
    st.error(f"{symbol} {snap}: no usable reporting period in the data (e.g. no prior period to compare with).")
    st.stop()
card = render_markdown(s)
facts = facts_for(s)
m = re.search(r"(20\d\d)", s["periods"].get("prior") or "")
prior_year = m.group(1) if m else None
messages = PROMPTS["guided_v2"](card, "markdown")

head = s["company"]
st.subheader(f"{head.get('name') or symbol} ({symbol})")
st.write(f"**Period:** {s['periods']['current']} vs {s['periods']['prior']}  ·  "
         f"**Layout:** {head.get('statement_layout', '')}  ·  **Facts in card:** {len(facts)}")
with st.expander("Data card given to the agent (markdown, THB million)"):
    st.markdown(card)
with st.expander("Same data as plain text (what the judges see)"):
    st.text(render_text(s))

if not agents:
    st.warning("Choose at least one agent.")
    st.stop()
# ---- one column per agent: generate (or load the saved output), then run the automatic checks
cols = st.columns(len(agents))
for col, name in zip(cols, agents):
    adapters, (run, ckpt) = AGENTS[name]
    with col:
        st.markdown(f"#### {name}")
        slot = st.empty()
        rm_val = None
        if weights_available(adapters):
            text, n_tok = generate(adapters, messages, max_new, slot)
            hit_max = n_tok >= max_new
            origin = "generated now (greedy)"
            if use_rm and (MODELS / "rm_v2").exists():
                try:
                    rm_val = rm_score("rm_v2", messages, text)
                except Exception as e:                       # keep the summary and checks if the RM cannot load
                    st.warning(f"Reward model rm_v2 could not be loaded or run: {e}")
        else:
            rec = saved_output(run, ckpt, symbol, snap)
            if rec is None:
                slot.warning(f"Weights for this agent are not in models/, and there is no saved output for "
                             f"{symbol} {snap}. Saved outputs exist for the held-out companies listed in the sidebar.")
                continue
            text, n_tok, hit_max = rec["output"], rec.get("output_tokens"), rec.get("hit_max_tokens")
            rm_val = rec.get("reward_rm_v2")
            origin = f"saved output from evaluation ({run}/{ckpt}, greedy)"
        slot.markdown(text)
        a = auto_rubric(text, facts, output_tokens=n_tok, hit_max_tokens=hit_max, prior_year=prior_year)
        st.caption(origin)
        c1, c2, c3 = st.columns(3)
        c1.metric("Gold score", f"{a['gold']:.2f}", help="Automatic quality score, 0–1 (never seen by PPO)")
        c2.metric("Tokens", n_tok if n_tok is not None else "–", delta="over 500" if (n_tok or 0) > 500 else None,
                  delta_color="inverse")
        c3.metric("RM score", f"{rm_val:.2f}" if rm_val is not None else "–", help="Reward model rm_v2 (the proxy)")
        flags = []
        if a["severe"]:
            flags.append("severe error")
        flags += a["gates"]
        if a["advice"]:
            flags.append("advice")
        st.write("**Flags:** " + (", ".join(flags) if flags else "none"))
        with st.expander("Automatic checks"):
            st.write(f"Numbers: {a['n_numbers']} · not found in the data: {a['n_unsupported']}"
                     + (f" ({', '.join(a['unsupported'][:8])})" if a["unsupported"] else ""))
            st.write("Wrong rise/fall: " + (", ".join(f"{e['item']}" for e in a["direction_errors"]) or "none"))
            st.write("1,000x scale errors: " + (", ".join(map(str, a["scale_errors"])) or "none"))
            st.write("Attribution flags: " + (", ".join(f"{x['type']}" for x in a["attribution_flags"]) or "none"))
            st.write(f"Invented-claim phrases: {sum(a['invented'].values())}  ·  words: {a['words']}")
            st.json({k: a[k] for k in a if k.startswith("score_")})

st.divider()
st.caption("Not investment advice. The agent can still make meaning-level errors; check every figure against the data card.")
