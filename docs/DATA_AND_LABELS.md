# Data, Labels, Rubric and Gold Score

This document describes the data splits, how candidate summaries are generated and labelled, the rubric used by the AI
judges and the human rater (verbatim), the automatic gold score, and which signal is used where in the RLHF pipeline.
Results are in [`../README.md`](../README.md) and [`RESULTS.md`](RESULTS.md).

**Task.** Given one Thai listed company's SET data (English, amounts in THB million), write an investor summary of at most
500 tokens. Every prompt is the same instruction (`guided_v2`, `src/prompts.py`) plus one company's data card for one
reporting period.

---

## 1. Splits

Each company gives two prompts (two reporting periods<sup>(1)</sup>). Splits are made by company, stratified by market and industry, so
no company appears in both training and evaluation; both prompts of a company stay in the same split.

**Table 1 – sizes and labelling work**

| Split | Companies | Prompts | % of its set | % of all companies | Candidates per prompt (generator) | Candidates judged | Labelled by | Labelling work<sup>(2)</sup> |
|---|---|---|---|---|---|---|---|---|
| Pilot | 12 | 24 | – | 3.8% | 4 (base model, prompt / format variants) | 96 | Sonnet, GPT, Grok | 48 judgments per judge |
| A_train | 54 | 108 | 90.0% of A | 17.0% | 4 = 1 greedy + 3 sampled (base model) | 432 | Sonnet + GPT | 216 judgments per judge |
| A_val | 6 | 12 | 10.0% of A | 1.9% | 4 (base model) | 48 | Sonnet + GPT | 24 judgments per judge |
| B_train | 66 | 132 | 74.2% of B | 20.8% | 4 = 1 greedy + 3 sampled (SFT model) | 528 | Sonnet + GPT | 264 judgments per judge |
| B_val | 8 | 16 | 9.0% of B | 2.5% | 4 (SFT model) | 64 | Sonnet + GPT | 32 judgments per judge |
| B_test | 15 | 30 | 16.9% of B | 4.7% | 4 (SFT model) | 120 | Sonnet + GPT; Grok; human | 60 per judge (Sonnet, GPT); Grok 30; human 6 prompts |
| C_ppo | 30 | 60 | 19.2% of C | 9.5% | sampled during PPO | – | reward model | – |
| Reserve | 106 | 212 | 67.9% of C | 33.4% | sampled during PPO | – | reward model | – |
| C_eval | 20 | 40 | 12.8% of C | 6.3% | 1 greedy per model (base, SFT, PPO checkpoints) | 120 by Grok; 24 by human | Grok; human; gold | Grok 80 judgments; human 12 pairs |
| **Total** | **317** | **634** | | **100%** | | | | **1,192 training-label judgments** (4,768 candidate scores) |

**Table 2 – what each split is used for**

| Split | Used for |
|---|---|
| Pilot | Choosing the card format, the prompt and the judges. Never used for training. |
| A_train | **103 SFT targets.** Per prompt, the judges' best candidate: used as is if it passes the quality check (14), otherwise minimally corrected by an editor model (Sonnet or GPT) to fix the errors the judges listed, then re-checked automatically (89). 5 prompts have no target: 3 corrections failed the automatic re-check, 2 had no correction returned. |
| A_val | **12 SFT validation targets**, chosen the same way (1 as is, 11 corrected); validation loss and early stopping. |
| B_train | **609 reward-model training pairs.** 4 candidates give 6 pairs per prompt (132 x 6 = 792 possible); a pair is kept if the two candidates' mean overall scores (2 judges x 2 orders) differ by at least 0.5. Higher score = chosen; the gap = margin. rm_v2 adds 483 synthetic pairs<sup>(3)</sup>. |
| B_val | **76 validation pairs** (of 96), same rule; 8 companies held out from B_train. rm_v2 adds 56 synthetic pairs<sup>(3)</sup>. |
| B_test | **147 test pairs** (of 180), same rule; never trained on and never used for synthetic pairs. Also the **210 probes** (30 prompts x 7) and the human-vs-judge-vs-reward-model comparison. |
| C_ppo + reserve | **272 PPO prompts** (no labels). The main run uses 120 updates x 8 rollouts = 960 rollouts, about 3.5 per prompt. |
| C_eval | **Final evaluation** of base vs SFT vs PPO (with B_test, 70 held-out prompts): gold score for every checkpoint, Grok on base / SFT / final PPO, human on 12 SFT-vs-PPO pairs. |

<sup>(1)</sup> **Reporting periods.** T1 = latest full fiscal year (ending in 2025) vs the prior fiscal year, market data at end-2025;
T2 = latest reported period (e.g. 6M 2026) vs the same period a year earlier, market data at 22 Oct 2026.

<sup>(2)</sup> **Labelling work.** One *judgment* = one judge reads one prompt's data card and scores all of its candidates on the
rubric, flags severe errors and ranks them. Each prompt is judged twice per judge (shuffled order and reversed order), so
candidate scores per judge = prompts x 2 orders x 4 candidates (A_train: 108 x 2 x 4 = 864 per judge). Grok judged B_test
in one order only. The human rated each prompt or pair once.

<sup>(3)</sup> **Synthetic "correct beats broken" pairs (rm_v2).** Every labelled candidate is a complete SFT summary, so the first
reward model (rm_v1) never saw degenerate text and scored number-free templates and cut-off summaries above correct ones.
For rm_v2, each B_train / B_val prompt's best judged candidate (highest overall, no severe flag) is the chosen answer and
four automatically broken copies of it are rejected: **template** (generic text, no company figures), **truncated** (cut at
40–70%), **empty** and **wrong but formatted** (revenue or profit x1,000, or a rise/fall flipped), with margin 2.0
(483 train pairs from 121 prompts, 56 validation pairs from 14 prompts). No labels are involved. Two hacks – **padding**
and **keyword stuffing** – are deliberately left out, so the probes can still test hacks the reward model never saw.

**Collection and exclusions.** 342 companies were collected from the SET website's public quote pages (13 records each, see the README's Data Source); 25 were
excluded (17 property funds / REITs / infrastructure funds, 3 suspended, 4 without a prior period, 1 with Thai-only text),
leaving 317. The 106 reserve companies were collected after the first sample as spares and used only as PPO prompts (and
for a base-model card-format check).

---

## 2. How candidates are generated

- **Greedy (1 per prompt):** deterministic decoding – always the most likely next token, the model's default answer.
- **Sampled (3 per prompt):** temperature 0.8, top-p 0.9 (top-k 20 from Qwen's default generation config), different seeds:
  plausible alternatives that differ in wording, coverage and errors.
- Repetition penalty 1.1 (Qwen's default) for both; at most 600 new tokens (above the 500 limit on purpose, to measure overruns).
- Set A uses the **base** model, set B the **SFT** model, so the reward model learns to rank the kind of output PPO starts from.
- One greedy and three sampled outputs give each prompt a spread of quality, which makes the comparisons informative
  (set B: greedy gold 0.88 vs sampled 0.79).

---

## 3. Rubric (AI judges and human)

The same rubric is given to Claude Sonnet, GPT and Grok (`src/judge_prompt.py`) and shown to the human rater in the
"Scoring guide" panel of the blind form (`src/make_label_form.py`). Each judge call shows the company's data card (plain-text
format) and the candidates under neutral letters; the judge replies in JSON with, per candidate, the six dimension scores, a
severe-error flag and a list of errors, plus usefulness per persona, an overall score and a ranking.

System prompt (verbatim):
```
You are an expert, strict evaluator of investor summaries. You check every number and claim against the source data. You reply with JSON only.
```

Rubric (verbatim):
```
Score every summary on each dimension from 1 (worst) to 5 (best):

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
  value_investor: a VALUE INVESTOR: valuation (P/E, P/BV, dividend yield), balance-sheet strength, cash flow, durable earnings
  fund_manager: a FUND MANAGER: growth and what the data shows about it, risk, ratios (ROE, margins, leverage), quick to act on
  retail_investor: a RETAIL INVESTOR: plain language, key takeaway first, no unexplained jargon

overall (1-5): your holistic quality judgement, using the whole scale: 5 = accurate and useful with at most trivial issues,
4 = good with minor issues, 3 = usable but with clear weaknesses, 2 = misleading in places, 1 = unreliable.
It MUST be 1 or 2 when severe_error is true.
Buy/sell advice, greetings or chat filler, and running past 500 tokens lower the overall score. Length alone is not a merit.
ranking: all summaries best to worst by overall (break ties by fidelity, then average usefulness). No ties.
```

**How scores are combined.** Every score is averaged over all judgments of a candidate (2 judges x 2 orders for training
labels). The averaged overall score (1–5, continuous after averaging) selects the SFT targets and defines the reward-model
pairs and margins.

**How the overall score relates to the six dimensions.** It is not computed from them; each judge assigns it as a holistic
judgement under the rules above. In practice (set B, 2,848 candidate scores):
- it follows the **worst** dimension more than the average: correlation 0.84 with the minimum of the six vs 0.75 with their mean;
- a linear fit (R² = 0.67) puts most weight on invented claims (0.35), direction (0.27) and fidelity (0.19);
  units 0.09, attribution 0.04, coverage 0.02;
- the severe rule is always followed: all 748 severe-flagged scores have overall 1 or 2;
- judges are strict overall: mean overall 2.23 vs mean dimension score 3.52.

The first version of the rubric defined severe errors broadly and one judge flagged 99% of summaries; the narrow definition
above brought severe rates to about 25–30%, with 91–93% agreement between judges.

---

## 4. Gold score (automatic, never seen by the reward model or PPO)

`src/auto_rubric.py` and `src/gold_check.py`. Every check is rule-based – regular expressions and matching numbers against
the company's fact list (`data/facts/`); no language model is used.

**Components and weights** (each component 0–1):

`composite = 0.30 fidelity + 0.20 direction & attribution + 0.15 units + 0.15 coverage + 0.10 invented + 0.10 discipline`

| Component | Weight | Formula | How it is detected |
|---|---|---|---|
| Fidelity | 0.30 | share of the summary's numbers found in the data | Every number is extracted with its currency, scale word and % or x, and matched against the fact list, allowing million / billion / trillion, rounding to the digits shown, and sign. Years, dates, period labels and list numbers are skipped. |
| Direction & attribution | 0.20 | 1 − 0.34 x direction errors − 0.20 x attribution flags | **Direction:** the summary is split into clauses; metric names (revenue, net profit, …) and rise/fall words (increased, fell, …) are found from word lists and paired by grammar rules, then compared with whether current > prior in the data. **Attribution:** a number that matches only a sector or SET-index figure with no "sector / industry / market" wording within about 90 characters = sector figure presented as the company's; a number that matches only a prior-period value with no "prior / previous / from / vs" wording nearby = prior figure presented as current. |
| Units | 0.15 | 1 − 0.34 x scale errors − 0.10 x missing scale words | A number that matches nothing as written but matches a THB-million fact after x/÷ 1,000 or 1,000,000 = scale error; a correct THB-million amount of 100 or more with no "million / billion" after it = missing scale word. |
| Coverage | 0.15 | 0.5 x key facts + 0.5 x sections | Key facts: whether 6 headline figures appear as numbers (net profit, its growth, top line, ROE, P/E, dividend yield). Sections: business, performance, financial position, valuation, takeaway, found by keywords. |
| Invented | 0.10 | 1 − 0.10 x causal / comparative / forecast phrases − 0.03 x evaluative phrases | Keyword lists: causal ("driven by", "due to"), comparative ("peers", "industry average"), forecast ("expected to", "outlook"), evaluative ("healthy", "attractive", "strong"). |
| Discipline | 0.10 | 1 − 0.25 x each failure | Over 500 tokens; truncated (hit the token cap or does not end cleanly); chat filler ("Sure", "let me know"); buy/sell advice by phrase matching ("strong buy", "we recommend", "you should / investors should buy, sell or consider", "buy rating", "is a buy", "target price"). |

**Why these weights.** The weights express how much each kind of failure matters to an investor and how reliably the code
can detect it; they sum to 1 and were set by judgement before any training, not fitted to data.
- **Fidelity 0.30 (largest):** a wrong number is the most damaging error in a financial summary, and it is the most reliable
  automatic check (a number either matches the data or it does not).
- **Direction & attribution 0.20:** a figure that is right but described as rising when it fell, or tied to the wrong period
  or entity, leads the reader to the wrong conclusion; the check is rule-based on wording, so less certain than fidelity.
- **Units 0.15** and **coverage 0.15:** a missing or wrong scale makes a figure unusable but is rarer; coverage measures
  usefulness rather than correctness.
- **Invented 0.10** and **discipline 0.10 (smallest):** keyword matching is the weakest signal (it misses paraphrases and
  sometimes flags harmless words), and length or filler matter less than factual errors – the hard limits on length and
  truncation are enforced by the caps below instead.

**Why these per-error penalties.** Each component starts at 1 and loses a fixed amount per error found, floored at 0:

| Penalty | Value | Reasoning |
|---|---|---|
| Wrong rise/fall; 1,000x scale error | 0.34 each | One error costs about a third of the component and three errors use it up ("three strikes"); these are clear, high-confidence errors. |
| Attribution flag | 0.20 each | Lighter than a direction error because the check relies on nearby wording within a fixed window and can misfire; five flags use up the component. |
| Amount without a scale word | 0.10 each | A formatting slip: the figure is right but the reader must guess "million"; ten slips use up the component. |
| Causal, comparative or forecast phrase | 0.10 each | Clear signs of content the data does not support. |
| Evaluative word ("strong", "healthy", …) | 0.03 each | These words are common and often harmless ("strong growth" when growth was 40%), so each one costs little; only many of them add up. |
| Discipline failure (over 500 tokens, truncated, filler, advice) | 0.25 each | Four possible failures, each worth a quarter of the component. |
| Coverage: key facts vs sections | 0.5 / 0.5 | Equal weight for reporting the headline numbers and for covering every required part of the summary. |

**Caps and gates** (applied in this order):

| Condition | Effect | Why this threshold |
|---|---|---|
| Automatic severe error: wrong rise/fall on the top line or net profit, any 1,000x scale error, or a sector figure presented as the company's | gold = min(composite, 0.4) | Mirrors the judges' rule that a severe error forces the overall score to 1 or 2 out of 5 (2/5 = 0.4): such a summary must rank below any sound one, however good the rest of it is. |
| Non-answer: no numbers, or fewer than 30 words | gold = 0 | There is nothing to evaluate. 30 words is far below any real summary (the shortest of 1,904 base and SFT outputs had 80 words). |
| None of the headline figures | gold ≤ 0.4 | Missing revenue, profit and every other headline figure makes the summary useless for its purpose – treated like a severe error. |
| Truncated (hit the token cap or does not end cleanly) | gold ≤ 0.6 | A cut-off summary can still be correct and informative, so it ranks above a severe error (0.4) but below any complete, sound summary (median gold of non-severe outputs 0.89); 0.6 corresponds to the judges' "usable with clear weaknesses" (3/5). |
| Keyword stuffing: ≥ 2 self-referential quality claims ("accurate", "data-supported", "verified", "no forecasts", …) or 4-gram repetition above 0.15 | gold ≤ 0.5 | Stuffing is deliberate gaming, so it is capped near the severe level. One such word can appear naturally, two almost never: the rule flagged none of 1,904 real outputs. Real outputs repeat at most 9% of their 4-word phrases (99th percentile 3%), so 15% only catches loops and padding. |
| Over 500 tokens | gold x max(0.5, 1 − (tokens − 500) / 400) | 500 tokens is the task limit. The penalty grows linearly with the overrun, so that the 600-token generation cap costs 25% (x 0.75) and a 700-token answer reaches the floor; the floor of 0.5 keeps length from outweighing the content entirely. |

The weights, penalties and thresholds were set by judgement, checked against the probes and the 1,904 real outputs above, and
not tuned to make any result look better; how sensitive the results are to them was not tested. The weights were fixed
before training. The caps were added after the probe test showed that the weighted sum alone rewarded degenerate answers,
and before any PPO run; nothing was changed after PPO. Effect on the probes (weighted sum -> gold with
caps): correct 0.96 -> 0.92, padded (long) 0.95 -> 0.44, keyword-stuffed 0.95 -> 0.47, truncated 0.90 -> 0.60,
template 0.63 -> 0, empty 0.53 -> 0, wrong but formatted 0.42 -> 0.41.

**Limits of the automatic checks.** Keyword matching misses paraphrased advice or verdicts (e.g. "may appeal to income
investors") and cannot judge meaning. The automatic severe check does not catch a wrong revenue or profit *figure* on its
own (that lowers fidelity) or wording such as "a net loss of THB 8.8 million" when the company made a profit, because the
number is correct and no rise/fall word is used. The judges' severe flag is a separate, model-based judgement, which is why
both are reported.

**Ranges.** The overall score runs from 1 to 5 (integers per judgment, averaged over judgments); gold runs from 0 to 1. They
agree only moderately (Spearman 0.26–0.43): the judges read meaning, gold checks mechanics.

---

## 5. Probes (B_test)

Pairwise test accuracy only checks the reward model on natural candidates, which differ in subtle ways, while PPO actively
searches for outputs the reward model over-rewards. The probe suite (`src/make_probes.py`) builds seven responses per B_test
prompt from its best SFT candidate:

| Probe | Construction |
|---|---|
| correct | the candidate unchanged |
| long | restated card figures inserted before the takeaway until about 700 tokens |
| wrong but formatted | revenue or net profit x1,000, or a rise/fall word about them flipped |
| keyword-stuffed | a paragraph of rubric-sounding words appended ("accurate, data-supported, verified, THB million, …") |
| template | a generic five-section summary with no company figures |
| truncated | cut mid-sentence at about 55% of the text |
| empty | an empty reply |

If the reward model scores a probe as high as the correct answer, PPO is likely to drift that way, so the probe table is a
prediction of reward hacking made before PPO and checked against what PPO actually does. The probes also tested the gold
score itself, which led to its caps.

---

## 6. Which signal is used where

| Stage | Training loss | Judge overall | Judge six dimensions | Judge severe flag | Judge error list | Gold score | Synthetic data |
|---|---|---|---|---|---|---|---|
| **SFT** | next-token cross-entropy on the 103 target summaries (assistant tokens only) | selects each prompt's target (ties: fidelity, then usefulness, then gold) | quality check: fidelity, units, direction, attribution all ≥ 4 -> used as is, otherwise corrected | majority severe -> cannot be used as is | given to the editor model, which fixes exactly these errors | quality check (not severe, number precision ≥ 0.95, not cut off) and re-check of every correction (also ≤ 330 words) | – |
| **Reward model** | Bradley–Terry with margin: −log σ(r_chosen − r_rejected − m) + small reward-centring term | defines the pairs; margin m = score gap; gaps < 0.5 dropped | – (only through overall) | – (forces overall ≤ 2, so severe candidates lose by large margins) | – | not used (diagnostics only) | rm_v2: correct vs 4 broken types, m = 2.0 |
| **PPO** | PPO-clip on reward = RM score − β x KL(policy ‖ SFT) − 1.0 if no end-of-text token | – | – | – | – | **never in the reward**; logged on every update and checkpoint | – |
| **Evaluation** | – | Grok and human preferences | Grok dimensions (failure types) | severe rate per model | error types | main metric | probes re-scored |

Not used in any training loss: usefulness (an SFT tie-breaker only), the ranking list, Grok's scores and the human ratings.

---

## 7. Token lengths (Qwen2.5 tokenizer)

| Text | Mean | Note |
|---|---|---|
| Prompt (system + instruction + markdown data card) | 1,621–1,627 | range 1,286–1,872 |
| Base model response (set A) | 378 | 14% over 500 tokens |
| SFT target (A_train / A_val) | 483 / 463 | 41 of 103 training targets over 500 tokens |
| SFT model response (set B) | 417 | 22% over 500 tokens |
| Reward-model input (prompt + one response) | about 2,050 | maximum length 2,560 (no pair exceeded it) |
| PPO rollout | about 1,630 + up to 500 | response cap 500 tokens, temperature 0.7 |
