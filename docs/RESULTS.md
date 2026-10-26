# Detailed Results

This document gives the detailed tables behind the summary in [`../README.md`](../README.md): card-format and prompt
selection, label quality, SFT, the reward models, every PPO checkpoint, and the held-out evaluation. Data, labels, the rubric
and the gold score are described in [`DATA_AND_LABELS.md`](DATA_AND_LABELS.md). All numbers come from files in this repository
(`outputs/base_runs/`, `outputs/judgments/`, `outputs/train_logs/`, `outputs/eval_runs/`, `labeling/results/`).

Unless stated otherwise, "gold" is the automatic gold score (0–1) and "severe" is the automatic severe-error rate.

---

## 1. Card format and prompt selection (base models, pilot companies)

Twelve pilot companies x 2 reporting periods = 24 prompts, greedy decoding.

**Card format** (`guided` prompt; `outputs/base_runs/exp1_formats_summary.csv`):

| Model | Card format | Gold | Severe | Numbers found in the data | Mean tokens | Over 500 tokens |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **markdown** | **0.801** | 17% | **0.965** | 403 | 17% |
| Qwen2.5-1.5B-Instruct | text | 0.658 | 17% | 0.951 | 499 | 50% |
| Qwen2.5-1.5B-Instruct | json | 0.586 | 54% | 0.825 | 474 | 42% |
| Qwen2.5-1.5B-Instruct | raw json | 0.484 | 75% | 0.502 | 463 | 25% |
| Gemma 3 1B | markdown | 0.589 | 42% | 0.881 | 502 | 54% |
| Gemma 3 1B | text | 0.437 | 62% | 0.884 | 556 | 75% |
| Gemma 3 1B | json | 0.462 | 46% | 0.866 | 559 | 88% |
| Gemma 3 1B | raw json | 0.315 | 67% | 0.401 | 462 | 71% |

Markdown tables with amounts in THB million gave the most accurate numbers for both models; Qwen was clearly more accurate
and better at keeping to the length limit than Gemma, so it was chosen as the policy.

**Prompt** (Qwen, markdown cards; `outputs/base_runs/exp2_summary.csv`):

| Prompt | Gold | Severe | Numbers found | Mean tokens | Over 500 | Truncated |
|---|---|---|---|---|---|---|
| minimal | 0.498 | 17% | 0.959 | 557 | 83% | 71% |
| guided | 0.802 | 17% | 0.967 | 414 | 8% | 0% |
| **guided_v2** | 0.766 | 25% | **0.973** | **387** | **0%** | 0% |

`guided_v2` adds explicit rules against rescaling numbers, inventing causes and chat filler (for Gemma it removed the chat
preamble in every output). It was chosen for its number precision and length discipline; on the gold score `guided` is
slightly higher on these 24 prompts (0.80 vs 0.77), a difference within noise.

**Amounts in THB million vs THB billion for large companies** (106 reserve companies x 2 periods, Qwen, `guided_v2`;
`outputs/base_runs/fmt_check_summary.csv`): gold 0.727 / 0.707 (T1 / T2) for THB million vs 0.693 / 0.734 for automatic
billions, severe 22–25% for both – no difference, so every card uses THB million.

---

## 2. Label quality

Agreement is computed with `src/analyze_judges.py`. *Pairwise preference agreement* = share of candidate pairs that two raters
order the same way (pairs with a tie for either rater are skipped); *order consistency* = the same judge's agreement with itself
when the candidate order is reversed.

| Comparison | Items | Pairwise preference agreement | Overall-score Spearman | Severe-flag agreement | Severe rate |
|---|---|---|---|---|---|
| Sonnet vs GPT, pilot | 24 prompts | 0.95 | 0.81 | 0.93 | 25% vs 30% |
| Sonnet vs Grok, pilot | 24 prompts | 0.95 | 0.88 | 0.97 | 25% vs 28% |
| GPT vs Grok, pilot | 24 prompts | 0.99 | 0.85 | 0.92 | 30% vs 28% |
| Sonnet vs GPT, set A | 120 prompts | 0.92 | – | 0.87 | – |
| Sonnet vs GPT, set B | 178 prompts | 0.97 | 0.87 | 0.91 | 25% vs 32% |
| Sonnet vs Grok, B_test | 30 prompts | 0.97 | 0.87 | 0.95 | 21% vs 24% |
| GPT vs Grok, B_test | 30 prompts | 0.95 | 0.85 | 0.93 | 32% vs 24% |
| Human vs Sonnet, B_test | 6 prompts | 0.86 | 0.55 | 0.83 | 25% vs 17% |
| Human vs GPT, B_test | 6 prompts | 0.95 | 0.62 | 0.88 | 25% vs 29% |
| Human vs Grok, B_test | 6 prompts | 0.89 | 0.54 | 0.83 | 25% vs 17% |

| Order consistency | Pilot | Set A | Set B |
|---|---|---|---|
| Sonnet | 0.85 | 0.85 | 0.86 |
| GPT | 0.85 | 0.85 | 0.89 |
| Grok | 0.91 | – | – |

- The human agrees with the AI judges on *which* summary is better nearly as often as the judges agree with each other, but
  uses the 1–5 scale differently (lower Spearman).
- Judge overall vs automatic gold: Spearman 0.43 (set B, both judges), 0.26–0.37 on B_test. The judges read meaning; gold
  checks mechanics.
- The labels carry no length preference: in the reward-model pairs the chosen summary is the longer one in only 44% (train)
  and 42% (test) of pairs (mean 219 vs 229 words).
- Human rating time: median 27 minutes per 4-candidate prompt with the full rubric, 9 minutes per SFT-vs-PPO pair.

---

## 3. SFT

**Settings.** Qwen2.5-1.5B-Instruct + LoRA (r 16, alpha 32, dropout 0.05, all attention and MLP projections), learning rate
1e-4 with cosine schedule and 5% warm-up, 2 sequences x 4 accumulation = 8 per step, loss on the summary tokens only, maximum
length 2,560 tokens, 6 epochs with early stopping on validation loss (patience 2). 78 steps, 12.6 minutes, 6.2 GB peak.

| Epoch | 0 (base) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Validation loss | 0.607 | 0.467 | 0.425 | 0.407 | 0.399 | 0.399 | **0.399** |
| Validation next-token accuracy | 83.4% | 86.2% | 87.3% | 87.8% | 88.3% | 88.2% | 88.2% |

Training loss ended at about 0.30. Early stopping did not trigger; the last epoch had the lowest validation loss.

**Generation quality of the SFT model** (gold with caps, greedy unless stated):

| Data | Gold | Severe | Numbers found | Over 500 tokens |
|---|---|---|---|---|
| Base model, set A (120 prompts) | 0.741 | 15% | 0.983 | 10% |
| SFT model, A_val (12 prompts) | 0.805 | 0% | 0.986 | 25% |
| SFT model, set B greedy (178 prompts) | 0.878 | 7% | 0.995 | 15% |
| SFT model, set B sampled (534 outputs) | 0.791 | 10% | 0.979 | 24% |

The SFT targets are longer than the base model's outputs (mean 483 vs 378 tokens; 41 of 103 above 500 tokens), and the SFT
model inherited this.

---

## 4. Reward models

**Settings.** Qwen2.5-0.5B-Instruct with a scalar head, full fine-tune (fp32 weights, bf16 autocast, gradient
checkpointing), learning rate 1e-5 cosine, 2 pairs x 4 accumulation = 8 pairs per step, 2 epochs, reward-centring
coefficient 0.01, maximum length 2,560 tokens. The reward is read at the final `<|im_end|>` token, as in PPO.
rm_v1: 154 steps, 15 minutes, 10.3 GB; rm_v2: 274 steps, 26 minutes.

**Learning curve and robustness** (B_test, 147 pairs; candidates = the 120 SFT outputs on B_test):

| Model | Training pairs (prompts) | Test accuracy | Margin < 1 / ≥ 1 | Validation accuracy | ECE | ρ(reward, length) | ρ(reward, gold) | ρ(reward, judge overall) |
|---|---|---|---|---|---|---|---|---|
| rm_lc25 | 147 (33) | 0.531 | 0.44 / 0.58 | – | 0.248 | 0.46 | −0.06 | – |
| rm_lc50 | 302 (66) | 0.643 | 0.50 / 0.71 | – | 0.195 | 0.45 | 0.16 | – |
| rm_lc75 | 451 (99) | 0.680 | 0.50 / 0.77 | – | 0.143 | 0.27 | 0.22 | – |
| rm_v1 | 609 (132) | **0.762** | 0.67 / 0.81 | 0.763 | 0.125 | 0.38 | 0.25 | 0.48 |
| rm_v2 | 609 + 483 synthetic | 0.728 | 0.58 / 0.80 | 0.826 | 0.109 | 0.30 | 0.26 | 0.43 |

Column notes: *margin < 1 / ≥ 1* = accuracy on pairs whose judge scores differ by less than / at least one point;
*ECE* (expected calibration error) = how far the model's confidence is from its actual accuracy, averaged over the five
confidence bins in the calibration table below and weighted by the number of pairs in each bin (0 = perfectly calibrated;
0.11 means the stated confidence is off by about 11 percentage points on average); *ρ* = Spearman rank correlation.

Mean reward of severe vs non-severe candidates: rm_v1 −1.07 vs −0.45; rm_v2 −0.27 vs 0.11.
Agreement with a judge the reward models never learned from (Grok, B_test pairs): rm_v1 0.75, rm_v2 0.72. Agreement with the
human's ranking (6 prompts, 36 pairs): rm_v1 0.72, rm_v2 0.64, averaged AI labels 0.67.

**Calibration** (B_test pairs; confidence = σ(|r_chosen − r_rejected|), accuracy = share where the higher-scored response is the judges' choice):

| Confidence bin | rm_v1: pairs / accuracy | rm_v2: pairs / accuracy |
|---|---|---|
| 0.5–0.6 | 23 / 0.65 | 23 / 0.61 |
| 0.6–0.7 | 17 / 0.82 | 14 / 0.57 |
| 0.7–0.8 | 22 / 0.77 | 18 / 0.72 |
| 0.8–0.9 | 27 / 0.70 | 31 / 0.68 |
| 0.9–1.0 | 58 / 0.81 | 61 / 0.84 |

Both models are over-confident above 0.8.

**Probe suite** (30 B_test prompts x 7 responses; mean reward, and % of prompts where the correct response scores higher):

| Probe | Gold | rm_v1 mean reward | rm_v1: correct higher | rm_v2 mean reward | rm_v2: correct higher |
|---|---|---|---|---|---|
| correct | 0.955 | −0.07 | – | 0.53 | – |
| long (padded to ~700 tokens)* | 0.431 | −0.64 | 80% | **0.72** | **50%** |
| keyword-stuffed* | 0.485 | −0.49 | 50% | 0.19 | 57% |
| wrong but formatted | 0.417 | −0.09 | 60% | −5.38 | 100% |
| template | 0 | **+2.38** | **10%** | −3.65 | 100% |
| truncated | 0.593 | **+1.57** | **23%** | −3.75 | 100% |
| empty | 0 | −1.96 | 87% | −3.87 | 100% |

\* not represented in rm_v2's synthetic training pairs. Gate before PPO: test accuracy ≥ 0.65 and correct higher than empty,
template and truncated in ≥ 80% of prompts – rm_v1 fails, rm_v2 passes.

---

## 5. PPO

**Settings.** Policy = Qwen2.5-1.5B-Instruct with the SFT adapter merged in, plus a new LoRA (r 16, alpha 32) trained by PPO;
the reference model is the same network with the PPO adapter disabled. Reward model frozen (bf16); value model initialised
from the reward model and trained. 272 prompts (C_ppo + reserve); 8 rollouts per update (micro-batch 1 x 8 accumulation),
2 PPO epochs per batch, temperature 0.7 with no other sampling filters, response cap 500 tokens, −1.0 penalty for a response
without an end-of-text token, KL estimator k1. About 67 s per update, 14.7 GB peak.

| Run | Reward model | KL coefficient | Learning rate | Updates | Time |
|---|---|---|---|---|---|
| ppo_v1 | rm_v2 | 0.05 | 1e-5 | 120 | 2.2 h |
| ppo_lowkl | rm_v2 | 0.005 | 3e-5 | 80 | 1.5 h |
| ppo_rmv1 | rm_v1 | 0.01 | 3e-5 | 80 | 1.5 h |

All three runs use a fixed KL coefficient (ppo_v1 was configured with an adaptive controller, but its horizon was too long
for 8 rollouts per update and the coefficient stayed at 0.05).

**Training statistics** (`outputs/figures/3_ppo_training_stats.png`, `2_rollout_dynamics.png`): KL to the SFT policy
settles at about 8 nats in ppo_v1 and climbs to 78 (ppo_lowkl) and 37 (ppo_rmv1); value loss falls from 5–9 to about 0.2
within 20 updates in every run; policy loss stays within ±0.01; entropy is stable in ppo_v1 (0.23–0.28) and falls to about
0.17 in the ablations. In ppo_lowkl the share of rollouts that end with an end-of-text token drops from 0.70 to 0 from update
41 on, while their reward-model score rises to about 4.1 and their gold falls from 0.81 to 0.56.

**Every checkpoint** (greedy, 70 held-out prompts = C_eval 40 + B_test 30, maximum 600 new tokens; KL = mean training KL
over the 10 updates before the checkpoint):

*KL* is the Kullback–Leibler divergence between the PPO policy and the SFT policy on the training rollouts – how far PPO has
moved the model away from where it started. It is measured in **nats**, the unit of information when natural logarithms are
used (1 nat ≈ 1.44 bits), and summed over the tokens of a response (0 = identical models). 8 nats per response means that
the summaries PPO writes are, on average, about e^8 ≈ 3,000 times more likely under the PPO policy than under the SFT policy –
a modest shift for a 400-token text; 78 nats (ppo_lowkl) is a very large one.

| Run / checkpoint | KL (nats) | rm_v2 score | rm_v1 score | Gold | Gold C_eval / B_test | Severe | Over 500 | Truncated | Tokens |
|---|---|---|---|---|---|---|---|---|---|
| Base | – | −0.05 | −0.94 | 0.732 | 0.726 / 0.740 | 27.1% | 8.6% | 2.9% | 375 |
| SFT | 0 | 1.11 | −0.02 | 0.842 | 0.838 / 0.847 | 8.6% | 20.0% | 2.9% | 419 |
| ppo_v1 step 20 | 0.7 | 1.07 | 0.25 | 0.846 | 0.830 / 0.866 | 8.6% | 25.7% | 5.7% | 440 |
| ppo_v1 step 40 | 2.8 | 1.28 | −0.30 | 0.870 | 0.869 / 0.871 | 4.3% | 18.6% | 5.7% | 429 |
| ppo_v1 step 60 | 7.2 | 1.93 | 0.51 | 0.879 | 0.881 / 0.876 | 1.4% | 25.7% | 5.7% | 433 |
| ppo_v1 step 80 | 8.2 | 2.53 | 1.01 | 0.889 | 0.887 / 0.892 | 5.7% | 20.0% | 2.9% | 418 |
| ppo_v1 step 100 | 7.1 | 2.95 | 1.60 | 0.889 | 0.850 / 0.941 | 5.7% | 18.6% | 2.9% | 429 |
| **ppo_v1 step 120** | 8.1 | **2.94** | 1.41 | **0.899** | 0.888 / 0.915 | **2.9%** | 25.7% | 4.3% | 445 |
| ppo_lowkl step 20 | 11.5 | 0.93 | −1.05 | 0.820 | 0.822 / 0.817 | 7.1% | 31.4% | 7.1% | 446 |
| ppo_lowkl step 40 | 45.9 | 1.31 | 0.46 | 0.555 | 0.597 / 0.498 | 7.1% | 84.3% | 71.4% | 564 |
| ppo_lowkl step 60 | 60.7 | **1.63** | 0.83 | 0.437 | 0.438 / 0.435 | 4.3% | 98.6% | 97.1% | 595 |
| ppo_lowkl step 80 | 78.3 | 1.47 | 0.73 | **0.433** | 0.413 / 0.460 | 11.4% | 98.6% | 97.1% | 596 |
| ppo_rmv1 step 20 | 7.8 | 0.77 | **1.14** | 0.556 | 0.571 / 0.536 | 14.3% | 90.0% | 57.1% | 577 |
| ppo_rmv1 step 40 | 21.2 | 0.34 | 0.26 | 0.728 | 0.743 / 0.707 | 15.7% | 62.9% | 18.6% | 523 |
| ppo_rmv1 step 60 | 30.6 | 0.34 | 0.46 | 0.661 | 0.639 / 0.691 | 22.9% | 62.9% | 21.4% | 523 |
| ppo_rmv1 step 80 | 37.4 | 0.62 | 1.15 | 0.720 | 0.692 / 0.757 | 20.0% | 55.7% | 17.1% | 510 |

Keyword stuffing appears only in ppo_lowkl (10–14% of outputs from step 40) and once in ppo_rmv1 step 80; no checkpoint
produces non-answers. With 70 prompts, differences of a few points are within noise (e.g. the step-100 C_eval dip).

---

## 6. Held-out evaluation of base, SFT and PPO

**Grok** (C_eval, 40 prompts x base / SFT / ppo_v1 step 120, both orders, 80 judgments):

| Mean (1–5) | Base | SFT | PPO |
|---|---|---|---|
| Overall | 1.75 | 2.66 | **3.12** |
| Fidelity | 2.95 | 3.84 | 4.09 |
| Units | 4.75 | 4.85 | 4.95 |
| Direction | 2.66 | 3.00 | 3.12 |
| Attribution | 3.45 | 4.12 | 4.33 |
| Invented claims | 1.60 | 3.14 | **4.20** |
| Coverage | 4.15 | 4.45 | 4.65 |
| Usefulness (3 personas) | 1.82 | 2.73 | 3.17 |
| Severe-error flag | 44% | 26% | 26% |
| Ranked first (of 80 judgments) | 6 | 25 | **49** |

Head-to-head on mean overall per prompt: PPO vs SFT 23 wins, 9 ties, 8 losses (74% of decided prompts); SFT vs base 30–3–7
(81%); PPO vs base 34–2–4 (89%).

**Blind human comparison** (12 C_eval pairs, SFT vs ppo_v1 step 120, same rubric):

| | SFT | PPO |
|---|---|---|
| Preferred | 4 | 8 |
| Mean overall | 2.92 | 3.33 |
| Fidelity / direction | 4.17 / 3.33 | 4.67 / 3.75 |
| Invented claims | 3.42 | 4.50 |
| Coverage | 4.25 | 4.33 |
| Severe errors flagged | 3 | 4 |

Eight of twelve is not significant on its own (one-sided binomial p ≈ 0.19); the human gave both summaries the same overall
score in 8 of 12 pairs, so most pairs are close. It agrees in direction with Grok's 74%.

**Failure types**

| Failure type | Measured by | Base | SFT | PPO |
|---|---|---|---|---|
| Invented causes, verdicts, comparisons | Grok "invented" (higher = fewer) | 1.60 | 3.14 | 4.20 |
| Severe factual error | automatic check (70 prompts) | 27.1% | 8.6% | 2.9% |
| Severe factual error | Grok (C_eval) / human (12 pairs) | 44% / – | 26% / 25% | 26% / 33% |
| Over 500 tokens | token count | 8.6% | 20.0% | 25.7% |
| Cut off at the cap | token count | 2.9% | 2.9% | 4.3% |

**Example: key takeaway for AMC, FY2025**
- SFT: "While facing operational difficulties, AMC continues to maintain modest profitability margins despite declining sales
  volumes. Investors should monitor future developments closely to assess potential improvements in financial health…"
- PPO: "AMC reported lower revenue and net profit in FY2025, but higher asset values and a slightly improved return on equity."

---

## 7. Reproducibility

Every table above can be regenerated from the repository: `src/gold_check.py` (generation summaries), `src/analyze_judges.py`
(agreement), `outputs/train_logs/*.json|csv` (training and checkpoint evaluations, written by the training scripts and
`src/eval_checkpoints.py`) and `src/make_figures.py` (figures). Commands are in the README.
