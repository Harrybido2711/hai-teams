# Judge pilot — four candidate judges compared

Source: `results/` rows. Judged model: `gpt-5.6-luna`. Judge settings: `judge_core.JUDGES` (prices read 2026-10-19). 3 replicates; a unit's label needs a strict majority of them (QA: the median when all answered). **Every headline is on the common units** — labelled by every judge covering at least 50% of them — and shows n.

## MultiChallenge — GPT's pass rate under each judge

Upstream headline = macro mean over the four axes (blank if an axis has no common item). Wilson 95% interval on the micro rate.

| judge | n common | macro | micro | micro 95% CI | per-rep macro (own items) | flip rate | labelled (own) |
|---|---|---|---|---|---|---|---|
| haiku-4.5 | 23 | 44.5 | 47.8 | 29%–67% | 46.9, 46.9, 46.9 | 0.0% | 27 |
| haiku-5.5 | 23 | 36.1 | 43.5 | 26%–63% | 43.8, 41.5, 47.7 | 7.4% | 27 |
| kimi-k2.6 | 23 | 44.5 | 47.8 | 29%–67% | 36.9, 47.7, 46.4 | 14.8% | 27 |
| sonnet-5.5 | 23 | 46.7 | 52.2 | 33%–71% | 38.4, 49.2, 49.2 | 0.0% | 23 |

Pairwise agreement on the majority verdict, on the items both labelled (McNemar exact p on the discordant items):

| judge A | judge B | n | agree | κ | only A passes | only B passes | p |
|---|---|---|---|---|---|---|---|
| haiku-4.5 | haiku-5.5 | 27 | 85.2% | 0.70 | 2 | 2 | 1.000 |
| haiku-4.5 | kimi-k2.6 | 27 | 92.6% | 0.85 | 1 | 1 | 1.000 |
| haiku-4.5 | sonnet-5.5 | 23 | 95.7% | 0.91 | 0 | 1 | 1.000 |
| haiku-5.5 | kimi-k2.6 | 27 | 92.6% | 0.85 | 1 | 1 | 1.000 |
| haiku-5.5 | sonnet-5.5 | 23 | 91.3% | 0.83 | 0 | 2 | 0.500 |
| kimi-k2.6 | sonnet-5.5 | 23 | 95.7% | 0.91 | 0 | 1 | 1.000 |

## Wonderbread QA — GPT's answers: mean score per criterion (1 = best, 3 = worst)

| judge | completeness (n) | soundness (n) | clarity (n) | compactness (n) | flip rate |
|---|---|---|---|---|---|
| haiku-4.5 | 2.08 (12) | 2.08 (12) | 1.08 (12) | 1.08 (12) | 0.0% |
| haiku-5.5 | 1.83 (12) | 1.42 (12) | 1.00 (12) | 1.33 (12) | 6.2% |
| kimi-k2.6 | 1.58 (12) | 1.50 (12) | 1.08 (12) | 1.00 (12) | 6.2% |
| sonnet-5.5 | 1.75 (12) | 2.00 (12) | 1.25 (12) | 1.33 (12) | 4.2% |

Pairwise agreement per criterion:

| judge A | judge B | criterion | n | exact | QWK |
|---|---|---|---|---|---|
| haiku-4.5 | haiku-5.5 | completeness | 12 | 75.0% | 0.59 |
| haiku-4.5 | haiku-5.5 | soundness | 12 | 16.7% | 0.10 |
| haiku-4.5 | haiku-5.5 | clarity | 12 | 91.7% | 0.00 |
| haiku-4.5 | haiku-5.5 | compactness | 12 | 75.0% | 0.31 |
| haiku-4.5 | kimi-k2.6 | completeness | 12 | 50.0% | 0.53 |
| haiku-4.5 | kimi-k2.6 | soundness | 12 | 25.0% | 0.44 |
| haiku-4.5 | kimi-k2.6 | clarity | 12 | 100.0% | 1.00 |
| haiku-4.5 | kimi-k2.6 | compactness | 12 | 91.7% | 0.00 |
| haiku-4.5 | sonnet-5.5 | completeness | 12 | 50.0% | 0.52 |
| haiku-4.5 | sonnet-5.5 | soundness | 12 | 58.3% | 0.44 |
| haiku-4.5 | sonnet-5.5 | clarity | 12 | 83.3% | 0.43 |
| haiku-4.5 | sonnet-5.5 | compactness | 12 | 75.0% | 0.31 |
| haiku-5.5 | kimi-k2.6 | completeness | 12 | 58.3% | 0.56 |
| haiku-5.5 | kimi-k2.6 | soundness | 12 | 58.3% | 0.58 |
| haiku-5.5 | kimi-k2.6 | clarity | 12 | 91.7% | 0.00 |
| haiku-5.5 | kimi-k2.6 | compactness | 12 | 66.7% | 0.00 |
| haiku-5.5 | sonnet-5.5 | completeness | 12 | 58.3% | 0.58 |
| haiku-5.5 | sonnet-5.5 | soundness | 12 | 41.7% | 0.46 |
| haiku-5.5 | sonnet-5.5 | clarity | 12 | 75.0% | 0.00 |
| haiku-5.5 | sonnet-5.5 | compactness | 12 | 66.7% | 0.25 |
| kimi-k2.6 | sonnet-5.5 | completeness | 12 | 83.3% | 0.87 |
| kimi-k2.6 | sonnet-5.5 | soundness | 12 | 50.0% | 0.67 |
| kimi-k2.6 | sonnet-5.5 | clarity | 12 | 83.3% | 0.43 |
| kimi-k2.6 | sonnet-5.5 | compactness | 12 | 66.7% | 0.00 |

## Wonderbread QA — anchor set: mean score per criterion (1 = best, 3 = worst)

| judge | completeness (n) | soundness (n) | clarity (n) | compactness (n) | flip rate |
|---|---|---|---|---|---|
| haiku-4.5 | 1.90 (30) | 2.03 (30) | 1.20 (30) | 1.37 (30) | 0.0% |
| haiku-5.5 | 1.83 (30) | 1.73 (30) | 1.10 (30) | 1.73 (30) | 2.5% |
| kimi-k2.6 | 1.63 (30) | 1.27 (30) | 1.10 (30) | 1.13 (30) | 2.5% |
| sonnet-5.5 | 1.53 (30) | 1.60 (30) | 1.23 (30) | 1.67 (30) | 5.8% |

**Against the human labels** (quadratic-weighted κ; upstream's GPT-4 judge as the reference row, on the same items):

| judge | criterion | n | exact | QWK | Spearman | human mean | judge mean |
|---|---|---|---|---|---|---|---|
| upstream gpt-4 (reference) | completeness | 30 | 86.7% | 0.81 | 0.86 | 1.60 | 1.73 |
| haiku-4.5 | completeness | 30 | 56.7% | 0.42 | 0.46 | 1.60 | 1.90 |
| haiku-5.5 | completeness | 30 | 63.3% | 0.42 | 0.49 | 1.60 | 1.83 |
| kimi-k2.6 | completeness | 30 | 70.0% | 0.63 | 0.64 | 1.60 | 1.63 |
| sonnet-5.5 | completeness | 30 | 60.0% | 0.55 | 0.57 | 1.60 | 1.53 |
| upstream gpt-4 (reference) | soundness | 30 | 93.3% | 0.91 | 0.88 | 1.33 | 1.40 |
| haiku-4.5 | soundness | 30 | 30.0% | 0.16 | 0.29 | 1.33 | 2.03 |
| haiku-5.5 | soundness | 30 | 46.7% | 0.20 | 0.18 | 1.33 | 1.73 |
| kimi-k2.6 | soundness | 30 | 73.3% | 0.32 | 0.28 | 1.33 | 1.27 |
| sonnet-5.5 | soundness | 30 | 56.7% | 0.38 | 0.37 | 1.33 | 1.60 |
| upstream gpt-4 (reference) | clarity | 30 | 96.7% | 0.78 | 0.80 | 1.10 | 1.07 |
| haiku-4.5 | clarity | 30 | 83.3% | 0.36 | 0.39 | 1.10 | 1.20 |
| haiku-5.5 | clarity | 30 | 86.7% | 0.26 | 0.26 | 1.10 | 1.10 |
| kimi-k2.6 | clarity | 30 | 86.7% | 0.26 | 0.26 | 1.10 | 1.10 |
| sonnet-5.5 | clarity | 30 | 73.3% | 0.07 | 0.08 | 1.10 | 1.23 |
| upstream gpt-4 (reference) | compactness | 30 | 96.7% | 0.92 | 0.89 | 1.17 | 1.20 |
| haiku-4.5 | compactness | 30 | 83.3% | 0.56 | 0.60 | 1.17 | 1.37 |
| haiku-5.5 | compactness | 30 | 36.7% | 0.12 | 0.24 | 1.17 | 1.73 |
| kimi-k2.6 | compactness | 30 | 90.0% | 0.74 | 0.56 | 1.17 | 1.13 |
| sonnet-5.5 | compactness | 30 | 50.0% | 0.33 | 0.40 | 1.17 | 1.67 |

Pairwise agreement per criterion:

| judge A | judge B | criterion | n | exact | QWK |
|---|---|---|---|---|---|
| haiku-4.5 | haiku-5.5 | completeness | 30 | 80.0% | 0.68 |
| haiku-4.5 | haiku-5.5 | soundness | 30 | 63.3% | 0.49 |
| haiku-4.5 | haiku-5.5 | clarity | 30 | 90.0% | 0.62 |
| haiku-4.5 | haiku-5.5 | compactness | 30 | 50.0% | 0.28 |
| haiku-4.5 | kimi-k2.6 | completeness | 30 | 66.7% | 0.64 |
| haiku-4.5 | kimi-k2.6 | soundness | 30 | 26.7% | 0.27 |
| haiku-4.5 | kimi-k2.6 | clarity | 30 | 83.3% | 0.36 |
| haiku-4.5 | kimi-k2.6 | compactness | 30 | 80.0% | 0.50 |
| haiku-4.5 | sonnet-5.5 | completeness | 30 | 56.7% | 0.60 |
| haiku-4.5 | sonnet-5.5 | soundness | 30 | 60.0% | 0.42 |
| haiku-4.5 | sonnet-5.5 | clarity | 30 | 83.3% | 0.51 |
| haiku-4.5 | sonnet-5.5 | compactness | 30 | 56.7% | 0.42 |
| haiku-5.5 | kimi-k2.6 | completeness | 30 | 60.0% | 0.51 |
| haiku-5.5 | kimi-k2.6 | soundness | 30 | 56.7% | 0.43 |
| haiku-5.5 | kimi-k2.6 | clarity | 30 | 86.7% | 0.26 |
| haiku-5.5 | kimi-k2.6 | compactness | 30 | 33.3% | 0.10 |
| haiku-5.5 | sonnet-5.5 | completeness | 30 | 43.3% | 0.40 |
| haiku-5.5 | sonnet-5.5 | soundness | 30 | 70.0% | 0.53 |
| haiku-5.5 | sonnet-5.5 | clarity | 30 | 86.7% | 0.53 |
| haiku-5.5 | sonnet-5.5 | compactness | 30 | 73.3% | 0.45 |
| kimi-k2.6 | sonnet-5.5 | completeness | 30 | 76.7% | 0.79 |
| kimi-k2.6 | sonnet-5.5 | soundness | 30 | 66.7% | 0.62 |
| kimi-k2.6 | sonnet-5.5 | clarity | 30 | 86.7% | 0.53 |
| kimi-k2.6 | sonnet-5.5 | compactness | 30 | 46.7% | 0.29 |

## Wonderbread SOP Generation — GPT's SOPs, by ablation

| judge | group | pairs | precision | recall | F1 | ordering | flip rate | scored (own) |
|---|---|---|---|---|---|---|---|---|
| haiku-4.5 | td | 15 | 0.596 | 0.824 | 0.674 | 0.525 | 1.0% | 48 |
| haiku-4.5 | td_kf | 16 | 0.815 | 0.982 | 0.882 | 0.786 | 1.0% | 48 |
| haiku-4.5 | td_kf_act | 16 | 0.883 | 0.982 | 0.926 | 0.859 | 1.0% | 48 |
| haiku-5.5 | td | 15 | 0.557 | 0.737 | 0.590 | 0.537 | 7.1% | 47 |
| haiku-5.5 | td_kf | 16 | 0.850 | 0.963 | 0.897 | 0.802 | 7.1% | 47 |
| haiku-5.5 | td_kf_act | 16 | 0.903 | 0.957 | 0.926 | 0.876 | 7.1% | 47 |
| kimi-k2.6 | td | 15 | 0.572 | 0.805 | 0.643 | 0.521 | 8.1% | 48 |
| kimi-k2.6 | td_kf | 16 | 0.803 | 0.975 | 0.876 | 0.780 | 8.1% | 48 |
| kimi-k2.6 | td_kf_act | 16 | 0.875 | 0.951 | 0.905 | 0.826 | 8.1% | 48 |
| sonnet-5.5 | insufficient coverage | — | — | — | — | — | — | 0 |

Pairwise agreement on line-match decisions (lines both labelled):

| judge A | judge B | lines | agree | κ |
|---|---|---|---|---|
| haiku-4.5 | haiku-5.5 | 1053 | 91.1% | 0.72 |
| haiku-4.5 | kimi-k2.6 | 1054 | 92.2% | 0.76 |
| haiku-4.5 | sonnet-5.5 | 0 | — | — |
| haiku-5.5 | kimi-k2.6 | 1053 | 89.6% | 0.69 |
| haiku-5.5 | sonnet-5.5 | 0 | — | — |
| kimi-k2.6 | sonnet-5.5 | 0 | — | — |

## Wonderbread SOP Generation — anchor set

| judge | group | pairs | precision | recall | F1 | ordering | flip rate | scored (own) |
|---|---|---|---|---|---|---|---|---|
| haiku-4.5 | all | 30 | 0.708 | 0.816 | 0.726 | 0.590 | 1.4% | 30 |
| haiku-5.5 | all | 30 | 0.730 | 0.767 | 0.709 | 0.632 | 8.1% | 30 |
| kimi-k2.6 | all | 30 | 0.682 | 0.736 | 0.674 | 0.605 | 10.5% | 30 |
| sonnet-5.5 | insufficient coverage | — | — | — | — | — | — | 0 |

Pairwise agreement on line-match decisions (lines both labelled):

| judge A | judge B | lines | agree | κ |
|---|---|---|---|---|
| haiku-4.5 | haiku-5.5 | 589 | 89.3% | 0.74 |
| haiku-4.5 | kimi-k2.6 | 589 | 90.0% | 0.76 |
| haiku-4.5 | sonnet-5.5 | 0 | — | — |
| haiku-5.5 | kimi-k2.6 | 589 | 87.4% | 0.70 |
| haiku-5.5 | sonnet-5.5 | 0 | — | — |
| kimi-k2.6 | sonnet-5.5 | 0 | — | — |

**Against the human precision / recall** (common pairs):

| judge | metric | n | MAE | Pearson | Spearman | human mean | judge mean |
|---|---|---|---|---|---|---|---|
| upstream gpt-4 (reference) | precision | 30 | 0.112 | 0.84 | 0.85 | 0.515 | 0.533 |
| upstream gpt-4 (reference) | recall | 30 | 0.078 | 0.88 | 0.82 | 0.631 | 0.606 |
| haiku-4.5 | precision | 30 | 0.193 | 0.76 | 0.79 | 0.515 | 0.708 |
| haiku-4.5 | recall | 30 | 0.185 | 0.67 | 0.55 | 0.631 | 0.816 |
| haiku-5.5 | precision | 30 | 0.224 | 0.72 | 0.74 | 0.515 | 0.730 |
| haiku-5.5 | recall | 30 | 0.145 | 0.79 | 0.63 | 0.631 | 0.767 |
| kimi-k2.6 | precision | 30 | 0.175 | 0.82 | 0.84 | 0.515 | 0.682 |
| kimi-k2.6 | recall | 30 | 0.116 | 0.76 | 0.63 | 0.631 | 0.736 |

## Reliability and cost — every judge, every set

Cost is OpenRouter's billed `cost`, summed over every call ever made (re-asks and superseded rows included). A refusal is the model's content filter answering instead of the judge; it is never scored. `stale` = rows dropped because the text they judged has since changed.

| judge | set | rows | ok | refused | parse fail | failed | skipped | stale | calls | cost $ | $ / unit | p50 s | p95 s | backends |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| haiku-4.5 | mc/sample | 81 | 81 | 0 | 0 | 0 | 0 | 0 | 81 | 0.1681 | 0.00623 | 2.9 | 5.5 | Anthropic 81 |
| haiku-5.5 | mc/sample | 81 | 81 | 0 | 0 | 0 | 0 | 0 | 81 | 0.0209 | 0.00077 | 1.9 | 3.2 | Anthropic 81 |
| kimi-k2.6 | mc/sample | 81 | 81 | 0 | 0 | 0 | 0 | 0 | 81 | 0.1111 | 0.00412 | 3.8 | 10.4 | Crusoe 81 |
| sonnet-5.5 | mc/sample | 81 | 70 | 11 | 0 | 0 | 0 | 0 | 81 | 0.3996 | 0.01480 | 3.1 | 4.1 | Anthropic 81 |
| haiku-4.5 | qa/gpt-5.6-luna | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 144 | 0.1037 | 0.00864 | 0.6 | 0.9 | Anthropic 144 |
| haiku-5.5 | qa/gpt-5.6-luna | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 144 | 0.0138 | 0.00115 | 0.8 | 2.3 | Anthropic 144 |
| kimi-k2.6 | qa/gpt-5.6-luna | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 144 | 0.0368 | 0.00306 | 0.4 | 0.8 | Crusoe 144 |
| sonnet-5.5 | qa/gpt-5.6-luna | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 144 | 0.2758 | 0.02298 | 1.2 | 2.0 | Anthropic 144 |
| haiku-4.5 | qa/anchor-human-sample | 360 | 360 | 0 | 0 | 0 | 0 | 0 | 360 | 0.2602 | 0.00867 | 0.7 | 1.5 | Anthropic 360 |
| haiku-5.5 | qa/anchor-human-sample | 360 | 360 | 0 | 0 | 0 | 0 | 0 | 360 | 0.0345 | 0.00115 | 0.9 | 3.4 | Anthropic 360 |
| kimi-k2.6 | qa/anchor-human-sample | 360 | 360 | 0 | 0 | 0 | 0 | 0 | 360 | 0.0956 | 0.00319 | 0.4 | 1.0 | Crusoe 360 |
| sonnet-5.5 | qa/anchor-human-sample | 360 | 360 | 0 | 0 | 0 | 0 | 0 | 360 | 0.6904 | 0.02301 | 1.3 | 3.3 | Anthropic 360 |
| haiku-4.5 | sop/gpt-5.6-luna | 3162 | 3162 | 0 | 0 | 0 | 0 | 0 | 3166 | 4.6779 | 0.09746 | 2.1 | 3.2 | Anthropic 3162 |
| haiku-5.5 | sop/gpt-5.6-luna | 3162 | 3158 | 0 | 4 | 0 | 0 | 0 | 3191 | 0.5682 | 0.01184 | 1.9 | 3.0 | Anthropic 3162 |
| kimi-k2.6 | sop/gpt-5.6-luna | 3162 | 3162 | 0 | 0 | 0 | 0 | 0 | 3171 | 3.3901 | 0.07063 | 2.4 | 6.7 | Crusoe 3162 |
| sonnet-5.5 | sop/gpt-5.6-luna | 20 | 0 | 20 | 0 | 0 | 0 | 0 | 20 | 0.0000 | 0.00000 | 1.1 | 2.3 | Anthropic 20 |
| haiku-4.5 | sop/anchor-human-sample | 1767 | 1767 | 0 | 0 | 0 | 0 | 0 | 1769 | 2.4775 | 0.08258 | 2.1 | 3.1 | Anthropic 1767 |
| haiku-5.5 | sop/anchor-human-sample | 1767 | 1767 | 0 | 0 | 0 | 0 | 0 | 1767 | 0.2897 | 0.00966 | 1.9 | 3.0 | Anthropic 1767 |
| kimi-k2.6 | sop/anchor-human-sample | 1767 | 1767 | 0 | 0 | 0 | 0 | 0 | 1772 | 1.7796 | 0.05932 | 2.8 | 7.8 | Crusoe 1767 |
| sonnet-5.5 | sop/anchor-human-sample | 20 | 0 | 20 | 0 | 0 | 0 | 0 | 20 | 0.0000 | 0.00000 | 1.1 | 8.2 | Anthropic 20 |

| judge | backend | total cost $ | list price in / out per M |
|---|---|---|---|
| haiku-4.5 | Anthropic | 7.6873 | 1.0 / 5.0 |
| haiku-5.5 | Anthropic | 0.9270 | 0.1 / 0.5 |
| kimi-k2.6 | Crusoe | 5.4132 | 0.7 / 3.5 |
| sonnet-5.5 | Anthropic | 1.3658 | 2.0 / 10.0 |
