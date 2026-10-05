# LLM-as-a-judge for Wonderbread and MultiChallenge — which judge, and what pipeline

Two benchmarks in this suite need an LLM judge: **Wonderbread** (QA and SOP Generation) and
**MultiChallenge**. AwareBench's 60 judged rows were dropped on 2026-10-05
([AWARENESS_NOTES.md §5.0](../Transition_processes_benchmarks/Awareness_in_LLM/AWARENESS_NOTES.md)).
Every judge model the two benchmarks shipped with is retired, so a substitute has to be chosen
([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6, question 2). The user prefers a **light, cheap judge** if
the evidence allows one.

This file answers three things:
1. Which judge model to use (§1).
2. A general pipeline for any LLM judge (§2).
3. The concrete pipeline for each of the two benchmarks (§3).

The evidence is ten papers from 2025–2026 (§4). The existing judge documentation is reused, not
repeated:
- [JUDGE_RECORD.md](JUDGE_RECORD.md) — what each judge is shown and how it scores, with code
  citations.
- [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md) — the thirteen fields any judge must
  record.
- [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) — the verbatim prompts (Appendices A–C).
- [GPT_LLM_AS_JUDGE_GUIDE.md](GPT_LLM_AS_JUDGE_GUIDE.md) — API mechanics.

**How this was produced (2026-10-05).**
- **Selection.** Only papers *first released* in 2025 or later. Venue and citations were checked on
  arXiv and Semantic Scholar; Google Scholar indexes all of them through arXiv, but they were not
  looked up there one by one.
- **Reading.** Every PDF was read in full. Each number carries its PDF page, and the headline numbers
  were re-checked against the PDF text.
- **Inference.** Anything that is our reasoning rather than a paper's finding is marked
  *(our inference)*.
- **Earlier version.** It covered seven papers first released in 2024. It is in commit `0667bc1`.

---

## 1. Which judge model

### The short answer

| Role | Model (OpenRouter id) | Price in / out per M tokens | Why |
|---|---|---|---|
| **Default light judge** | Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`) | $1 / $5 | Outside every evaluated family. Measured in both a 21-judge pairwise study (Norman) and a rubric-based study (Pombal). Stable (test–retest 0.935). Anthropic judges had the lowest position bias of any provider. |
| **Cheaper challenger** | Kimi K2.5 (`moonshotai/kimi-k2.5`) | $0.45 / $2.25 | Outside every evaluated family. Best JudgeBench κ of any judge at ≤ $1/M input (0.720), position bias 0.004 (Norman). No rubric-based evidence, and its training lineage is undisclosed (see below). |
| **Escalation / reference** | Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`) | $2 / $10 | Use it if the light judges fail validation. Sonnet was far less lenient than Haiku on failed rubric items (false PASS 0.03 vs 0.12, Pombal). The MultiChallenge authors also tried a Claude 3.5 Sonnet judge and reported "the same conclusions". |

**Excluded:** every model from a family we evaluate.
- **Which families:** Google (Gemini, Gemma), OpenAI (including GPT-oss), Alibaba Qwen, DeepSeek, and
  xAI (still in `Tempo_results.xlsx`).
- **Why:** a judge over-credits outputs from its own model *and its own family*, even on fully
  objective rubric items (Pombal; Li).
- **What this rules out:** the MultiChallenge authors' own March-2026 judge, Gemini 2.5 Pro. It would
  be grading two of our five models' siblings.

**How to choose among the three: run a bake-off, then pick the cheapest that passes.** This is
PaperBench's rule: they chose o3-mini over o1 because it scored F1 0.83 against 0.84 "at one-tenth of
the cost" (p.17–18).
1. Run all three candidates on our own human-labelled validation set (§3).
2. Discard any candidate that misses the acceptance bar.
3. Of the rest, take the cheapest whose κ is not meaningfully below the best — inside its
   bootstrap CI.

Running both benchmarks on the same judge is simpler, but each benchmark is validated separately and
may end up with a different winner.

**Disclosure.** This review was written with Claude (Anthropic), and two of the three candidates are
Claude models. They are on the list because of the family-exclusion rule and the numbers below.
**The bake-off decides — not this table.**

### Why a light judge is defensible here — and the one risk to measure

**For it:**
1. **The decisions are narrow.**
   - MultiChallenge asks one human-written YES/NO question per item.
   - Wonderbread QA asks for a 1–3 score per criterion, with a human reference on two of the four.
   - SOP Generation asks a line-match question.
   - Light judges become competitive exactly when evaluation is broken into instance-specific
     binary questions (RocketEval). The MultiChallenge authors built the rubric for this reason:
     their rubric judge reached 93.95% agreement with humans, against 37.33% for a judge given the
     whole conversation (Sirdeshmukh, Table 4).
2. **A correct reference lets a small judge beat a big one.** Qwen 2.5 7B with a human reference
   scored κ 0.63, against 0.47 for GPT-4o without one (Krumdick, Table 6).
3. **Mid-tier can match frontier.** Gemini 2.5 Flash with a CoT+rubric prompt reached 71.0% / κ 0.549
   at about $0.001 per call. The best frontier configuration (Claude Sonnet 4) reached 69.5% at about
   $0.015 (Soumik). The gap is not statistically significant; the cost gap is 15×.

**Against it — the specific risk is false PASS:**
- **Smaller judges are more lenient.** They mark failed rubric items as satisfied more often:
  - HealthBench: Haiku 0.12 vs Sonnet 0.03 (Pombal, Table 26).
  - Qwen-4B 0.44 vs Qwen-235B 0.09, on the same benchmark.
  - IFEval: 8 of 12 judges passed more than half of the constraints that responses actually failed.
    Haiku was at 0.40 and Sonnet at 0.37 (Table 25).
- **Why it matters for MultiChallenge.** Humans passed only about 23% of responses *(our arithmetic
  from Sirdeshmukh, Table 2)*, so false PASS inflates every score.
- **Cheapest is not enough on its own.** GPT-4o-mini scored F1 0.59, close to random, while the cheap
  *reasoning* model o3-mini scored 0.83 (PaperBench, Table 3).

The bake-off therefore measures the **false-PASS rate per evaluated family**, not just agreement. It
also includes Haiku with a small reasoning budget as a fourth configuration, capped per
[model-parameters.md](../.claude/references/model-parameters.md).

**Two risks we cannot rule out from the literature:**
- **Training lineage.** Preference Leakage shows a judge favours models it shares training data with.
  "Inheritance" relatedness scored 19.3–22.3% average leakage, more than same-family (8.9%) (Li,
  Table 2). Kimi, and Claude, do not disclose their training data. *(Our inference)* The per-family
  false-PASS check in validation is the empirical guard: a judge that is soft on one family shows it
  there.
- **Thinking was suppressed in Norman.** Every judge there ran with reasoning off. Whether reasoning
  helps our judges is measured in the bake-off, not assumed.

### What it costs

Light versus frontier is a difference of tens of dollars, not hundreds.
- **What cost covers:** one full judging pass over **all five evaluated models**, at OpenRouter list
  prices. Batch routes halve these.
- **Token estimates (our assumptions):**
  - MultiChallenge: ~830 input + 150 output per call. A final response is assumed at ~700 tokens;
    ours are not generated yet.
  - Wonderbread QA: ~900 input (the prompt with its few-shots is ~740) + a bare-number output.
  - SOP Generation: ~950 input + ~20 output per call, at 19 calls per demo (the median in the
    authors' shipped results) × 162 gold demos. That demo count depends on
    [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6 question 4.

| Judge | MultiChallenge (1,365 calls) | Wonderbread QA (2,400) | SOP Generation (~15,400) | Total per pass | × 3 replicates |
|---|---:|---:|---:|---:|---:|
| Kimi K2.5 | ~$1.0 | ~$1.0 | ~$7 | **~$9** | ~$28 |
| Claude Haiku 4.5 | ~$2.2 | ~$2.2 | ~$16 | **~$21** | ~$62 |
| Claude Sonnet 5.5 | ~$4.3 | ~$4.4 | ~$33 | **~$41** | ~$124 |

**The real cost is human labelling for validation (§3), not API calls.** SOP Generation is the only
place where a light judge's price matters, because of its call volume.

---

## 2. A general pipeline for an LLM judge

Eight stages. Each one names the rule this project already has, and the paper that supports it.

| # | Stage | What to do | Already in this repo | Evidence |
|---|---|---|---|---|
| 1 | **Specify** | State the criterion, the scale and its *direction*, what the judge is shown, and how scores combine — before choosing a model. | The 13 fields of [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md) | Guerdan: an underspecified task makes validation pick the wrong judge |
| 2 | **Decompose** | Turn holistic judging into narrow per-item questions — binary where possible — one call each. Never collapse many items into one verdict. | MultiChallenge already has this shape; Wonderbread QA calls each criterion separately | Rubric judge 93.95% vs holistic 37.33% (Sirdeshmukh). Collapsing a rubric tree inflated a score from 0.25 to 0.93 (PaperBench, Fig. 6). RocketEval |
| 3 | **Ground** | Give the judge a verified reference or rubric wherever one exists. Never one an LLM wrote unchecked. | Wonderbread's `Human Label`; MultiChallenge's `TARGET_QUESTION` | Krumdick: a correct reference closes the gap between small and large judges; a wrong one is worse than none |
| 4 | **Shortlist** | Exclude every family under evaluation. Shortlist light candidates plus one stronger reference. | §1 above | Pombal; Li; Norman |
| 5 | **Prompt and output** | Keep the upstream prompt verbatim, so the judge model is the only thing that changes. Fix bugs, not wording. Hide model identity. Use structured output or a strict parser, and count parse failures and refusals as their own category. | C2 rule: "never silently score a parse failure as zero" ([JUDGE_RECORD.md](JUDGE_RECORD.md)) | Soumik: style bias is the largest bias and varies by family. Pombal: show one rubric at a time |
| 6 | **Decode and replicate** | Temperature 0 where the model honours it. ≥ 3 replicates with test–retest reported. Pin the model snapshot. | Wonderbread D3: pass temperature 0 to Judge 2 | Norman's validation protocol (§4) |
| 7 | **Validate** | Use a human-labelled sample from *each* benchmark, with ≥ 2 raters on at least a subset. | — | See below |
| 8 | **Run and record** | Checkpoint per row. A judge error is its own count, never a failed item. Declare the substitute judge, fill the record's fields for it, and state that numbers are not comparable to the paper's. | Wonderbread C1; MultiChallenge D2; "Substituting a different judge model" in [JUDGE_RECORD.md](JUDGE_RECORD.md) | Gu: evaluation drift across model versions |

**What stage 7 reports:**
- Raw agreement **and** a chance-corrected statistic — κ or Krippendorff's α — with bootstrap CIs.
- The trivial baseline: always-FAIL or always-majority.
- False-PASS and false-FAIL rates, per evaluated family.
- An acceptance bar fixed *before* the results are seen.

**Evidence for stage 7:**
- At 85% raw agreement, κ was only about 0.48 (Norman).
- A wrong validation metric picked a judge 31% worse (Guerdan).
- The per-family false-PASS rate comes from Pombal.

---

## 3. The pipeline for each benchmark

### 3.1 MultiChallenge

What the judge does today ([JUDGE_RECORD.md](JUDGE_RECORD.md) §2):
- **Input:** one call per (item, attempt). The judge sees the final response and the item's rubric
  question only. The prompt is [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) Appendix C, including
  "Be VERY STRICT!".
- **Output:** a structured `{reasoning, verdict ∈ YES/NO}`.
- **Scoring:** an item passes on `verdict == PASS_CRITERIA`. Attempts combine as any-pass, and the
  headline is a macro mean over the four axes.

| Step | Decision |
|---|---|
| **Judge input** | Keep the final response plus rubric only — no conversation (D5). This is the condition the authors validated at 93.95%. Adding context is the condition that scored 37.33% (Sirdeshmukh, Table 4). |
| **Harness repairs** (already decided) | D1: drop the `max_tokens` kwarg that crashes the judge. D2: exclude `axis == 'NA'` and report `judge_error_count`. Pre-count generation failures (the `FAIL THIS QUESTION` string). Keep the `PASS_CRITERIA` comparison (D6). |
| **Judge swap** | Replace the hard-coded `gpt-4o-2024-08-06` with the chosen judge through OpenRouter's OpenAI-compatible client. Structured output is not supported on every route. *Verify* it for the chosen model; otherwise use JSON plus a strict parse into `{YES, NO}` with a parse-fail counter. |
| **Decoding** | Temperature 0. Run 3 judge replicates per response, take the majority verdict, and report the flip rate. This deviates from upstream's single call; declare it. |
| **Validation set** | Hand-label ~200 (response, rubric) pairs from *our* models' responses, stratified by axis and by evaluated model. Two raters on at least 60, to get the human–human ceiling. Raters see the same input as the judge. *(Size is our inference — see Guerdan and Norman.)* |
| **Acceptance bar** (proposed — adjust before running) | Agreement ≥ 90% overall and per axis (the authors' frontier judge: 92.3–94.9% per axis). κ reported, and clearly above the always-FAIL baseline. *(Humans passed about 23% of responses, so always-FAIL already scores about 77% agreement.)* No evaluated family's false-PASS rate above 1.5× the lowest. |
| **Cost** | About $1–4 per pass over five models (§1). |

**Open decision for the user.** In March 2026 the authors revised 54 tasks to "reduce ambiguity" and
moved their judge to Gemini 2.5 Pro, reporting judge–human agreement up more than 5 points (Scale,
*MultiChallenge Update*, 2026-03-23).
- **What we have:** our vendored copy is the February 2025 commit.
- **Option 1 — the revised tasks:** better rubrics, but the numbers are not comparable to the paper.
- **Option 2 — the vendored set:** comparable to the paper, but the rubrics are known to be
  ambiguous.
- **Either way,** the Gemini judge is excluded (§1).

### 3.2 Wonderbread — Question Answering

What the judge does today ([JUDGE_RECORD.md](JUDGE_RECORD.md) §1, Judge 1):
- **Volume:** 120 items × 4 criteria, one call each — 480 per model.
- **Prompt:** verbatim in [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) Appendix A, with three few-shot
  examples.
- **Reference:** the human reference goes to completeness and soundness only.
- **Output:** a bare number, **1 = best, 3 = worst**.

| Step | Decision |
|---|---|
| **Judge input** | Unchanged: the reference for two criteria, none for the other two. Clarity and compactness have no reference, so they are the criteria most exposed to style bias (Soumik) and to leniency (Pombal). Watch them in validation. |
| **Output handling** (already decided) | Strict parse to {1, 2, 3} plus a `parse_fail` counter (C2). Keep the direction 1 = best and label it in every table (A6). |
| **Run safety** (already decided) | Checkpoint per row: upstream writes its CSV only after the loop, so one error loses everything (C1). Cap the rate-limit retry. |
| **Decoding** | Temperature 0 (already upstream). 3 replicates per (item, criterion), with the median score as the result. Report disagreements. |
| **Validation set — a head start** | The 30-item human-vs-GPT-4 sample is already on disk, giving 120 human scores. Re-judge those exact responses with each candidate. GPT-4 reached 86.7–96.7% exact agreement and Spearman 0.80–0.89 per criterion (record A3). |
| **…and its limit** | n = 30, one rating per item, raters undescribed. Guerdan shows one forced-choice rating per item cannot reliably pick a judge when a criterion is ambiguous. Add ~50 items from *our* models' responses, scored by two raters, and allow "either of two scores is reasonable" where they disagree. |
| **Acceptance bar** (proposed) | Per criterion, exact agreement on the 30-item set no more than 5 points below GPT-4's. Quadratic-weighted κ reported with CI on the extended set. |
| **Cost** | About $1–4 per pass over five models. |

### 3.3 Wonderbread — SOP Generation

What the judge does today ([JUDGE_RECORD.md](JUDGE_RECORD.md) §1, Judge 2):
- **Decision:** for each line of one SOP, return the index of the matching line in the other SOP,
  or −1. This runs in both directions.
- **Volume:** p + g calls per demonstration.
- **Prompt:** [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) Appendix B, in JSON mode.
- **Score:** precision, recall and ordering are tallies of these decisions.

| Step | Decision |
|---|---|
| **Why it matters most for cost** | Around 3,100 calls per model — six times QA. This is where a light judge's price actually matters (§1). |
| **Settings** (already decided) | Pass temperature 0, since upstream defaults to 1.0 (D3). Clear `sop_cache` per configuration, because otherwise an edited prompt silently returns old completions (C2). Keep JSON mode and its one regeneration on malformed output. |
| **Known scorer trap** | `preprocess_sop` strips everything up to the first `.` in each line (B4), which reshapes both what is matched and the denominators. Decide whether to keep upstream behaviour (comparable) or fix it (correct), and declare which. |
| **Validation set** | There is no human comparison upstream (A3). Label ~150 line-match decisions from ~10 demos with two raters, deliberately including lines that should *not* match. Report false-match rate (the leniency risk, Pombal) and miss rate. Where a line plausibly matches two lines, record both as acceptable (Guerdan's response sets). |
| **Acceptance bar** | Set before running. There is no published figure to anchor on. |
| **Scope dependency** | Text-only or multimodal generation is [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6 question 4. The judge is text-only either way. |

SOP Improvement is out of scope: its scorer does not execute (record D1).

---

## 4. The ten papers

Citation counts are from Semantic Scholar, 2026-10-05. Page numbers are PDF pages.

| # | Paper | Venue | Cites | What it settles for us |
|---|---|---|---:|---|
| 1 | Sirdeshmukh et al., *MultiChallenge* | Findings of ACL 2025 | 191 | Why the judge sees only the response and rubric; the 93.95% bar |
| 2 | Wei et al., *RocketEval* | ICLR 2025 | 44 | Light judges work on checklist-style binary questions |
| 3 | Starace et al., *PaperBench* | ICML 2025 (per Semantic Scholar; the PDF prints "Pre-print") | 307 | Validate, then pick the cheapest judge that is not worse |
| 4 | Guerdan et al., *Validating LLM-as-a-Judge Systems under Rating Indeterminacy* | NeurIPS 2025 | 31 | How validation itself can pick the wrong judge |
| 5 | Li et al., *Preference Leakage* | ICLR 2026 | 151 | Same-family and shared-training-data judges are biased |
| 6 | Krumdick et al., *No Free Labels* | COLM 2026 | 65 | References make small judges reliable |
| 7 | Pombal et al., *Self-Preference Bias in Rubric-Based Evaluation* | COLM 2026 | 12 | Self- and family-preference in binary rubric judging; leniency of light judges |
| 8 | Soumik, *Judging the Judges: Bias Mitigation Strategies* | TMLR 2026 | 6 | Mid-tier plus a structured prompt matches frontier at 15× lower cost |
| 9 | Norman et al., *Reliability without Validity* | arXiv preprint, June 2026 | 21 | 21 current judges with prices; a minimum validation protocol |
| 10 | Gu et al., *A Survey on LLM-as-a-Judge* | The Innovation 7(6), 2026 | 1,891 | The framework behind §2 |

#### 1 · Sirdeshmukh et al. (2025) — *MultiChallenge* · [PDF](papers/2025_Sirdeshmukh_MultiChallenge_ACLFindings2025.pdf) · [arXiv:2501.17399](https://arxiv.org/abs/2501.17399)

**Setup.** Six frontier models' responses on all 273 items were labelled by human raters, with two
reviewer layers (p.6). The judge was GPT-4o, with Claude 3.5 Sonnet also tried.

**Findings**
- **Why a rubric.** A judge given the full conversation "yields low alignment with human raters"
  (p.4). Each item therefore gets a human-written YES/NO question that "requires only the final model
  response as context" (p.5).
- **Agreement.** The rubric judge scores 93.95% overall and 92.26–94.85% per axis. The full-context
  judge scores 37.33% (Table 4, p.7).
- **Rankings.** Judge and human scores rank all six models identically (Tables 2–3).
- **Judge model.** Claude gave "the same conclusions" as GPT-4o (p.6–7). No per-judge numbers and no
  light judge were reported.

**Caveats**
- Agreement is raw percent, with no κ. Humans passed about 23% of responses, so always-FAIL already
  scores about 77% *(our arithmetic from Table 2)*.
- Items whose rubric was too hard for a frontier judge were excluded from the release (p.9).
- The full-context baseline also lacked the rubric, so the paper does not show whether the extra
  context or the missing rubric caused the drop.

#### 2 · Wei et al. (2025) — *RocketEval* · [PDF](papers/2025_Wei_RocketEval_ICLR2025.pdf) · [arXiv:2503.05142](https://arxiv.org/abs/2503.05142)

**Setup.** 13 open judges from 0.5B to 12B, compared with GPT-4o.
- GPT-4o writes 5–10 binary checklist questions per query (p.6).
- The light judge answers each one separately.
- The score uses p(Yes)/(p(Yes)+p(No)) from token probabilities, not the decoded word.

**Findings**
- **Diagnosis.** Light judges fail at "comprehension and analysis" of complex responses, not at
  answering narrow questions (p.5).
- **Model ranking.** Gemma-2-2B reaches Spearman 0.965 with human-derived rankings, comparable to
  GPT-4o's 0.979. Under plain CoT it scored 0.818 (Table 3, p.9). The cost reduction is more than
  50-fold.
- **No supervision needed at 2B and up.** The unsupervised score matches the supervised one for
  judges of about 2B and larger. It fails for 1B models.

**Caveat**
- Parity is in *ranking models*, not in per-item agreement. Gemma-2-2B's instance agreement is 57.9%
  against GPT-4o's 66.6% (Table 2).
- The ranked models were chosen to be well separated.
- Logprob access is needed to copy the scoring method.

#### 3 · Starace et al. (2025) — *PaperBench* · [PDF](papers/2025_Starace_PaperBench_ICML2025.pdf) · [arXiv:2504.01848](https://arxiv.org/abs/2504.01848)

**Setup.** 8,316 binary rubric leaves. JudgeEval is built from human-graded leaves of five partial
replications (p.6).

**Findings**
- **Judge comparison** (Table 3, p.6):

  | Judge | F1 | Cost per paper |
  |---|---:|---:|
  | GPT-4o-mini | 0.59 | $8 |
  | GPT-4o | 0.73 | $120 |
  | o1-mini | 0.78 | $72 |
  | **o3-mini** | **0.83** | **$66** |
  | o1 | 0.84 | $830 |

- **The choice.** They chose o3-mini as "the most cost-effective" (p.6), noting that o1's edge "may
  be due to noise and remains futile given the much higher costs" (p.18).
- **Atomic grading matters.** Grading collapsed subtrees inflated one submission's score from 0.25 to
  0.93 (Fig. 6, p.19).

**Caveat**
- The JudgeEval size and the labellers are not stated.
- An OpenAI judge graded OpenAI agents, and self-preference was not examined.

#### 4 · Guerdan et al. (2025) — *Rating Indeterminacy* · [PDF](papers/2025_Guerdan_Rating-Indeterminacy_NeurIPS2025.pdf) · [arXiv:2503.05965](https://arxiv.org/abs/2503.05965)

**Setup.** 9 judges on 11 rating tasks — toxicity, NLI, SummEval and others.

**Findings**
- **Rating indeterminacy.** When criteria "admit multiple valid interpretations", forced-choice
  validation picks the wrong judge.
- **Example.** On toxicity, Claude 3.5 Sonnet ranked first under forced choice but has "31% worse
  consistency with human decisions than GPT o3-Mini" (p.3).
- **What helps.** Fully specify the task (e.g., an explicit rule for borderline cases), or collect
  "select all that could apply" ratings on a validation subset.
  - About 100 paired items suffice (p.10).
  - "very few ratings per item (e.g., 1-3)" lead to poor judge selection (p.61).

**Caveat**
- The human response sets are simulated.
- The 31% figure is a single example.

#### 5 · Li et al. (2026) — *Preference Leakage* · [PDF](papers/2025_Li_Preference-Leakage_ICLR2026.pdf) · [arXiv:2502.01534](https://arxiv.org/abs/2502.01534)

**Setup.** Students are fine-tuned on data from GPT-4o, Gemini-1.5 or LLaMA-3.3, and then judged by
those same models, pairwise, on Arena-Hard and AlpacaEval.

**Findings — average preference-leakage score by relatedness** (Table 2, p.7):

| Relatedness | Leakage |
|---|---:|
| Same model | 23.6% |
| Inheritance | 19.3–22.3% |
| Same family, same series | 8.9% |
| Same family, different series | 2.8% |

- **Hard to detect.** Judges recognise their related students at about chance (Table 5).
- **Subjective questions leak more.** Mathematics 7.7, Programming 31.4 (Fig. 3).
- **The one mitigation that worked.** Contextual calibration on a held-out set, 17.8 → 7.3 (Table 7).

**Caveat**
- Pairwise only, with 2024 judges.
- The family rows have no significance test.

#### 6 · Krumdick et al. (2026) — *No Free Labels* · [PDF](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) · [arXiv:2503.05061](https://arxiv.org/abs/2503.05061)

**Setup.** Five judges, from Qwen 2.5 7B to GPT-4o. Three expert labels per response on finance and
math questions. References given: none, the judge's own, or a human one.

**Findings**
- **The two conditions for a trustworthy judge.** Agreement is high only if the judge "(1) can
  already answer the underlying question or (2) is provided with a correct reference" (p.9).
- **Single-grading κ, no reference → human reference** (Table 6, p.23):

  | Judge | No reference | Human reference |
  |---|---:|---:|
  | Qwen 2.5 7B | 0.24 | 0.63 |
  | Phi 4 | 0.27 | 0.72 |
  | Llama 3.3 70B | 0.42 | 0.78 |
  | GPT-4o | 0.47 | 0.68 |

- **The headline claim.** "providing a human reference to a relatively small model … can yield
  better judgments than using a larger model without human references" (p.7).
- **What does not help.** The judge's own reference makes self-preference worse (p.10). A 5-judge
  jury does not escape the no-reference limit (Table 10).

**Caveat**
- Correctness tasks only.
- Experts graded against the reference, which favours judges that are given one.

#### 7 · Pombal et al. (2026) — *Self-Preference Bias in Rubric-Based Evaluation* · [PDF](papers/2026_Pombal_Self-Preference-in-Rubric-Evaluation_COLM2026.pdf) · [arXiv:2604.06996](https://arxiv.org/abs/2604.06996)

**Setup.**
- **Judges:** 12, each also a generator — Gemma 3, Llama 4, Qwen 3, GPT-5, GPT-oss-120B, Claude
  Sonnet 4.5 and Haiku 4.5.
- **Data:** IFEval and LiveCodeBench, which have programmatic ground truth, and HealthBench.
- **Format:** binary per-rubric verdicts.

**Findings**
- **Self-preference survives objective rubrics.** Judges "can be more than 50% more likely to
  incorrectly mark [a failed rubric] as satisfied when the output is their own". GPT-5 was 20× more
  likely on LiveCodeBench (p.1, p.5).
- **It extends to the family.** On LiveCodeBench the family ratio is 11.91 for GPT-5 and 8.95 for
  GPT-oss (Table 5).
- **Size and leniency.** Smaller judges are more lenient in absolute terms. HealthBench false PASS
  on other families' failed rubrics (Table 26):

  | Smaller judge | False PASS | Larger judge | False PASS |
  |---|---:|---|---:|
  | Haiku | 0.12 | Sonnet | 0.03 |
  | Qwen-4B | 0.44 | Qwen-235B | 0.09 |
  | Gemma-4B | 0.48 | Gemma-27B | 0.34 |
- **Haiku's own self-preference** ratios are 1.15 / 1.71 / 0.90 on the three benchmarks (Table 1).
- **What helps and what does not.**
  - Ensembles lower self-preference without removing it.
  - More reasoning raises accuracy but not fairness (p.19).
  - One rubric at a time shows slightly *more* self-preference than all at once (p.6).

**Caveat**
- No human labels.
- The HealthBench reference is a vote of five of the judges.

#### 8 · Soumik (2026) — *Judging the Judges: Bias Mitigation Strategies* · [PDF](papers/2026_Soumik_Bias-Mitigation-Strategies_TMLR2026.pdf) · [arXiv:2604.23178](https://arxiv.org/abs/2604.23178)

**Setup.** Five judges: Gemini 2.5 Pro and Flash, Claude Sonnet 4, GPT-4o and Llama 3.3 70B. Nine
debiasing strategies. MT-Bench (400 items), LLMBar (200) and a controlled set — all pairwise.

**Findings**
- **The headline.** Gemini 2.5 Flash with a 2-call CoT+rubric prompt reached the highest agreement,
  71.0% / κ 0.549, at about $0.001 per evaluation — "roughly 15× cheaper" than Claude Sonnet 4's best
  (p.1).
- **Style is the largest bias.** Four of five judges preferred markdown 73–97% of the time; humans did
  57% (p.21).
- **Position swap can hurt.** It cost GPT-4o 11.1 points on LLMBar, where one answer is clearly
  better (p.9).

**Caveat**
- The Flash-over-Sonnet gap sits inside overlapping CIs.
- Single author, pairwise only.
- Two of the nine strategies have no results in the PDF.

#### 9 · Norman et al. (2026) — *Reliability without Validity* · [PDF](papers/2026_Norman_Reliability-without-Validity.pdf) · [arXiv:2606.19544](https://arxiv.org/abs/2606.19544)

**Setup.** 21 judges from 9 providers. MT-Bench, JudgeBench and RewardBench. About 541,000 judgments
at temperature 0, with reasoning suppressed.

**Findings**
- **Raw agreement overstates.** Exact match exceeds κ by 33.8–41.3 points on MT-Bench: "a judge
  reporting '85% agreement' … has κ ≈ 0.48" (p.5).
- **Rankings do not transfer.** Judge rankings move by up to 14–15 places between benchmarks.
- **Light, out-of-family judges** (Tables 2–3, 6):

  | Judge | JudgeBench κ | RewardBench κ | Position bias | Test–retest (MT-Bench) |
  |---|---:|---:|---:|---:|
  | Kimi K2.5 | 0.720 | 0.873 | 0.004 | 0.917 |
  | MiniMax M2.7 | 0.715 | 0.834 | 0.020 | 0.888 — the least stable |
  | Claude Haiku 4.5 | 0.653 | 0.873 | 0.041 | 0.935 |
  | GLM-5 | 0.596 | 0.838 | 0.052 | 0.934 |
  | Llama 3.3 70B | 0.283 | 0.769 | 0.057 | 0.954 |

  - The three Anthropic judges average κ 0.770 on JudgeBench, with position bias 0.020.
- **The Minimum Viable Validation Protocol** (p.8):
  1. Report κ or α as the headline.
  2. Measure position bias.
  3. Run ≥ 3 replicates at temperature 0, with caching off.
  4. Validate on ≥ 2 benchmarks of different label types.
  5. If test–retest exceeds 0.95, check that position bias is below 0.10.

**Caveat**
- Preprint, with several internal number inconsistencies.
- Pairwise only.
- List prices only.

#### 10 · Gu et al. (2026) — *A Survey on LLM-as-a-Judge* · [PDF](LLM_as_judge.pdf) · [arXiv:2411.15594](https://arxiv.org/abs/2411.15594)

The journal version (The Innovation, January 2026) of the field's most-cited survey. Kept as the
framework, not as evidence.

**What it contributes**
- **The four building stages:** prompt design, model selection, post-processing, and the evaluation
  pipeline (Fig. 2A, p.4).
- **The "quick practice" loop** (p.7).
- **A meta-evaluation:**
  - Asking for an explanation alongside the score lowered position consistency.
  - A majority vote over 5 runs helped modestly; mean and best-of did not (Tables 2–3, p.16–17).
- **Rules adopted in §2:** avoid same-model judging and anonymise model names (p.14). Watch for
  evaluation drift across model versions (p.22).

**Caveat:** the survey's prose contradicts its own tables in places.

---

## 5. Further reading — 2025 and later, not downloaded

These summaries come from the abstracts only.

- **Salinas et al., *Tuning LLM Judge Design Decisions for 1/1000 of the Cost*** — ICML 2025,
  [arXiv:2501.17178](https://arxiv.org/abs/2501.17178). Searches judge model, prompt and decoding
  jointly, and finds competitive open-weight judges.
- **Xie et al., *Small Language Models as Judges for Rubric-Based Reinforcement Learning*** —
  Findings of EMNLP 2026, [arXiv:2608.30005](https://arxiv.org/abs/2608.30005). Rubric
  criterion-level judging with models down to 1.7B.
- **Rao et al., *JEV vs. LLMs as Rubric Judges*** — 2026,
  [arXiv:2609.29769](https://arxiv.org/abs/2609.29769).
  - Compares flash-tier LLM judges and a classifier, holistic versus one criterion at a time, on nine
    human-labelled panels.
  - The classifier is ahead on binary checklist criteria and behind on ordinal ones.
- **Hong et al., *From Rubrics to Reliable Scores (Rulers)*** — EMNLP 2026,
  [arXiv:2601.08654](https://arxiv.org/abs/2601.08654). Locked rubrics, evidence-grounded verdicts,
  and calibration to human score boundaries. Relevant to Wonderbread's 1–3 scale.
- **Yu et al., *Mitigating Rubric Interference in LLM Judges*** — 2026,
  [arXiv:2608.14684](https://arxiv.org/abs/2608.14684). Judging several rubrics in one call shifts
  each verdict, which supports one call per criterion.
- **Lail et al., *On Cost-Effective LLM-as-a-Judge Improvement Techniques*** — ICML 2026 workshop,
  [arXiv:2604.13717](https://arxiv.org/abs/2604.13717). Ensembling, criteria injection and adaptive
  escalation to a stronger model.
- **Arora et al., *HealthBench*** — OpenAI 2025,
  [arXiv:2505.08775](https://arxiv.org/abs/2505.08775). Physician-written per-conversation rubrics
  graded by a model judge — the closest large-scale analogue to MultiChallenge's design.

---

## 6. Files

Licences were checked on each paper's arXiv abstract page or in the ACL Anthology, 2026-10-05.

| PDF | Licence |
|---|---|
| [MultiChallenge](papers/2025_Sirdeshmukh_MultiChallenge_ACLFindings2025.pdf) | CC BY 4.0 (ACL Anthology version) |
| [RocketEval](papers/2025_Wei_RocketEval_ICLR2025.pdf) | CC BY-NC-SA 4.0 |
| [PaperBench](papers/2025_Starace_PaperBench_ICML2025.pdf) | CC BY 4.0 |
| [Rating Indeterminacy](papers/2025_Guerdan_Rating-Indeterminacy_NeurIPS2025.pdf) | CC BY 4.0 |
| [Preference Leakage](papers/2025_Li_Preference-Leakage_ICLR2026.pdf) | CC BY 4.0 |
| [No Free Labels](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) | CC BY-NC-SA 4.0 |
| [Self-Preference in Rubric Evaluation](papers/2026_Pombal_Self-Preference-in-Rubric-Evaluation_COLM2026.pdf) | CC BY 4.0 |
| [Bias Mitigation Strategies](papers/2026_Soumik_Bias-Mitigation-Strategies_TMLR2026.pdf) | CC BY 4.0 |
| [Reliability without Validity](papers/2026_Norman_Reliability-without-Validity.pdf) | CC BY 4.0 |
| [Survey (journal version)](LLM_as_judge.pdf) | CC BY-NC-ND |

All ten are committed. The filename year is the year of first release; the venue follows it.
