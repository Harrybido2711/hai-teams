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

The evidence is **fifteen papers accepted at 2026 top venues** (§4): five at ICML, three at ICLR,
three at COLM, two at ACL (main track), one at NeurIPS and one at EMNLP (main track). The
benchmark's own MultiChallenge paper (2025) is kept as a primary source and not counted. The
existing judge documentation is reused, not repeated:
- [JUDGE_RECORD.md](JUDGE_RECORD.md) — what each judge is shown and how it scores, with code
  citations.
- [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md) — the thirteen fields any judge must
  record.
- [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) — the verbatim prompts (Appendices A–C).
- [GPT_LLM_AS_JUDGE_GUIDE.md](GPT_LLM_AS_JUDGE_GUIDE.md) — API mechanics.

**How this was produced (2026-10-08).**
- **Selection.** Only papers accepted at ICLR, ICML, ACL, EMNLP, NeurIPS or COLM 2026; arXiv first
  release may be 2025, never 2024. Four searches screened the full accepted lists by theme. AAAI-26
  was screened and had nothing on-topic. NAACL 2026 has no proceedings in the ACL Anthology.
- **Venue proof.** Every acceptance was confirmed on an official source:
  - the conference's own accepted-paper data (iclr.cc, icml.cc, colm.cc), or its poster page
    (neurips.cc);
  - the ACL Anthology;
  - EMNLP 2026's official programme sheet, linked from 2026.emnlp.org/program.
- **PDF versions.** Several PDFs are the arXiv version and print no venue line. §4 says which.
- **Reading.** Each new PDF was read in full, appendices included, by a separate reader. Each number
  carries its PDF page (physical page, 1-based). A number read off a plot is marked *figure-read*.
  Three papers (Li, Krumdick, Pombal) carry over from the previous version, which read them in full.
- **Inference.** Anything that is our reasoning or our arithmetic rather than a paper's finding is
  marked *(our inference)*.
- **Earlier versions.** Commit `b6bfe59` covered ten 2025–2026 papers. That set included four
  2025-venue papers, a journal paper, a preprint and a survey first released in 2024 — see §6.
  Commit `0667bc1` covered 2024 papers.

---

## 1. Which judge model

### The short answer

| Role | Model (OpenRouter id) | Price in / out per M tokens | Why |
|---|---|---|---|
| **Default light judge** | Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`) | $1 / $5 | Outside every evaluated family. The only light out-of-family judge with published rubric numbers. κ 0.808 against human graders on Audio MultiChallenge's binary rubrics. That ties GPT-4.1 and trails only two OpenAI reasoning models (Gosai). Rubric accuracy 0.81 / 0.92 / 0.84 against Sonnet's 0.84 / 0.95 / 0.88 (Pombal). Accepts `temperature`. |
| **Cheapest candidate — untested** | Claude Haiku 5.5 (`anthropic/claude-haiku-5.5`) | $0.10 / $0.50 | Listed on OpenRouter 2026-10-07, one day before this review, and evaluated by no paper. It is in the bake-off only because it costs a tenth of Haiku 4.5. Does **not** accept `temperature` on OpenRouter. |
| **Cheap challenger** | Kimi K2.6 (`moonshotai/kimi-k2.6`) | $0.44 / $2.45 | Outside every evaluated family. 2nd of 18 judges on deep-research rubric verification: balanced accuracy 92.2, against Claude Opus 4.7's 91.7 and Sonnet 4.6's 89.4. 4th on agentic coding (Peng). Run with thinking on, as there. It replaces K2.5, which has no top-venue evidence. Training lineage is undisclosed (see below). |
| **Escalation / reference** | Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`) | $2 / $10 | Use it if the light judges fail validation. On HealthBench, Sonnet's false-PASS rate on failed rubrics was 0.03 against Haiku's 0.12 (Pombal). Sonnet 4.5 matched the best in-family frontier judges on binary rubric criteria (Sharma), and matched or beat Opus 4.5 in a leave-one-human-out test (Li J.). Does **not** accept `temperature` on OpenRouter. |

Prices and the `temperature` support are from OpenRouter's model list on 2026-10-08.

**Excluded:** every model from a family we evaluate.
- **Which families:** Google (Gemini, Gemma), OpenAI (including GPT-oss), Alibaba Qwen, DeepSeek, and
  xAI (still in `Tempo_results.xlsx`).
- **Why:** a judge over-credits outputs from its own model *and its own family*, even on fully
  objective rubric items (Pombal; Li).
- **What this rules out:** the best judges in several studies. o4-mini topped Audio MultiChallenge
  (κ 0.873); Gemini-3-Pro topped IF-RewardBench; GPT-5.4 and Gemini-3.1 Pro topped RuVerBench. So
  did the MultiChallenge authors' own March-2026 judge, Gemini 2.5 Pro.

**Out-of-family, but not shortlisted, on the evidence:**
- **GLM.** On constraint-violation detection, GLM-4.6 scored N-F1 0.531 and GLM-4.5-Air 0.393, against
  0.744 for humans (Wen). Zhipu co-authored and funded that paper. GLM-5.1 did well on deep research
  (91.5) but fell to 8th on agentic coding (Peng).
- **Llama.** Llama-3.3-70B scored N-F1 0.335 (Wen). Llama-3.1-8B was at chance on deep research,
  51.6 (Peng).
- **Mistral Medium.** κ 0.787, below Haiku 4.5 (Gosai).

**How to choose: run a bake-off, then pick the cheapest that passes** *(our rule)*.
1. **Run all five configurations** on our own human-labelled set (§2 stage 7, §3):
   - Haiku 4.5, reasoning off;
   - Haiku 4.5 with a small reasoning budget, capped per
     [model-parameters.md](../.claude/references/model-parameters.md);
   - Haiku 5.5;
   - Kimi K2.6;
   - Sonnet 5.5.
2. **Discard any that misses the acceptance bar.** The bar is fixed before the results are seen.
3. **Rank the survivors paired on the same items,** by false-PASS rate and by Youden's J (q0 + q1 − 1).
   J sets how wide the corrected interval is (Lee). Ranking on a sparse overlap set is unreliable: at
   25% overlap the wrong best of ten judges was picked 36% of the time (Li J., Table 2, p.8).
4. **Take the cheapest** whose numbers are not meaningfully worse than the best (McNemar on the
   disagreements).

Running both benchmarks on the same judge is simpler, but each benchmark is validated separately and
may end up with a different winner.

**Disclosure.** This review was written with Claude (Anthropic), and three of the five configurations
are Claude models. They are on the list because of the family-exclusion rule and the numbers below.
**The bake-off decides — not this table.**

### Why a light judge is defensible here — and the one risk to measure

**For it:**
1. **The decisions are narrow, and narrow decisions are where light judges hold up.**
   - MultiChallenge asks one human-written YES/NO question per item. Wonderbread QA scores one
     criterion per call. SOP Generation asks a line-match question.
   - **Small judges hold up on final answers, and only collapse on long trajectories.** Qwen3.5-9B was
     6.6 points behind the best judge on deep-research final reports, but 17.8 behind on 49K-token
     coding trajectories (Peng, Table 2). Our inputs are final answers.
   - **A small judge on per-unit questions beat holistic frontier judges.** A 14B judge asking one
     presence question per unit beat holistic GPT-4o: Spearman 0.599 against 0.510 (Ananthram,
     Table 6).
   - **A checklist removes the multi-turn penalty.** With a checklist, judges lose nothing on
     multi-turn items (Gemini-3-Pro N-F1 0.693 → 0.688), while holistic pairwise judging does drop
     (Wen, Tables 3–4).
2. **A correct reference lets a small judge beat a big one.** Qwen 2.5 7B with a human reference
   scored κ 0.63, against 0.47 for GPT-4o without one (Krumdick, Table 6).
3. **In strong families the light sibling matches the flagship.** On N-F1, GPT-5-mini scored 0.628
   against GPT-5.1's 0.610, and Gemini-3-Flash 0.660 against Pro's 0.681. GLM is the counter-example:
   Air 0.393 against GLM-4.6's 0.531 (Wen, Table 3). Haiku 4.5 tied GPT-4.1 (Gosai).

**Against it — the specific risk is false PASS:**
- **Light judges' deficit is in accepting what should fail.** In proof verification, small judges
  wrongly accepted flawed proofs 13.3 points more often than frontier judges, but wrongly rejected
  correct ones only 1.1 points more often (Naik, p.5). One failure mode: the judge "fabricated its
  own argument to fix the gap, then declared it correct" (Fig. 4, p.5).
- **Every judge is weaker at catching violations than at confirming compliance.** Across 14 general
  judges, F1 on the "followed" class was 0.75–0.91 and on "not followed" 0.15–0.68. Humans scored
  0.744 on "not followed" (Wen, Table 3).
- **Smaller judges are more lenient on failed rubric items.**
  - HealthBench: Haiku 0.12 against Sonnet 0.03 (Pombal, Table 26).
  - IFEval: 8 of 12 judges passed more than half of the constraints that responses actually failed.
    Haiku was at 0.40 and Sonnet at 0.37 (Table 25).
- **Why it matters for MultiChallenge.** Humans passed only about 23% of responses *(our arithmetic
  from Sirdeshmukh, Table 2)*, so false PASS inflates every score — by roughly the false-PASS rate ×
  0.77 *(our inference)*.

**But the direction of error is specific to each judge, not just to its size.**
- On reference-anchored procedure checking, Claude Opus 4.5 erred towards FAIL: false PASS 6.1%,
  false FAIL 38.2%. Its pass rate was 34.5%, against 51% for the human majority (Chang, Fig. 3,
  p.7 — *our arithmetic* for the error rates).
- Opus 4.7 is described as strict on deep-research reports (Peng, p.7).

The bake-off therefore measures **both error directions, per evaluated model**, not just agreement.

**Three risks the literature does not settle:**
- **Training lineage.** A judge favours models it shares training data with. "Inheritance"
  relatedness scored 19.3–22.3% average leakage, more than same-family (8.9%) (Li, Table 2). Kimi, and
  Claude, do not disclose their training data. *(Our inference)* The per-model false-PASS check in
  validation is the empirical guard: a judge that is soft on one family shows it there.
- **How much of "self-preference" is really judge weakness.** Once items the judge itself fails on
  are controlled for, only 51% of earlier self-preference findings stay significant. Judge
  uncertainty explains 89.6% of the measured effect on average (Roytburg, p.1–2). So excluding
  families is a cheap precaution, not a proven necessity. The sharper check, offered as an option in
  §2, is whether a judge's false PASSes concentrate on items it cannot solve itself.
- **Coverage gaps.** No paper has evaluated Haiku 5.5. No top-venue paper has measured Kimi on a short
  binary rubric like MultiChallenge's.

### What it costs

Light versus frontier is a difference of tens of dollars per pass, not hundreds.
- **What cost covers:** one full judging pass over **all five evaluated models**, at OpenRouter list
  prices, with no reasoning tokens. Batch routes halve these where they exist.
- **Token estimates (our assumptions):**
  - MultiChallenge: ~830 input + 150 output per call. A final response is assumed at ~700 tokens;
    ours are not generated yet.
  - Wonderbread QA: ~900 input (the prompt with its few-shots is ~740) + a bare-number output.
  - SOP Generation: ~950 input + ~20 output per call, at 19 calls per demo (the median in the
    authors' shipped results) × 162 gold demos. That demo count depends on
    [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6 question 4.

| Judge | MultiChallenge (1,365 calls) | Wonderbread QA (2,400) | SOP Generation (~15,400) | Total per pass | × 3 replicates |
|---|---:|---:|---:|---:|---:|
| Claude Haiku 5.5 | ~$0.2 | ~$0.2 | ~$1.6 | **~$2** | ~$6 |
| Kimi K2.6 | ~$1.0 | ~$1.0 | ~$7 | **~$9** | ~$28 |
| Claude Haiku 4.5 | ~$2.2 | ~$2.2 | ~$16 | **~$21** | ~$62 |
| Claude Sonnet 5.5 | ~$4.3 | ~$4.4 | ~$32 | **~$41** | ~$123 |

- **Reasoning tokens are billed as output.** At ~1,000 per call, they add per pass roughly $10
  (Haiku 5.5), $47 (Kimi K2.6), $96 (Haiku 4.5) and $192 (Sonnet 5.5) *(our arithmetic)*.
- **The real cost is human labelling, not API calls.** §2 asks for ~400 human labels per benchmark.
  For scale, How2Everything paid $3,600 for 200 items × 3 annotators (Chang, pp.26–28).
- **SOP Generation is the only place where a light judge's price matters,** because of its call volume.

---

## 2. A general pipeline for an LLM judge

Eight stages. Each one names the rule this project already has, and the paper that supports it.

| # | Stage | What to do | Already in this repo | Evidence |
|---|---|---|---|---|
| 1 | **Specify** | State the criterion, the scale and its *direction*, what the judge is shown, and how scores combine — before choosing a model. Write the raters' codebook at the same time, and pilot it. | The 13 fields of [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md) | Human α rose from 0.273 to 0.593 only after an iterated codebook and a qualification test (Chang, p.6). Decide what a label *means* first: in PoSh a correct detail absent from the reference was the top "error" (Ananthram, p.26) |
| 2 | **Decompose** | Narrow per-item questions, binary where possible, one call each. Never batch several items into one call. | MultiChallenge already has this shape; Wonderbread QA calls each criterion separately | 4–5 rubrics per call cost 9–35 points on long inputs and up to 6.5 on short ones (Peng, Table 13). A checklist removes the multi-turn penalty (Wen). Unit-level matching beat holistic judging (Ananthram) |
| 3 | **Ground** | Give the judge a verified reference or rubric wherever one exists. Never let an LLM write, expand or "clarify" it. | Wonderbread's `Human Label`; MultiChallenge's `TARGET_QUESTION` | A correct reference closes the small–large gap; a wrong one is worse than none (Krumdick). LLM-expanded rubric text cost 15–22 Macro-F1 points (Sharma, Table 7) |
| 4 | **Shortlist** | Exclude every family under evaluation. Shortlist light candidates plus one stronger reference. Include a reasoning-on configuration. | §1 above | Pombal; Li; Roytburg. Turning thinking off cut τb by up to 33% (Wen, Table 5) |
| 5 | **Prompt and output** | Keep the upstream prompt verbatim, so the judge model is the only thing that changes. Fix bugs, not wording. Hide model identity. Use structured output or a strict parser, and count parse failures and refusals as their own category. | C2 rule: "never silently score a parse failure as zero" ([JUDGE_RECORD.md](JUDGE_RECORD.md)) | A one-line strict/flexible suffix moved accuracy by −9.2 to +11.8 (Peng, Table 3). A prompt swap moved balanced accuracy by 27 points (Naik, p.4). Peng scored unparsable answers at random, which hides them — don't |
| 6 | **Decode and replicate** | Temperature 0 where the route accepts it. ≥ 3 replicates, majority verdict, with the flip rate and self-consistency reported. Pin the model snapshot. | Wonderbread D3: pass temperature 0 to Judge 2 | Haiku 5.5 and Sonnet 5.5 accept no `temperature` on OpenRouter, so replicates carry the stability check. Voting gains saturate by 3–5 samples and cannot fix consistent errors (Peng, p.8; Wen, Table 5). Light judges are up to ~25% less self-consistent (Naik, p.1) |
| 7 | **Validate** | A human-labelled sample from *each* benchmark, stratified by evaluated model, with a three-rater panel. | — | See below |
| 8 | **Run and record** | Checkpoint per row. A judge error is its own count, never a failed item. Report the **corrected** score with its interval, not the raw judge rate. Declare the substitute judge, fill the record's fields for it, and state that numbers are not comparable to the paper's. | Wonderbread C1; MultiChallenge D2; "Substituting a different judge model" in [JUDGE_RECORD.md](JUDGE_RECORD.md) | Lee; Chen — see below |

**How stage 7 labels** *(sizes are our inference from the papers cited)*:
- **About 200 items per benchmark,** drawn uniformly *within each evaluated model* and independently of
  the judge's verdict. This one set is both the bake-off set and stage 8's calibration set.
- **One primary rater labels all of them.** A shared panel of ~100 gets two more raters (three in
  all), picked stratified on the primary rater's label (Li J.'s "STRAT" design).
  - At n = 200, double-labelling 50 items gives only 53–61% reliable decisions; 100 items give 85–87%
    (Li J., Table 17, p.22).
  - Two raters are not enough. The leave-one-human-out score ω can then only be 0, 0.5 or 1, and
    ω = 0.5 is the inconclusive case.
- **Raters see exactly what the judge sees.**
- **Pilot the codebook first** (stage 1).

**What stage 7 reports:**
- **Per-class error rates.** False-PASS and false-FAIL rates — equivalently specificity q0 and
  sensitivity q1 — and the judge's pass rate next to the human pass rate.
  - Overall agreement hides strictness: a 6.5-point spread in agreement hid a 26-point spread in pass
    rates (Chang, Fig. 3).
  - Name the positive class in every table. Naik's "false negative" is our false PASS.
- **Chance-corrected agreement.** κ, or quadratic-weighted κ for an ordinal scale, with bootstrap
  CIs clustered by conversation or SOP. Also report J = q0 + q1 − 1.
- **The leave-one-human-out test** (Li J., Eqs 5–6, p.4). Hold each human out in turn. For that
  human, the judge passes if it agrees with the remaining raters at least as well as the held-out
  rater does, minus ε = 0.05. Accept the judge if it passes for at least half of the held-out
  humans.
  - Use observed agreement there, plus AC1 under label skew. With skewed labels, observed agreement
    approved weak judges more often: 36% against AC1's 21% at 5% overlap (Li J., Fig. 2, p.7).
- **Every row broken down by evaluated model.** Judge error shifts with the model that wrote the
  response (Schwinn), and judge rankings shift with the generator, ρ 0.78–0.89 (Peng, Table 5).
- **An acceptance bar fixed before the results are seen.**
- **Optional — the solvability split.** Have the judge answer a subset itself, then compare its false
  PASS on items it solves with items it fails. A judge whose leniency tracks its own failures skews
  comparisons between models *(our inference from Roytburg)*.

**What stage 8 reports:**
- **The raw judge pass rate p̂ is never reported alone.** Report the Rogan–Gladen corrected rate
  θ̂ = (p̂ + q̂0 − 1) / (q̂0 + q̂1 − 1), with Lee's adjusted interval (Eqs 5–8, p.4; code on p.17).
  Pool q̂0 and q̂1 over all evaluated models.
- **The per-model check.** Next to it, report the PPI++ estimate from that model's own labels, which
  does not assume the judge errs alike on every model (Chen). Flag any model where the two disagree
  beyond their intervals.
- **Why both** *(our inference)*. The two papers do not contradict each other:
  - PPI++ is the efficient estimator when calibration items are a random sample of the same responses
    (Chen, p.10).
  - Only Rogan–Gladen stays unbiased when the pass rate differs between calibration and test (Lee,
    Table 2, p.9). Pooling labels across models creates exactly that difference.
- **Expect wide intervals.**
  - With ~200 labels, a corrected per-model 95% interval of about ±0.07–0.11 *(our computation from
    Lee Eq. 10 and Chen at θ ≈ 0.23, q0 = 0.90–0.95, q1 = 0.80–0.90)*.
  - Judge plus calibration beats human-only labels at θ ≈ 0.23 only if q ≳ 0.88 *(our arithmetic from
    Lee, Prop. 6.1)*.
  - On Chatbot Arena, none of seven judges reached that region, Haiku 4.5, Sonnet 4.6 and Opus 4.6
    included (Lee, p.7).
- **Score gaps between models are rescaled, not reordered.** With a pooled q, θ̂_A − θ̂_B =
  (p̂_A − p̂_B) / (q̂0 + q̂1 − 1) *(our inference)*.

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
- **Attempts:** we run `--attempts 1` (record §2 B2). Keep it: under any-pass, false PASSes compound
  as 1 − (1 − FPR)^k *(our inference)*.

| Step | Decision |
|---|---|
| **Judge input** | Keep the final response plus rubric only — no conversation (D5). This is the condition the authors validated at 93.95%; the whole conversation without a rubric scored 37.33% (Sirdeshmukh, Table 4). The authors' audio follow-up gives the judge the full conversation, several atomic rubrics per item and a neutral prompt with no strictness line, but it never compares the two (Gosai, p.5, p.15). Treat full context as an ablation, run only if validation shows the final-response-only call failing. |
| **Harness repairs** (already decided) | D1: drop the `max_tokens` kwarg that crashes the judge. D2: exclude `axis == 'NA'` and report `judge_error_count`. Pre-count generation failures (the `FAIL THIS QUESTION` string). Keep the `PASS_CRITERIA` comparison (D6). |
| **Judge swap** | Replace the hard-coded `gpt-4o-2024-08-06` with the chosen judge through OpenRouter's OpenAI-compatible client. OpenRouter lists structured output for all four candidate models. *Verify* it on the chosen route; otherwise use JSON plus a strict parse into `{YES, NO}` with a parse-fail counter. |
| **Decoding** | Temperature 0 where accepted (Haiku 4.5, Kimi K2.6), default otherwise. Run 3 judge replicates per response, take the majority verdict, and report the flip rate. This deviates from upstream's single call; declare it. |
| **Validation set** | ~200 (response, rubric) pairs from *our* models' responses: equal numbers per evaluated model, spread across the four axes, drawn independently of the judge. One primary rater on all 200. A STRAT panel of ~100 — about 23 primary-PASS and 77 primary-FAIL — gets two more raters. Raters see the same input as the judge. |
| **Pre-screen** (optional, free) | IF-RewardBench's 202 multi-turn instructions, 22 of them from MultiChallenge, can rank candidates on violation detection before our own labels exist (Wen, Table 10). Up to 78 of the 202 are script-labelled rather than human-labelled. Its checklists are not MultiChallenge's rubrics and its judge sees the conversation, so this screens and does not certify *(our inference)*. |
| **Acceptance bar** (proposed — adjust before running) | Specificity (1 − false PASS) ≥ 0.90 and sensitivity ≥ 0.85 on human labels, with CIs; J ≥ 0.75 *(set near Lee's q ≳ 0.88 region — our inference)*. Passes leave-one-human-out with three raters (ω ≥ 2/3). No evaluated model's false-PASS rate above 1.5× the lowest. Raw agreement still reported overall and per axis: the authors' frontier judge reached 92.3–94.9% per axis, and always-FAIL already reaches about 77%. |
| **Reporting** | Per model: the raw judge pass rate p̂ (labelled as raw), the pooled-q Rogan–Gladen rate with its interval as the headline, and the per-model PPI++ check (§2 stage 8). |
| **Cost** | About $0.2–4 per pass over five models (§1). |

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
| **Judge input** | Unchanged: the reference for two criteria, none for the other two. Keep the three few-shot examples: stripping a rubric's examples cost 1.2–3.6 points (Sharma, Table 7). Clarity and compactness have no reference, so they are the criteria most exposed to leniency (Pombal) and to the weakness judges show on style criteria (Wen, Fig. 4). Validate each criterion separately. |
| **Scale** | Keep the upstream 1–3 labels verbatim. ResearchRubrics' "binary is ~20 points better" comes from collapsing labels after judging, not from a better judge (Sharma, p.8 — *our inference*). Report the 3×3 confusion matrix per criterion, so disagreement on the middle level is visible. |
| **Output handling** (already decided) | Strict parse to {1, 2, 3} plus a `parse_fail` counter (C2). Keep the direction 1 = best and label it in every table (A6). |
| **Run safety** (already decided) | Checkpoint per row: upstream writes its CSV only after the loop, so one error loses everything (C1). Cap the rate-limit retry. |
| **Decoding** | Temperature 0 where accepted (already upstream). 3 replicates per (item, criterion), with the median score as the result. Report disagreements. |
| **Validation set — a head start** | The 30-item human-vs-GPT-4 sample is already on disk, giving 120 human scores. Re-judge those exact responses with each candidate. GPT-4 reached 86.7–96.7% exact agreement and Spearman 0.80–0.89 per criterion (record A3). |
| **…and its limit** | n = 30, one rating per item, raters undescribed. With no second rater, it cannot run the leave-one-human-out test. Add ~100 items from *our* models' responses, stratified by model. One primary rater scores all four criteria; a 50-item STRAT panel gets two more raters. |
| **Plan for a failed criterion** | Every one of ten judges failed the leave-one-human-out test on SummEval's 5-point quality ratings (Li J., p.17). Expect clarity or compactness may fail too. Then report that criterion as judge-unvalidated, or human-scored, rather than lowering the bar. |
| **Acceptance bar** (proposed) | Per criterion: exact agreement on the 30-item set no more than 5 points below GPT-4's, and a pass on the leave-one-human-out test on the panel. Quadratic-weighted κ with a CI on the extended set. |
| **Reporting** | Per criterion and model: the judge's mean score and a corrected mean. The corrected mean uses either the per-category estimator (Chen, §5.4) or the k-category confusion-matrix inversion (Lee, App. B). Lee warns the inversion needs enough labels in each category. |
| **Cost** | About $0.2–4 per pass over five models. |

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
| **No batching** | Keep one line per call. Batching three rubric items cost up to 6.5 points even on short inputs (Peng, Table 13). If cost ever argues for batching lines, validate it separately. |
| **Known scorer trap** | `preprocess_sop` strips everything up to the first `.` in each line (B4), which reshapes both what is matched and the denominators. Decide whether to keep upstream behaviour (comparable) or fix it (correct), and declare which. |
| **One-to-many matches** | One generated line can merge two gold steps, or a collective can stand for several units. Forcing an alignment fails there (Ananthram, pp.4–5). Check how upstream's tallies treat duplicate indices, and tag such lines in the validation set so their error rate is measured *(our inference)*. |
| **Validation set** | No human comparison exists upstream (A3). Label all line decisions, both directions, for ~12 demos spread across evaluated models (~230 decisions). Sample whole demos, because lines cluster within an SOP. A primary rater does all of them; two more raters do half the demos. Raters label *match against the gold SOP*, not "is this step correct": a correct extra step is a non-match (Ananthram, p.26). |
| **What to report** | Per direction, separately. In PoSh, per-unit checks were much weaker in the reference → generation direction: F1 0.754, against 0.941 the other way (Ananthram, Table 9, p.26). That direction is Wonderbread's recall. Report false-match and miss rates per candidate: a fail-biased judge such as Opus 4.5 on procedures (Chang) would under-match and depress both precision and recall. Report also per-SOP judge-vs-human precision, recall and ordering, and mean SOP line count per model next to the scores. Verbosity raised judged procedure scores (Chang, p.34). |
| **Expect low human agreement** | Agreement on *where* a procedure fails was α 0.307, against 0.593 for the verdict alone (Chang, p.7). Line-level matching is likely closer to the first *(our inference)*, so budget a codebook pilot. |
| **Acceptance bar** | Set before running. No published figure exists for line-level matching. For whole procedures, humans agreed with a leave-one-out majority 84.7–88.5% of the time (Chang, p.7). |
| **Scope dependency** | Text-only or multimodal generation is [JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6 question 4. The judge is text-only either way. |

SOP Improvement is out of scope: its scorer does not execute (record D1).

---

## 4. The fifteen papers

Citation counts are from Semantic Scholar, 2026-10-08. Page numbers are PDF pages.

| # | Paper | Venue (where confirmed) | Cites | What it settles for us |
|---|---|---|---:|---|
| 1 | Gosai et al., *Audio MultiChallenge* | ACL 2026 main (Anthology 2026.acl-long.1654) | 24 | Haiku 4.5 against human graders on MultiChallenge-style rubrics |
| 2 | Wen et al., *IF-RewardBench* | ACL 2026 main (Anthology 2026.acl-long.1092) | 8 | Every judge is weak at catching violations; light vs frontier per family; thinking and voting |
| 3 | Pombal et al., *Self-Preference Bias in Rubric-Based Evaluation* | COLM 2026 (colm.cc accepted list) | 12 | Self- and family-preference in binary rubric judging; Haiku's leniency |
| 4 | Li et al., *Preference Leakage* | ICLR 2026 poster (iclr.cc accepted list) | 151 | Same-family and shared-training-data judges are biased |
| 5 | Roytburg et al., *Are LLM Evaluators Really Narcissists?* | ICML 2026 (icml.cc accepted list) | 7 | Most measured self-preference is judge weakness on hard items |
| 6 | Naik et al., *Do We Need Frontier Models to Verify Mathematical Proofs?* | COLM 2026 (colm.cc accepted list) | 3 | Small judges' deficit is false accepts |
| 7 | Krumdick et al., *No Free Labels* | COLM 2026 (colm.cc accepted list) | 65 | References make small judges reliable |
| 8 | Sharma et al., *ResearchRubrics* | ICLR 2026 poster (iclr.cc accepted list) | 84 | Never let an LLM expand a rubric; what "binary beats ternary" really shows |
| 9 | Peng et al., *Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?* (RuVerBench) | EMNLP 2026 main (programme sheet 6955-MAIN) | 4 | Kimi K2.6 ≈ Opus 4.7; one rubric per call; prompt sensitivity |
| 10 | Ananthram et al., *PoSh* | ICLR 2026 poster (iclr.cc accepted list) | 1 | Two-way unit matching, the closest design to SOP precision/recall |
| 11 | Chang et al., *How2Everything* | ICML 2026 (icml.cc accepted list) | 0 | Judging a procedure against a reference; error direction per judge |
| 12 | Lee et al., *How to Correctly Report LLM-as-a-Judge Evaluations* | ICML 2026 (icml.cc accepted list) | 31 | The corrected pass rate and its interval |
| 13 | Chen et al., *Efficient Inference for Noisy LLM-as-a-Judge Evaluation* | ICML 2026 (icml.cc accepted list) | 10 | When PPI++ beats Rogan–Gladen |
| 14 | Li, Mukherjee & Pal, *LLM Judge Validation Under Sparse Overlap* | NeurIPS 2026 poster (neurips.cc/virtual/2026/poster/154278) | 0 | How many items need several raters, and how to pick them |
| 15 | Schwinn et al., *A Coin Flip for Safety* | ICML 2026 (icml.cc accepted list) | 20 | Validate per generating model; agreement among judges is not correctness |
| — | Sirdeshmukh et al., *MultiChallenge* (primary source) | Findings of ACL 2025 | 194 | Why the judge sees only the response and rubric; the 93.95% bar |

"Li" alone means Li et al. (Preference Leakage); "Li J." means Li, Mukherjee & Pal.

### 4.1 Which judge model

#### 1 · Gosai et al. (2025) — *Audio MultiChallenge* · [PDF](papers/2025_Gosai_Audio-MultiChallenge_ACL2026.pdf) · [arXiv:2512.14865](https://arxiv.org/abs/2512.14865)

**Setup.**
- **Data.** The direct follow-up to MultiChallenge, on the same axes. 452 conversations, 1,712
  binary atomic rubrics (Table 1, p.3).
- **Judge call.** The judge sees the full conversation plus one rubric item. The prompt is
  HealthBench-style, with no strictness line, and returns JSON `{explanation, criteria_met}`
  (Fig. 8, p.15). The chosen judge, o4-mini, ran at temperature 1 with one call per rubric (p.14).
- **Human labels.** Graders labelled each rubric once, on GPT-4o Audio or Gemini 2.5 Pro responses
  (p.8).

**Findings — agreement with human graders on 1,712 rubrics** (Table 3, p.8):

| Judge | Family (ours) | Cohen's κ | Macro-F1 |
|---|---|---:|---:|
| o4-mini (chosen) | OpenAI — excluded | 0.873 | 0.937 |
| GPT-5-mini | OpenAI — excluded | 0.870 | 0.935 |
| GPT-4.1 | OpenAI — excluded | 0.811 | 0.906 |
| **Claude Haiku 4.5** | **out of family** | **0.808** | **0.904** |
| Mistral Medium | out of family | 0.787 | 0.894 |
| DeepSeek V3.1 | DeepSeek — excluded | 0.765 | 0.882 |

- **o4-mini was chosen on agreement alone;** cost is never discussed (p.8).
- **The authors exclude only the models they evaluate, not their families** (p.8).

**Caveats**
- No human–human agreement, confusion matrix, per-axis split or CI. So whether Haiku's gap is
  leniency or harshness is unknown.
- Haiku's thinking setting is not stated.
- PASS is the majority class here: the judge-scored rubric pass rate is 65–80% (Table 2). That is
  the opposite of text MultiChallenge's ~23% *(our inference)*.

#### 2 · Wen et al. (2026) — *IF-RewardBench* · [PDF](papers/2026_Wen_IF-RewardBench_ACL2026.pdf) · [arXiv:2603.04738](https://arxiv.org/abs/2603.04738)

**Setup.**
- **Data.** 842 instructions: 393 single-turn, 202 multi-turn (22 of them from MultiChallenge) and
  247 system-prompt. 5.38 constraints each (Table 2, p.5; Table 10, p.18). 74.6% of constraints are
  "followed" (p.5).
- **Judge call.** A binary Followed / Not-followed verdict per constraint. All constraints go in one
  call, with full context, and the prompt says "as strict as possible" (G.2, p.15).
- **Human labels.** Two annotators per item plus an inspector, κ 0.67 before adjudication (p.15). Some
  gold labels come from scripts, not humans: the IFBench, Multi-IF and IHEval items (p.13).

**Findings**
- **Violation detection — F1 on the "not followed" class** (Table 3, p.6). Human 0.744.
  - Excluded families: Gemini-3-Pro 0.681, Gemini-3-Flash 0.660, GPT-5-mini 0.628, GPT-5.1 0.610,
    DeepSeek-V3.2 0.496, Qwen-2.5-7B 0.151.
  - Out of family: **GLM-4.6 0.531, GLM-4.5-Air 0.393, Llama-3.3-70B 0.335, Llama-3.1-8B 0.297.**
  - F1 on the "followed" class sits at 0.75–0.91 for all of them. A judge that always says "followed"
    would score about 0.854 there *(our arithmetic)*. "all general LLMs exhibit relatively low negative
    F1 scores" (p.7).
- **Multi-turn.** With a checklist, judges lose nothing on multi-turn items: Gemini-3-Pro scores
  N-F1 0.693 → 0.688. Without one, pairwise judging drops: Gemini-3-Flash τb 0.589 → 0.460
  (Tables 3–4, p.6).
- **Thinking and voting.** Turning thinking off cut τb by 11% for Qwen-3-32B and 33% for GLM-4.6.
  A majority vote over 3–9 samples added 3–28% τb, saturating by K = 5–7 (Table 5, p.8).
- **Hardest constraints.** Situation and style constraints are the hardest, with the largest gap to
  humans (Fig. 4, p.7).

**Caveats**
- No Claude, Kimi or Mistral judge was tested.
- Zhipu (GLM) co-authored the paper and funded it.
- The thinking and voting results come from two open judges, measured in τb.
- The judge's context was never ablated.

#### 3 · Pombal et al. (2026) — *Self-Preference Bias in Rubric-Based Evaluation* · [PDF](papers/2026_Pombal_Self-Preference-in-Rubric-Evaluation_COLM2026.pdf) · [arXiv:2604.06996](https://arxiv.org/abs/2604.06996)

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
- **Rubric accuracy, IFEval / LiveCodeBench / HealthBench** (Table 1, p.6): Haiku 0.81 / 0.92 / 0.84;
  Sonnet 0.84 / 0.95 / 0.88. Haiku's self-preference ratios are 1.15 / 1.71 / 0.90.
- **Size and leniency.** Smaller judges are more lenient in absolute terms. HealthBench false PASS
  on other families' failed rubrics (Table 26):

  | Smaller judge | False PASS | Larger judge | False PASS |
  |---|---:|---|---:|
  | Haiku | 0.12 | Sonnet | 0.03 |
  | Qwen-4B | 0.44 | Qwen-235B | 0.09 |
  | Gemma-4B | 0.48 | Gemma-27B | 0.34 |
- **What helps and what does not.**
  - Ensembles lower self-preference without removing it.
  - More reasoning raises accuracy but not fairness (p.19).
  - One rubric at a time shows slightly *more* self-preference than all at once (p.6).

**Caveat**
- No human labels.
- The HealthBench reference is a vote of five of the judges.

#### 4 · Li et al. (2026) — *Preference Leakage* · [PDF](papers/2025_Li_Preference-Leakage_ICLR2026.pdf) · [arXiv:2502.01534](https://arxiv.org/abs/2502.01534)

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

#### 5 · Roytburg et al. (2026) — *Are LLM Evaluators Really Narcissists?* · [PDF](papers/2026_Roytburg_Self-Preference-Sanity-Checks_ICML2026.pdf) · [arXiv:2601.22548](https://arxiv.org/abs/2601.22548)

**Setup.**
- **Judges.** 16: Llama 3.x and 4-Scout, Qwen2.5 from 0.5B to 72B, Gemma-2, GPT-4 and GPT-3.5.
- **Data.** Pairwise judging, re-running four earlier self-preference pipelines on nine datasets.
  The oracle is exact match, code execution, or an LLM majority. No human gold (pp.5, 13–14).
- **The control.** For every item where the judge wrongly prefers itself, compare against a proxy
  response from another model with the same oracle outcome on the same item. The proxy is equally
  hard but carries no "self" (Eqs 5–9, pp.4–5).

**Findings**
- **About half of earlier findings survive.** "only 51% of examples in previous findings retain
  statistical significance" (p.1). That is 28 of 54 rows of Table 1 *(our recount)*.
- **Judge uncertainty explains most of the effect.** It accounts for "an average of 89.6% of measured
  self-preference" (p.2).
- **The residual is model-specific, not tied to one provider.** The largest is out-of-family:
  Llama-3.3-70B on MMLU, 65.4 → 23.3 (Table 1, p.6).
- **Same-provider proxies barely matter.** Dropping proxies from the judge's own provider moved the
  baseline by 0.003 (Table 8, pp.14–15).
- **Chain of thought is not a reliable fix** (Table 16, p.21).

**Caveats**
- Pairwise only, on 2024-era open models, with no human labels.
- The abstract and the introduction describe the 89.6% differently.

#### 6 · Naik et al. (2026) — *Do We Need Frontier Models to Verify Mathematical Proofs?* · [PDF](papers/2026_Naik_Frontier-Models-for-Proof-Verification_COLM2026.pdf) · [arXiv:2604.02450](https://arxiv.org/abs/2604.02450)

**Setup.**
- **Judges.** Binary proof verification by frontier GPT-5.2 and Gemini 3.1 Pro, against small
  gpt-oss-20b/120b and Qwen3.5-35B/122B. Three runs per proof, under three upstream prompts (pp.3–4).
- **Labels.** Human expert grades from IMO-GradingBench, ProofArena and ProofBench.
- **Terminology.** "Positive" means flawed, so the paper's **FNR is our false PASS**.

**Findings**
- **The deficit is false accepts.** "the mean difference in FNRs between small and frontier models
  is 13.3% while the mean difference in FPRs is only 1.1%" (p.5).
- **Balanced accuracy.** Small judges are "only up to ∼10% behind": Qwen3.5-35B 76.6 against Gemini
  3.1 Pro (low) 87.7 — strictly 11.1 points (p.4).
- **Self-consistency** is up to ~25% lower for small judges (p.1).
- **The prompt matters as much as the model.** A prompt swap moved GPT-5.2-high's balanced accuracy by
  27.3 points (p.4).
- **The prompt ensemble narrows the gap but does not close it.** It uses 12 calls over 8 prompts per
  proof, and no cost is reported (Fig. 11, p.20).

**Caveats**
- Every judge belongs to a family we exclude.
- One domain only.
- The ensemble was tuned partly on its own test set.
- The PDF is the arXiv v1.

#### 7 · Krumdick et al. (2026) — *No Free Labels* · [PDF](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) · [arXiv:2503.05061](https://arxiv.org/abs/2503.05061)

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

### 4.2 How to pose the judging call

#### 8 · Sharma et al. (2025) — *ResearchRubrics* · [PDF](papers/2025_Sharma_ResearchRubrics_ICLR2026.pdf) · [arXiv:2511.07685](https://arxiv.org/abs/2511.07685)

**Setup.**
- **Data.** 101 deep-research prompts with 2,593 expert-written criteria.
- **Judges.** GPT-5, Claude Sonnet 4.5 and Gemini 2.5 Pro, one criterion per call (Fig. 19, p.25).
- **Human labels.** Nine expert annotators over 303 responses. No inter-rater agreement is reported
  (p.8).

**Findings**
- **Macro-F1 against humans, binary / ternary** (Table 6, p.11; means *our arithmetic*):
  - GPT-5: 0.723 / 0.546.
  - **Claude Sonnet 4.5: 0.734 / 0.529.**
  - Gemini 2.5 Pro: 0.735 / 0.561.
- **"Binary" was not a separate judge call.** It collapses Partially Satisfied into Not Satisfied
  after judging (p.8). So the ~20-point gap is partly mechanical *(our inference)*.
- **LLM-expanded rubric text lowered agreement in every cell.** It cost 19.3–21.6 points binary and
  15.0–18.8 ternary. Stripping a rubric's examples cost 1.2–2.8 and 2.5–3.6 (Table 7, p.11).
- **Length correlates with the judged score at r 0.17–0.28** (Fig. 7, p.10).

**Caveats**
- No light judge was tested, and there is no human–human ceiling.
- The prose on the example effect contradicts Table 7.
- Table 7's judge is not named.

#### 9 · Peng et al. (2026) — *RuVerBench* · PDF local only (arXiv licence) · [arXiv:2606.29920](https://arxiv.org/abs/2606.29920)

**Setup.**
- **Data.** 2,458 rubric instances: deep research (DR) 1,615 and agentic coding (AC) 843.
- **Human labels.** Two independent label sets, then adjudication. Agreement 90.4%, κ 0.808 (p.4,
  p.14).
- **Judges.** 18, one rubric per call, with the upstream prompts verbatim. Metric: mean balanced
  accuracy (pp.4–5, 12, 15).

**Findings — balanced accuracy, DR / AC** (Table 2, p.5; human figures p.6, p.14):

| Judge | Family (ours) | DR | AC |
|---|---|---:|---:|
| Gemini-3.1 Pro | Google — excluded | 94.7 | 86.5 |
| GPT-5.4 | OpenAI — excluded | 91.4 | 89.4 |
| **Kimi K2.6** | **out of family** | **92.2** | **84.3** |
| **Claude Opus 4.7** | **out of family** | **91.7** | **85.0** |
| **GLM-5.1** | out of family | 91.5 | 80.8 |
| **Claude Sonnet 4.6** | **out of family** | **89.4** | **83.0** |
| Qwen3.5-9B | Qwen — excluded | 88.1 | 71.6 |
| GPT-OSS-20B | OpenAI — excluded | 85.6 | 72.3 |
| Llama-3.1-8B | out of family | 51.6 | 61.7 |
| Human (mean of two label sets vs gold) | — | 94.49 | 90.50 |

- **Batching hurts.** In AC, 4–5 rubrics per call lost 9–35 points. In DR, it moved accuracy by
  −6.5 to +2.3 (Table 13, p.18).
- **Prompt wording moves results.** A strict/flexible suffix shifted AC by −9.2 to +11.8 and DR by
  −4.0 to +2.4 (Table 3, p.7).
- **Voting helps little.** Voting at temperature 1 gained at most 1.2 in DR and 4.5 in AC, mostly by
  3–5 votes. "more votes cannot correct errors that the same judge makes consistently" (p.8).
- **Judges are not interchangeable.** Gemini, GPT-5.4 and Opus share only 16.1% of their DR errors
  (p.7).
- **Rankings depend on who generated the output:** Spearman 0.78–0.89 between generators (Table 5,
  p.12).

**Caveats**
- Only balanced accuracy is reported: no per-judge false-positive or false-negative rate.
- The ablations ran on six judges, none of them Claude. No Haiku was tested.
- Unparsable answers were scored at random.
- The PDF is the arXiv v2 and prints no venue.

#### 10 · Ananthram et al. (2025) — *PoSh* · [PDF](papers/2025_Ananthram_PoSh_ICLR2026.pdf) · [arXiv:2510.19060](https://arxiv.org/abs/2510.19060)

**Setup.**
- **Task.** Judging detailed image descriptions against a reference.
- **Units.** Built from a scene graph by spaCy plus coreference, with no LLM in the parse.
- **The judge.** Qwen3-14B answers one presence question per unit, scored 1–5 by logit weighting.
  - Mistakes check generated units in the reference: precision.
  - Omissions check reference units in the generation: recall (pp.4–5, 7, 17).
- **Human labels.** 24 art-history students. Coarse α 0.41–0.51 (Table 5, p.21).

**Findings**
- **Spearman with humans, mistakes / omissions / overall** (DOCENT; Tables 3 and 6, pp.9, 25):
  - PoSh: 0.519 / 0.581 / 0.599.
  - Holistic GPT-4o (reference + image): 0.484 / 0.380 / 0.510.
  - Holistic Qwen3-32B: 0.282 / 0.286 / 0.289.
  - Holistic GPT-5 (reference + image): 0.604 / 0.477 / 0.602.
- **The gain is mostly on omissions,** because every reference unit is listed and checked (p.9).
- **But per-unit verification is weaker in the recall direction:** F1 0.941 for mistakes against
  0.754 for omissions, on 620 questions (Table 9, p.26).
- **The top precision-side error** is "generations specifying correct details not present in the
  reference" (p.26).
- **Forcing an alignment fails on collectives** ("trio" vs three individuals). A presence question
  avoids that (pp.4–5).

**Caveats**
- No out-of-family judge and no judge-size ablation.
- The verifier check covers only two description pairs, and reports the best F1 over thresholds.
- The domain is image description.

#### 11 · Chang et al. (2026) — *How2Everything* · [PDF](papers/2026_Chang_How2Everything_ICML2026.pdf) · [arXiv:2602.08808](https://arxiv.org/abs/2602.08808)

**Setup.**
- **The judge call.** The judge sees the goal, a reference procedure and the generated procedure. It
  lists "critical failures", each linked to steps, then gives a binary verdict (Fig. 22, pp.49–53).
- **Human labels.** 200 items, three Prolific annotators. α was 0.273 in early pilots and 0.593 after
  an iterated codebook and a qualification test. α on the failure's *location* was 0.307 (pp.6–7).

**Findings — agreement with the human majority, n = 200** (Fig. 3, p.7). Error rates are *our
arithmetic* from the figure's class sizes:

| Judge | Agreement | False PASS | False FAIL | Judge pass rate |
|---|---:|---:|---:|---:|
| GPT-5 | 83.0% | 16.3% | 17.6% | 50.0% |
| GPT-4.1 | 82.5% | 21.4% | 13.7% | 54.5% |
| How2Judge (Qwen3-8B distilled from GPT-5) | 80.5% | 20.4% | 18.6% | 51.5% |
| **Claude Opus 4.5** | **77.5%** | **6.1%** | **38.2%** | **34.5%** |
| Gemini 2.5 Pro | 76.5% | 1.0% | 45.1% | 28.5% |
| Humans, leave-one-out | 84.7–88.5% | — | — | majority 51.0% |

- **Absolute scores depend on the judge; rankings did not.** All five frontier generators ranked
  identically under every judge, while absolute scores differed by up to ~28 points (Fig. 5, p.9).
- **Verbosity raised scores:** an odds ratio of 1.012–1.018 per percentage point of generated length
  over reference length (p.34).
- **Even GPT-5 needed two runs.** Its distillation labels were kept only when two runs agreed (p.29).

**Caveats**
- n = 200 and no CIs.
- Opus is the only out-of-family judge.
- The verdict is holistic, not line matching.
- The PDF is the arXiv v1.

### 4.3 How to validate and report

#### 12 · Lee et al. (2025) — *How to Correctly Report LLM-as-a-Judge Evaluations* · [PDF](papers/2025_Lee_Correctly-Report-LLM-Judge_ICML2026.pdf) · [arXiv:2511.21140](https://arxiv.org/abs/2511.21140)

**Setup.**
- **Model.** Binary correct/incorrect. q0 is specificity and q1 sensitivity, both estimated on a
  human-labelled calibration set (Eq. 3, p.3).
- **Data.** Chatbot Arena: six models, 10% calibration, one crowd vote per pair.
- **Judges.** GPT-4.1-mini is the main judge. Claude Haiku 4.5, Sonnet 4.6 and Opus 4.6, GPT-4.1 and
  Gemini-3-Flash appear in Fig. 6 and App. K (pp.6–8, 22).

**Findings**
- **The estimator.** θ̂ = (p̂ + q̂0 − 1) / (q̂0 + q̂1 − 1) (Eq. 5), with an adjusted-Wald interval
  (Eqs 6–8, p.4) and code on p.17. It requires q0 + q1 > 1.
- **Coverage.** Naive p̂ intervals have near-zero coverage except at a few θ; this interval holds
  about 95% (Fig. 5, p.7). On real data it covered 95–99% (Table 1, p.8).
- **It survives a prevalence shift.** It stays unbiased when the pass rate differs between
  calibration and test. The PPI difference estimator and calibration-only estimators do not (Table 2,
  Fig. 8, p.9).
- **Judge plus calibration often loses to human-only labels.** It only wins inside a region of
  (q0, q1), which is empty below q ≈ 0.854 (p.16). "none of the judges we evaluate falls inside the
  favorable region" (p.7).
- **Ranking recovery.** For Haiku 4.5, the exact model ranking was recovered 7% of the time before
  correction and 40% after (Table 4, p.22).
- **A k-category version** inverts the estimated confusion matrix (App. B, p.13).

**Caveats**
- Preference data with one crowd vote per pair.
- The estimator assumes Pr(judge verdict | true label) is the same everywhere. The authors name
  length and style as the threat to that (App. A, p.12).
- The text says m ≈ 200 gives an interval shorter than 0.1 (p.4). The paper's own code gives 0.130 at
  m = 200; 0.1 is crossed at m ≈ 362 with equal allocation, or ≈ 238 with adaptive allocation *(our
  computation with the paper's code)*.

#### 13 · Chen et al. (2026) — *Efficient Inference for Noisy LLM-as-a-Judge Evaluation* · [PDF](papers/2026_Chen_Noisy-Judge-Inference_ICML2026.pdf) · [arXiv:2601.05420](https://arxiv.org/abs/2601.05420)

**Setup.**
- **Estimators compared.** Rogan–Gladen, PPI, PPI++, MLE and the efficient influence function (EIF),
  when calibration items are a random sample of the same population (pp.4–10).
- **Real data.** Arena-human-preference-140k: Claude Opus 4 against Gemini 2.5 Flash, Gemini 2.5 Pro
  and Qwen3-235B. Judges GPT-4o-mini and GPT-5.2, with about 45 calibration labels (pp.15–16).

**Findings**
- **Variance ordering.** EIF ≡ MLE ≡ PPI++ ≺ PPI ≺ Rogan–Gladen (p.10). "the asymptotic variance of
  the PPI estimator is strictly smaller than that of the Rogan–Gladen estimator" (p.3).
- **Real data.**
  - PPI++/EIF intervals were 0.27–0.30 wide. Rogan–Gladen's were 0.76–0.96, at 100% coverage.
  - Naive coverage fell to 0% (p.16).
  - Those judges had q̂0 + q̂1 of only 1.14–1.46 (Table 1, p.16). That is the weak-judge regime where
    Rogan–Gladen blows up.
- **Ordinal outcomes.** A per-category EIF is the most efficient option (§5.4, p.19).

**Caveats**
- Prevalence shift is explicitly deferred (p.17).
- The real-data judges are in-family and weak.
- The PDF is the arXiv v1.

#### 14 · Li, Mukherjee & Pal (2026) — *LLM Judge Validation Under Sparse Overlap* · [PDF](papers/2026_Li_Sparse-Overlap-Validation_NeurIPS2026.pdf) · [arXiv:2609.31857](https://arxiv.org/abs/2609.31857)

**Setup.**
- **The test.** Leave-one-human-out: accept the judge if ω ≥ 0.5 (Eqs 5–6, p.4).
- **Data.** Sparse overlap is simulated by masking dense human matrices: WAX, CeBaB, SummEval and
  Lesion.
- **Judges.** 10, including Claude Opus 4.5, Claude Sonnet 4.5, Llama-3.1 and Mistral-v0.3 (p.6).

**Findings**
- **Sparse overlap picks the wrong judge.** "at 5% pairwise overlap, wrong-decision rates reach 25%
  and the probability of selecting the wrong best judge among ten candidates is 65%" (p.1). The wrong
  best judge is still picked 36% of the time at 25% overlap (Table 2, p.8).
- **Stratified selection has a cost.** Picking the overlap items by the primary rater's label halves
  false rejections (.290 → .142) but raises false approvals (.112 → .172) (Table 3, p.9).
- **The recipe.** "F = p̂o with STRAT, ρ ≥ 0.25, K = 3–5 raters, extra overlap for ranking" (p.10).
  At n = 200, 25% overlap gives 53–61% reliable decisions and 50% overlap gives 85–87% (Table 17,
  p.22).
- **Borderline judges stay borderline.** Judges at ω = 0.5 remain unreliable at any overlap. Treat
  them as inconclusive, and require ω ≥ 0.6 when false approvals are costly (Table 4, p.9; E.2, p.23).
- **Every judge failed SummEval's 5-point quality ratings** (p.17).
- **Sonnet 4.5 is enough.** It matched or beat Opus 4.5 on ω in all three matrices where judges pass
  (Tables 13–15).

**Caveats**
- Simulated sparsity, a single judge run, and categorical labels rather than rubrics.
- Excluding the easy SummEval rejections, the wrong-decision rate at 5% overlap is ≈ 33% *(our
  computation)*.
- The PDF is the arXiv v2 and prints "Preprint".

#### 15 · Schwinn et al. (2026) — *A Coin Flip for Safety* · [PDF](papers/2026_Schwinn_Coin-Flip-for-Safety_ICML2026.pdf) · [arXiv:2603.06594](https://arxiv.org/abs/2603.06594)

**Setup.**
- **Judges.** Four fine-tuned safety classifiers.
- **Labels.** 6,642 human labels on outputs from 4 models under up to 5 attacks, but only on outputs
  that another classifier had flagged (pp.3–5).

**Findings**
- **Accuracy is near chance:** 0.51–0.59 per judge (Table 1, p.9).
- **The generating model matters more than the attack.** Judge-averaged accuracy spreads 0.08 across
  generating models and 0.03 across attacks *(our computation from Figs 3 and 10)*.
- **The same outputs can be easy or hopeless depending on the judge:** AUROC ranges from 0.37 to 0.91
  (Fig. 11, p.12).
- **Consensus is not correctness.** "judges achieve near-unanimous consensus … yet consistently fail
  to align with human ground truth" (p.7).

**Caveats**
- Small safety classifiers, not general judges, so the numbers transfer weakly.
- The selection effect: only flagged outputs were labelled.
- The PDF is the arXiv v2 and prints "Preprint".

### 4.4 Primary source — the benchmark's own paper

#### Sirdeshmukh et al. (2025) — *MultiChallenge* · [PDF](papers/2025_Sirdeshmukh_MultiChallenge_ACLFindings2025.pdf) · [arXiv:2501.17399](https://arxiv.org/abs/2501.17399)

**Setup.** Six frontier models' responses on all 273 items were labelled by human raters, with two
reviewer layers (p.6). The judge was GPT-4o, with Claude 3.5 Sonnet also tried.

**Findings**
- **Why a rubric.** A judge given the full conversation "yields low alignment with human raters"
  (p.4). Each item therefore gets a human-written YES/NO question that "requires only the final model
  response as context" (p.5).
- **Agreement.** The rubric judge scores 93.95% overall and 92.26–94.85% per axis. The full-context
  judge scores 37.33% (Table 4, p.7).
- **Rankings.** Judge and human scores rank all six models identically (Tables 2–3).
- **Judge model.** Claude gave "the same conclusions" as GPT-4o (p.6–7).

**Caveats**
- Agreement is raw percent, with no κ. Humans passed about 23% of responses, so always-FAIL already
  scores about 77% *(our arithmetic from Table 2)*.
- Items whose rubric was too hard for a frontier judge were excluded from the release (p.9).
- The full-context baseline also lacked the rubric, so the paper does not show whether the extra
  context or the missing rubric caused the drop.

---

## 5. Further reading — 2026 top venues, not downloaded

Venues were confirmed as in §4. The summaries come from abstracts and the search pass, not from a
full reading.

- **Messing, *Hidden Measurement Error in LLM Pipelines*** — NeurIPS 2026,
  [arXiv:2604.11581](https://arxiv.org/abs/2604.11581). Standard errors that ignore the choice of
  judge, prompt and replicate noise are 40–60% too small.
- **Kuai et al., *A Statistical Framework for Auditing Behavioral Dependence and Induced Bias in LLM
  Judges*** — COLM 2026, [arXiv:2604.07650](https://arxiv.org/abs/2604.07650). Models that share
  failure modes are over-endorsed by the judge, even across families.
- **Xu et al., *Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based
  LLM-as-a-Judge*** — Findings of EMNLP 2026, [arXiv:2602.02219](https://arxiv.org/abs/2602.02219).
  Judges favour score options by position. The order of criteria in one prompt shifts the scores,
  which supports one criterion per call.
- **Fujinuma, *Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge*** — Findings of ACL
  2026, [arXiv:2510.18196](https://arxiv.org/abs/2510.18196). Relabelling the same scale (0–4, 1–5,
  2–6) changes correlation with humans. That is a reason to keep Wonderbread's 1–3 labels verbatim.
- **Mukherjee et al., *The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human
  Alignment*** — Findings of EMNLP 2026, [arXiv:2606.03043](https://arxiv.org/abs/2606.03043). Score
  judge and held-out human against the same reference, and report the spread of scores.
- **Gao et al., *Which Metrics Save the Most Human Annotation?*** — EMNLP 2026,
  [arXiv:2608.26638](https://arxiv.org/abs/2608.26638). Choose a judge by the human labels it saves
  inside PPI. Paired designs beat unpaired ones.
- **Mani et al., *No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference*** — ICML
  2026, [arXiv:2505.20178](https://arxiv.org/abs/2505.20178). PPI++ beats human-only estimation only
  above a judge–human correlation threshold set by n.
- **Zhou et al., *RubricBench*** — ACL 2026, [arXiv:2603.01562](https://arxiv.org/abs/2603.01562).
  Human-written rubrics beat judge-generated ones by about 27 points.
- **Rao & Callison-Burch, *Autorubric*** — COLM 2026,
  [arXiv:2603.00077](https://arxiv.org/abs/2603.00077). Few-shot and option-shuffling effects differ
  by judge family, so check Wonderbread's few-shots per candidate.
- **Hong et al., *From Rubrics to Reliable Scores (Rulers)*** — EMNLP 2026,
  [arXiv:2601.08654](https://arxiv.org/abs/2601.08654). Locked rubrics, quoted evidence before the
  verdict, then calibration to human score boundaries.
- **Chen et al., *Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation*** — EMNLP 2026,
  [arXiv:2606.01629](https://arxiv.org/abs/2606.01629). A reference answer helped more consistently
  than a rubric. Kimi-K2.6 and GLM-5.1 are among the judges.
- **Li et al., *WebDevJudge*** — ICLR 2026 (oral), [arXiv:2510.18560](https://arxiv.org/abs/2510.18560).
  Judges miss functional equivalence between differently worded outputs. In SOP matching that becomes
  false −1s.
- **Zhang et al., *Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for
  LLM-as-a-Judge*** — ICML 2026, [arXiv:2605.10805](https://arxiv.org/abs/2605.10805). Thinking mode
  helps on maths and code, is limited elsewhere, and costs 3.4–11.2× the tokens.
- **Lin et al., *CUARewardBench*** — ICML 2026, [arXiv:2510.18596](https://arxiv.org/abs/2510.18596).
  Step-level judging is much harder than trajectory-level judging. Prompt templates trade precision
  against recall.

---

## 6. Dropped from the previous version

Their full extractions are in commit `b6bfe59`. Each was dropped for its venue, not its content.

| Paper | Why dropped |
|---|---|
| Wei et al., *RocketEval* | ICLR **2025** |
| Starace et al., *PaperBench* | ICML **2025** |
| Guerdan et al., *Rating Indeterminacy* | NeurIPS **2025**. Li J. (#14) is the 2026 work on the same validation question |
| Soumik, *Bias Mitigation Strategies* | TMLR 2026 — a journal, not a conference |
| Norman et al., *Reliability without Validity* | Still a preprint on 2026-10-08. It was the source of the Kimi K2.5 numbers, now replaced by Peng's K2.6 |
| Gu et al., *A Survey on LLM-as-a-Judge* | First released on arXiv in **2024** (2411.15594). The PDF stays at [LLM_as_judge.pdf](LLM_as_judge.pdf), which [GPT_LLM_AS_JUDGE_GUIDE.md](GPT_LLM_AS_JUDGE_GUIDE.md) cites |

---

## 7. Files

Licences were checked on each paper's arXiv abstract page or in the ACL Anthology (CC BY 4.0 for
2016 onward), 2026-10-08.

| PDF | Licence | In git |
|---|---|---|
| [Audio MultiChallenge](papers/2025_Gosai_Audio-MultiChallenge_ACL2026.pdf) | CC BY 4.0 (ACL Anthology version) | yes |
| [IF-RewardBench](papers/2026_Wen_IF-RewardBench_ACL2026.pdf) | CC BY 4.0 (ACL Anthology version; the arXiv copy is non-exclusive) | yes |
| [Self-Preference in Rubric Evaluation](papers/2026_Pombal_Self-Preference-in-Rubric-Evaluation_COLM2026.pdf) | CC BY 4.0 | yes |
| [Preference Leakage](papers/2025_Li_Preference-Leakage_ICLR2026.pdf) | CC BY 4.0 | yes |
| [Self-Preference Sanity Checks](papers/2026_Roytburg_Self-Preference-Sanity-Checks_ICML2026.pdf) | CC BY 4.0 | yes |
| [Frontier Models for Proof Verification](papers/2026_Naik_Frontier-Models-for-Proof-Verification_COLM2026.pdf) | CC BY 4.0 | yes |
| [No Free Labels](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) | CC BY-NC-SA 4.0 | yes |
| [ResearchRubrics](papers/2025_Sharma_ResearchRubrics_ICLR2026.pdf) | CC BY 4.0 | yes |
| `papers/2026_Peng_RuVerBench_EMNLP2026.pdf` | arXiv non-exclusive | **no** — local only, via [papers/.gitignore](papers/.gitignore) |
| [PoSh](papers/2025_Ananthram_PoSh_ICLR2026.pdf) | CC BY 4.0 | yes |
| [How2Everything](papers/2026_Chang_How2Everything_ICML2026.pdf) | CC BY 4.0 | yes |
| [Correctly Report LLM-as-a-Judge](papers/2025_Lee_Correctly-Report-LLM-Judge_ICML2026.pdf) | CC BY-SA 4.0 | yes |
| [Noisy-Judge Inference](papers/2026_Chen_Noisy-Judge-Inference_ICML2026.pdf) | CC BY 4.0 | yes |
| [Sparse-Overlap Validation](papers/2026_Li_Sparse-Overlap-Validation_NeurIPS2026.pdf) | CC BY-NC-SA 4.0 | yes |
| [Coin Flip for Safety](papers/2026_Schwinn_Coin-Flip-for-Safety_ICML2026.pdf) | CC BY 4.0 | yes |
| [MultiChallenge](papers/2025_Sirdeshmukh_MultiChallenge_ACLFindings2025.pdf) | CC BY 4.0 (ACL Anthology version) | yes |

The filename year is the year of first release; the venue follows it. A fresh clone lacks the
RuVerBench PDF: fetch it from arXiv.
