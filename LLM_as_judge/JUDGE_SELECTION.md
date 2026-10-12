# Judge selection — which benchmarks need an LLM judge, what the literature says, and which model we use

**Status, 2026-10-12.**
- **The provisional default judge is Claude Haiku 4.5.**
- **The final choice is made by a bake-off on our own human-labelled data.** That bake-off has not
  run yet, and the human labels it needs have not been collected.
- **This file is the argument, start to finish.** The page-cited evidence, the full per-paper
  extractions and the per-benchmark pipelines are in [README.md](README.md). Where a number here
  and there disagree, README.md is authoritative.

## Summary

1. **Two of the suite's ten benchmarks need an LLM judge:** MultiChallenge and Wonderbread. Between
   them they have three judged tasks. Every judge model they shipped with is retired, so a
   substitute is forced.
2. **The substitute must not share a family with any model we evaluate.** For MultiChallenge it
   must not be lenient; for SOP Generation it must not be over-strict; and it must be cheap enough
   for SOP Generation's call volume.
3. **Fifteen papers from 2026 top venues** show three things:
   - A light judge is viable for narrow, one-item-per-call questions like ours.
   - Its main risk is passing responses that should fail.
   - Which way a judge errs depends on the judge, so it has to be measured, not assumed.
4. **Four out-of-family candidates go to a bake-off:** Claude Haiku 4.5, Claude Sonnet 5.5, Kimi
   K2.6 and Claude Haiku 5.5. We take the cheapest one that passes a bar fixed in advance, and
   prefer one judge for all three tasks.
5. **Reported scores will be corrected for the judge's measured error rates,** with confidence
   intervals, and declared as not comparable to the original papers' numbers.

---

## 1. Which benchmarks need an LLM judge

The rule ([JUDGE_RECORD.md](JUDGE_RECORD.md)): a benchmark needs a judge record wherever **a model's
opinion enters the score**. Then the number cannot be recomputed from the dataset and the responses
alone.

**Seven of the ten need no judge.** Each is scored against a key or by a symbolic checker:

| Benchmark | How it is scored |
|---|---|
| BIG-Bench Hard | Exact match |
| MMLU | Accuracy |
| DocVQA | ANLS |
| EmoBench | MCQ accuracy |
| NegotiationToM | Exact match and F1 |
| PlanBench | The VAL/PDDL validator |
| Multi-party Goal Tracking | Exact match, with the residue reviewed by a human, not a model |

**Three shipped with a judge.** One of them has since been dropped:

| Benchmark | Judged part | Status |
|---|---|---|
| **MultiChallenge** | All 273 items | **Kept** |
| **Wonderbread** | Question Answering and SOP Generation | **Kept**. A third judged subtask, SOP Improvement, has a scorer that never executes, so it is out of scope |
| AwareBench | 60 of 4,075 rows | **Dropped 2026-10-05** by the user. The 60 rows have no per-item reference or rubric. The paper's own scores for them differ by 28 points between its two evaluator prompts ([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6) |

That leaves **three judged tasks**:

| Task | What the judge decides | Calls per evaluated model | Reference given to the judge | Original judge |
|---|---|---:|---|---|
| **MultiChallenge** | YES/NO: does the final response satisfy one human-written rubric question? | 273 | The rubric only; no gold answer exists | `gpt-4o-2024-08-06` |
| **Wonderbread QA** | A 1–3 score (1 = best) on each of four criteria: completeness, soundness, clarity, compactness | 480 | A human answer, for completeness and soundness only | `gpt-4-0125-preview` |
| **Wonderbread SOP Generation** | For each line of one procedure, the index of the matching line in the other, or −1, in both directions. Precision, recall and ordering are tallies of these decisions | ~3,100 | The human-written gold SOP | `gpt-4-1106-preview` |

**Substitution is forced.** All of these snapshots are retired ([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md)
§6, question 2), so an exact reproduction is impossible at any price. A substituted judge must be
declared. Its numbers are not comparable to the published ones and are never tabulated beside them.

## 2. What the replacement judge has to get right

These requirements follow from the three tasks, before any paper is read.

| # | Requirement | Why |
|---|---|---|
| R1 | **No family in common with any evaluated model** | A judge that grades its own family's outputs can favour them. Our models come from six vendor slots, listed in [PLAN.md](../PLAN.md) |
| R2 | **Few false PASSes on MultiChallenge** | Humans pass only about 23% of responses. A lenient judge inflates every score, and raw agreement hides it: an always-FAIL judge already agrees with humans about 77% of the time |
| R3 | **No under-matching on SOP Generation** | A judge that refuses valid matches lowers both precision and recall |
| R4 | **Cheap at SOP Generation's volume** | About 3,100 calls per model. The user prefers a light judge if the evidence allows one |
| R5 | **Stable and declared** | A pinned snapshot, repeated runs to measure stability, and every setting logged |

R2 and R3 pull in opposite directions: MultiChallenge is safest with a strict judge, SOP Generation
with one that is not over-strict. That is why the judge's error direction has to be measured
separately on each task.

## 3. The literature

**Selection.**
- Only papers accepted at a 2026 top venue: ICLR, ICML, ACL, EMNLP, NeurIPS or COLM. An arXiv first
  release in 2025 is allowed; 2024 is not.
- Each acceptance was confirmed on an official source: the conference's accepted-paper data, the
  ACL Anthology, or EMNLP 2026's programme sheet.
- Fifteen papers were kept: five from ICML, three each from ICLR and COLM, two from ACL, and one
  each from NeurIPS and EMNLP. Each was read in full.
- The benchmark's own paper, MultiChallenge (Findings of ACL 2025), is used as a primary source.

The fifteen answer three questions:

| Question | Papers |
|---|---|
| Which judge model? | Gosai (*Audio MultiChallenge*), Wen (*IF-RewardBench*), Pombal, Li (*Preference Leakage*), Roytburg, Naik, Krumdick |
| How should the judging call be posed? | Sharma (*ResearchRubrics*), Peng (*RuVerBench*), Ananthram (*PoSh*), Chang (*How2Everything*) |
| How should the judge be validated, and its scores reported? | Lee, Chen, Li J. (*Sparse Overlap*), Schwinn |

Full titles, venues and arXiv links are in §8.

## 4. What the literature established

Each finding below gives what the paper found, then what it changes for us. Numbers are the
papers'. Our own reasoning is marked *(our inference)*.

**F1 — Judges favour their own family, so family exclusion is a sound precaution.**
- **Self-preference survives objective rubrics.** On fully objective rubric items, judges were more
  than 50% more likely to pass a failed item when the output was their own, and the effect extends
  to the judge's family (Pombal).
- **Shared training data matters more than shared family.** Judges favoured models trained on
  their outputs: "inheritance" leaked nearly as much as being the same model, and more than being
  in the same family (Li).
- **But part of the measured effect is weakness.** Once the items a judge itself gets wrong are
  controlled for, only 51% of earlier self-preference findings stay significant (Roytburg).

*Consequence:* exclude every evaluated family (R1). The exclusion costs little, because strong
out-of-family judges exist. Also check, per evaluated model, whether a judge is softer on some
models' outputs than on others.

**F2 — Narrow, one-item-per-call questions are where light judges hold up.**
- **Small judges hold up on final answers.** A 9B judge was 6.6 points behind the best on final
  answers, but 17.8 behind on long coding trajectories (Peng).
- **Per-unit questions beat holistic frontier judging.** A 14B judge asking one question per unit
  beat holistic GPT-4o (Ananthram).
- **A checklist removes the multi-turn penalty.** Judges given a per-constraint checklist lost
  nothing on multi-turn items (Wen).
- **A correct reference can let a small judge beat a large one.** A 7B judge with a human reference
  beat GPT-4o without one (Krumdick).
- **The rubric beat the whole conversation by a wide margin.** MultiChallenge's own rubric judge
  agreed with humans 93.95% of the time. A judge given the whole conversation without a rubric
  agreed 37.33% (Sirdeshmukh).

*Consequence:* all three of our tasks already have this shape, so a light judge is defensible (R4).
Keep one item per call: putting 4–5 rubric items in one call cost 9–35 points on long inputs
(Peng).

**F3 — A light judge's main risk is passing what should fail.**
- **Small judges accept more flawed answers.** Small judges accepted flawed proofs 13.3 points more
  often than frontier judges, but rejected correct ones only 1.1 points more often (Naik).
- **Every judge catches violations less well than it confirms compliance.** Across 14 judges, F1 on
  the "violated" class ranged 0.15–0.68, against 0.75–0.91 on the "followed" class; humans scored
  0.744 on "violated" (Wen).
- **Haiku is more lenient than Sonnet.** On HealthBench's failed rubric items, Claude Haiku passed
  0.12 of them and Claude Sonnet 0.03 (Pombal).

*Consequence:* the bake-off ranks judges on their false-PASS rate, not on agreement (R2).

**F4 — Which way a judge errs depends on the judge, not just its size.**
- **Agreement hides strictness.** On reference-anchored procedure checking, Claude Opus 4.5 had
  only 6.1% false PASS but 38.2% false FAIL. Its pass rate was 34.5%, against 51% for humans. A
  6.5-point spread in agreement across judges hid a 26-point spread in pass rates (Chang).
- **Strictness can flip between domains.** One judge can be strict on one domain and permissive on
  another (Peng).

*Consequence:* measure both error directions on every task. A judge that is safe for MultiChallenge
can be harmful for SOP Generation (R2 vs R3).

**F5 — Wording changes results, so prompts stay verbatim.**
- **Small wording changes move scores.** A one-line strict/flexible suffix moved accuracy by −9.2 to
  +11.8 points (Peng). Swapping the prompt moved balanced accuracy by 27 points (Naik).
- **Rewritten rubrics do worse.** Rubric text expanded by an LLM lowered agreement with humans by
  15–22 points (Sharma).

*Consequence:* keep every upstream prompt, rubric and reference verbatim, so the judge model is the
only thing that changes. Never let an LLM rewrite them.

**F6 — Validation needs several raters on shared items.**
- **Little overlap picks the wrong judge.** When only 5% of items were labelled by more than one
  human, the validation reached the wrong verdict 25% of the time. It picked the wrong best of ten
  judges 65% of the time (Li J.).
- **The recipe.** At least 25% overlap, three to five raters, and overlap items chosen by
  stratifying on the first rater's label. At our scale of ~200 items, 100 double-labelled items give
  85–87% reliable verdicts; 50 give only 53–61%.
- **Reliable labelling needs a codebook.** Human agreement on judging procedures rose from α 0.273
  to 0.593 only after an iterated codebook and a qualification test (Chang).

*Consequence:* per benchmark, one rater labels ~200 items and two more label a shared panel of ~100.
Write and pilot a codebook before labelling.

**F7 — A raw judge pass rate is biased; correct it.**
- **The correction.** Measure the judge's specificity and sensitivity against human labels. Then
  report the Rogan–Gladen corrected pass rate, with an interval that carries the calibration
  uncertainty (Lee).
- **The per-model check.** When calibration items are a random sample of the same responses, the
  PPI++ estimator is more efficient (Chen).
- **Do not expect the correction to beat human labels.** In Lee's analysis, a judge plus calibration
  beats human-only labels only when the judge is quite accurate. At MultiChallenge's ~23% pass rate
  that means q ≳ 0.88 *(our arithmetic)*. None of seven current judges reached that region, Claude
  Haiku 4.5, Sonnet 4.6 and Opus 4.6 included (Lee).

*Consequence:* report the corrected rate and interval as the headline, with the raw judge rate
labelled as raw.

**F8 — Judge accuracy shifts with the model being judged.**
- **Which model wrote the output matters.** It moved judge accuracy more than which attack produced
  it (Schwinn). Judge rankings also changed with the generating model (Peng).
- **Agreement among judges is not correctness.** In Schwinn's study, judges were nearly unanimous
  and still wrong.

*Consequence:* draw the validation set evenly across evaluated models, and report judge error per
model.

## 5. From the findings to the candidates

**Excluded (F1):** every model from the six vendor slots.
- **The families:** Google (Gemini, Gemma), OpenAI (including GPT-oss), Alibaba Qwen, DeepSeek and
  xAI. xAI's removal from the reported results is provisional, so it is still treated as evaluated.
- **What this removes:** the top judge in several of the papers — o4-mini, Gemini-3-Pro, GPT-5.4.
- **Kimi and Llama stay eligible, on one condition.** Each has been run on other benchmarks in this
  suite (Big Bench Hard; Llama also on MMLU), but neither is one of the six. They stay eligible only
  while neither is run on MultiChallenge or Wonderbread.

**The remaining models, ranked by evidence.** The ranking weighs four things:
- how many of the fifteen papers tested the model as a judge;
- how close those tasks are to ours;
- whether the exact version we would call was tested;
- whether the model is light.

| Tier | Model | Papers | Assessment |
|---|---|---:|---|
| **A — bake-off** | 1 · Claude Haiku 4.5 | 3 | The only direct evidence on a MultiChallenge-style task: κ 0.808 against human graders, level with GPT-4.1 (Gosai). Its known weakness is leniency (F3) |
| | 2 · Claude Sonnet 5.5 | 5, on versions 4.5/4.6 | The most papers, and the strictest Claude judge. Version 5.5 itself is untested |
| | 3 · Kimi K2.6 | 1 | 2nd of 18 judges on rubric verification, level with Claude Opus 4.7 (Peng). Cheap; one paper only |
| | 4 · Claude Haiku 5.5 | 0 | Released 2026-10-07 and untested. Included only because it costs a tenth of Haiku 4.5 |
| **B — not recommended** | Claude Opus | 4 | Strong but not light, and fail-biased on procedures (F4). Useful as a reference |
| | GLM | 2 | Inconsistent, and weak at catching violations (Wen) |
| | Mistral Medium | 1 | Below Haiku 4.5 on the same task (Gosai) |
| **C — evidence against** | Llama | 5 | Consistently weak. Strong only when given a human reference (Krumdick) |

MiniMax and Cohere have no 2026 top-venue evidence. **No paper compares all four Tier A
candidates on the same task**, which is why the bake-off is needed.

## 6. The decision

### 6.1 Default and shortlist

| Role | Model | Price in / out per M tokens | Cost per full pass, all three tasks, five models |
|---|---|---|---:|
| **Default light judge (provisional)** | Claude Haiku 4.5 | $1 / $5 | ~$21 |
| Cheapest candidate, untested | Claude Haiku 5.5 | $0.10 / $0.50 | ~$2 |
| Cheap challenger | Kimi K2.6 | $0.44 / $2.45 | ~$9 |
| Escalation if the light judges fail | Claude Sonnet 5.5 | $2 / $10 | ~$41 |

- **Where the prices come from:** OpenRouter list prices, 2026-10-08, without reasoning tokens.
- **What costs the most:** human labelling (~400 labels per benchmark), not the API calls.
- **A settings difference:** Haiku 5.5 and Sonnet 5.5 accept no `temperature` setting on OpenRouter.
  For them, stability comes from repeated runs, not from temperature 0.

**Disclosure.** This analysis was written with Claude (Anthropic), and three of the four candidates
are Claude models. They are candidates because of the family-exclusion rule and the evidence above.
The bake-off decides.

### 6.2 How the final choice is made

1. **Five configurations** run on the same human-labelled items:
   - Haiku 4.5 without reasoning;
   - Haiku 4.5 with a small reasoning budget;
   - Haiku 5.5;
   - Kimi K2.6;
   - Sonnet 5.5.
2. **Validation sets** (F6, F8), drawn evenly across evaluated models:

   | Task | Primary rater | Two more raters |
   |---|---|---|
   | MultiChallenge | ~200 items | ~100 of them |
   | Wonderbread QA | the existing 30-item human-vs-GPT-4 sample, plus ~100 new items | 50 of the new items |
   | SOP Generation | every line decision, both directions, for ~12 demos | half the demos |

3. **Acceptance bars, fixed before any result is seen.** All are proposals, to adjust before
   running.
   - *MultiChallenge:*
     - specificity (1 − false PASS) ≥ 0.90 and sensitivity ≥ 0.85, each with an interval;
     - the leave-one-human-out test passed with three raters;
     - no evaluated model's false-PASS rate above 1.5× the lowest.
   - *Wonderbread QA:*
     - per criterion, agreement on the 30-item sample no more than 5 points below the original
       GPT-4 judge's;
     - the leave-one-human-out test passed on the new items.
     - A criterion where no judge passes is human-scored or reported as unvalidated, never given a
       lower bar. Clarity and compactness, which have no reference, are the likely cases.
   - *SOP Generation:* set before running, from false-match and miss rates in each direction. No
     published figure exists for line-level matching.
4. **Selection rule.** Among the configurations that pass, rank on the same items by false-PASS
   rate and Youden's J. Take the cheapest whose numbers are not meaningfully worse than the best.
5. **One judge for all three tasks — proposed 2026-10-12.**
   - *Within a benchmark, one judge for every evaluated model:* required.
   - *Across the three tasks:* preferred. Take the cheapest candidate that passes all three bars.
     Only a task where it fails gets its own judge, declared in that benchmark's record.

### 6.3 How results will be reported

- **Corrected pass rate as the headline.** Per evaluated model, the Rogan–Gladen corrected pass rate
  with its interval; the raw judge rate beside it, labelled raw; and a per-model PPI++ estimate as a
  check (F7).
- **The judge's error rates are published.** Specificity, sensitivity and agreement with humans, per
  task and per evaluated model.
- **Everything is declared.** The judge's model id, snapshot, settings and number of replicates go
  into the record required by [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md). Every
  number is marked as not comparable to the original paper's.

## 7. What is still open

- **The bake-off and the human labels.** Neither exists yet. Until both do, Haiku 4.5 is a
  provisional default, not a result.
- **Haiku 5.5 is untested anywhere.** No published evaluation of it exists.
- **Kimi K2.6 has one supporting paper,** and none on a short YES/NO rubric like MultiChallenge's.
- **MultiChallenge task version.** In March 2026 the authors revised 54 tasks and moved their own
  judge to Gemini 2.5 Pro. Whether we use the revised tasks or our vendored February-2025 set is the
  user's decision. Either way the Gemini judge is excluded.
- **Wonderbread's demo count for SOP Generation** depends on text-only vs multimodal scope
  ([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6, question 4). The judge itself is text-only either way.
- **Two Wonderbread QA criteria may not be judgeable.** If clarity and compactness fail for every
  candidate, they will be human-scored or reported as unvalidated.

## 8. Papers

Page-cited extractions of each are in [README.md](README.md) §4. The arXiv identifier is given; the
venue was confirmed as described in §3.

| # | Paper | Venue | arXiv |
|---|---|---|---|
| 1 | Gosai et al., *Audio MultiChallenge* | ACL 2026 | [2512.14865](https://arxiv.org/abs/2512.14865) |
| 2 | Wen et al., *IF-RewardBench* | ACL 2026 | [2603.04738](https://arxiv.org/abs/2603.04738) |
| 3 | Pombal et al., *Self-Preference Bias in Rubric-Based Evaluation of LLMs* | COLM 2026 | [2604.06996](https://arxiv.org/abs/2604.06996) |
| 4 | Li et al., *Preference Leakage: A Contamination Problem in LLM-as-a-judge* | ICLR 2026 | [2502.01534](https://arxiv.org/abs/2502.01534) |
| 5 | Roytburg et al., *Are LLM Evaluators Really Narcissists?* | ICML 2026 | [2601.22548](https://arxiv.org/abs/2601.22548) |
| 6 | Naik et al., *Do We Need Frontier Models to Verify Mathematical Proofs?* | COLM 2026 | [2604.02450](https://arxiv.org/abs/2604.02450) |
| 7 | Krumdick et al., *No Free Labels* | COLM 2026 | [2503.05061](https://arxiv.org/abs/2503.05061) |
| 8 | Sharma et al., *ResearchRubrics* | ICLR 2026 | [2511.07685](https://arxiv.org/abs/2511.07685) |
| 9 | Peng et al., *Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?* | EMNLP 2026 | [2606.29920](https://arxiv.org/abs/2606.29920) |
| 10 | Ananthram et al., *PoSh* | ICLR 2026 | [2510.19060](https://arxiv.org/abs/2510.19060) |
| 11 | Chang et al., *How2Everything* | ICML 2026 | [2602.08808](https://arxiv.org/abs/2602.08808) |
| 12 | Lee et al., *How to Correctly Report LLM-as-a-Judge Evaluations* | ICML 2026 | [2511.21140](https://arxiv.org/abs/2511.21140) |
| 13 | Chen et al., *Efficient Inference for Noisy LLM-as-a-Judge Evaluation* | ICML 2026 | [2601.05420](https://arxiv.org/abs/2601.05420) |
| 14 | Li, Mukherjee & Pal, *LLM Judge Validation Under Sparse Overlap* | NeurIPS 2026 | [2609.31857](https://arxiv.org/abs/2609.31857) |
| 15 | Schwinn et al., *A Coin Flip for Safety* | ICML 2026 | [2603.06594](https://arxiv.org/abs/2603.06594) |
| — | Sirdeshmukh et al., *MultiChallenge* (the benchmark's own paper) | Findings of ACL 2025 | [2501.17399](https://arxiv.org/abs/2501.17399) |
