# LLM-as-a-judge: choosing the judge model and building the pipeline

A literature review of ten papers from 2024–2026, written to answer two questions before we replace
the judges in our three judged benchmarks:

1. **How should the judge model be chosen?** Should we pick it to suit the benchmark's task, take the
   top frontier model, or do something else — a fine-tuned judge, a panel, a judge from outside the
   evaluated families?
2. **How should the judging pipeline be built?** This covers the prompt, references, output format,
   decoding, bias controls, and validation against humans.

Why the question is live: every judge model the three benchmarks shipped with has been retired, so a
substitution is forced ([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md) §6, question 2). What each judge is
shown and how it scores is recorded in [JUDGE_RECORD.md](JUDGE_RECORD.md). The fields any new judge
must document are in [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md). API-level
mechanics for a GPT judge are in [GPT_LLM_AS_JUDGE_GUIDE.md](GPT_LLM_AS_JUDGE_GUIDE.md). This file
covers only what the literature says.

**How this was produced (2026-10-05).**
- **Selection.** Candidates were chosen for relevance to the two questions and checked on arXiv and
  Semantic Scholar. Google Scholar indexes all of them through arXiv, but they were not looked up
  there one by one.
- **Reading.** Every PDF was read in full. Each number below carries the page it came from, and the
  headline numbers were re-checked against the PDF text.
- **Page numbers** are PDF pages. The one exception is Gu et al., which uses the journal's printed
  numbers; its PDF page is the printed page + 1.
- **Inference.** Anything that is our inference and not a paper's claim is marked *(our inference)*.

---

## Short answers

**Q1 — choosing the judge.** None of the ten papers supports "just take the top model", and none
supports a fine-tuned judge model. What they support is a procedure:

1. **Shortlist** strong general models. Reasoning models belong on the list when the criterion is
   correctness.
2. **Exclude** any model from the same family as a model being judged.
3. **Validate** each candidate on a human-labelled sample of *each* benchmark, using a
   chance-corrected agreement metric.
4. **Pick** the best on that measurement, or pool the best two or three as a cross-family panel.

Three findings drive this:
- **Judge quality is task-specific.** Bavaresco; Norman: "a poor estimator of its ranking on others".
- **Judges favour their own outputs.** Panickssery; Verga; Ye.
- **A judge is reliable only where it can solve the item itself, or is given a correct reference.**
  Krumdick; Tan.

**Q2 — building the pipeline.**
1. Give the judge a *verified* reference or per-item rubric wherever one exists, and never one the
   judge wrote itself.
2. Build the prompt from the human annotation guidelines, with explicit rules for edge cases.
3. Hide which model wrote the answer.
4. Constrain the output to one label or score, and log invalid outputs instead of imputing them.
5. Swap or shuffle any ordered inputs.
6. Use temperature 0 with at least 3 replicate runs.
7. Validate against human labels, reporting raw agreement **and** κ / Scott's π / Krippendorff's α.
8. Freeze and record the judge, prompt and decoding settings.

The procedure is set out as numbered steps in [§4](#4-synthesis--q2-building-the-pipeline).

---

## 1. The ten papers at a glance

Citation counts are from Semantic Scholar, 2026-10-05.

| # | Paper | Venue | Cites | Answers | One-line takeaway |
|---|---|---|---:|---|---|
| 1 | Thakur et al., *Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges* | GEM² workshop @ ACL 2025 | 273 | Q1, Q2 | Only the largest judges approach human agreement. Percent agreement hides weak judges; use Scott's π. |
| 2 | Bavaresco et al., *LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks* | ACL 2025 | 363 | Q1 | No judge is best everywhere. Validate against task-specific human labels. |
| 3 | Huang et al., *An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4* | Findings of ACL 2025 | 131 | Q1 | Fine-tuned judges are task-specific classifiers and collapse off their home turf. |
| 4 | Tan et al., *JudgeBench: A Benchmark for Evaluating LLM-based Judges* | ICLR 2025 | 369 | Q1, Q2 | On hard correctness pairs, reasoning models win. A judge's accuracy tracks whether it can solve the task itself. |
| 5 | Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations* | NeurIPS 2024 | 859 | Q1 | Self-recognition correlates linearly with self-preference. Don't let a model judge itself. |
| 6 | Verga et al., *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models* (PoLL) | arXiv preprint, 2024 | 368 | Q1 | A panel of three small models from three families beats GPT-4 alone at 1/7 the cost. |
| 7 | Gu et al., *A Survey on LLM-as-a-Judge* | The Innovation 7(6), 2026 | 1,891 | Q2 | A four-stage framework for building a judge, plus a meta-evaluation of common improvement strategies. |
| 8 | Ye et al., *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge* (CALM) | ICLR 2025 | 440 | Q2 | 12 bias types. Telling the judge to avoid a bias does not remove it. |
| 9 | Krumdick et al., *No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding* | COLM 2026 | 65 | Q1, Q2 | Without a correct reference, a judge is reliable only on questions it can answer itself. |
| 10 | Norman et al., *Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias* | arXiv preprint, June 2026 | — | Q1, Q2 | 21 current judges and ~541k judgments. Gives a five-step minimum validation protocol. |

**Venue notes.**
- **#2, #3:** the venue comes from the arXiv comment.
- **#5, #8:** the venue comes from Semantic Scholar. The PDFs here are arXiv v1 and v2, which print no
  venue. Numbers are from these versions.
- **#9:** COLM 2026 is printed in v4 of the PDF.
- **#10:** too new to have a citation count.

---

## 2. Paper by paper

### Part A — choosing the judge

#### 1 · Thakur et al. (2025) — *Judging the Judges* · [PDF](papers/2024_Thakur_Judging-the-Judges_GEM2025.pdf) · [arXiv:2406.12624](https://arxiv.org/abs/2406.12624)

**Setup**
- **Judges:** 13 in total. 11 are LLMs: Llama-2/3/3.1 at several sizes, Gemma 2B, Mistral 7B,
  JudgeLM-7B (a fine-tuned judge) and GPT-4 Turbo. The other 2 are lexical baselines: exact match and
  "contains".
- **Exam-takers:** 9 models answering 400 TriviaQA questions.
- **Task:** the judge sees the question, the reference answers and the response, and outputs
  "correct" or "incorrect". Every answer was also labelled by humans.

**Findings**
- **Human agreement is near perfect:** Scott's π = 96.2 (p.5).
- **The best judges are big:** Llama-3 70B (88), Llama-3.1 70B (88) and GPT-4 Turbo (87) (Fig. 1b,
  p.2). In the abstract's words, "only the best (and largest) models show reasonable alignment with
  humans, though they still differ with up to 5 points from human-assigned scores".
- **Percent agreement hides weak judges.**
  - Llama-3 8B has more than 80% agreement but π = 59 (p.5).
  - Judges above 90% agreement "may still differ more than 10 points in their assigned score" (p.6).
- **Ranking is easier than scoring.** Even "contains" ranks the exam-takers at Spearman 0.99
  (Table 8, p.20). The paper's conclusion: "identifying which models are better should not be equated
  to assigning them the correct score" (p.6).
- **Weak spot — under-specified answers.** Recall is 33.9% for GPT-4 and 23.3% for Llama-3 70B
  (Table 2, p.7).
- **Leniency.** The share of answers marked correct runs above 0.5: GPT-4 0.69, Llama-3 70B 0.90
  (p.27). Some judges accept dummy answers such as "Yes" (p.7–8).
- **Fine-tuned judge.** JudgeLM-7B reaches only π = 65.

**Implication**
- **Q1:** use the largest, most capable general model when absolute scores matter — and we report
  absolute scores.
- **Q2:**
  - "We recommend computing both percent agreement and Scott's π, paired with qualitative analysis"
    (p.8).
  - Write edge-case rules into the prompt (App. G): under-specified counts as incorrect, extra
    correct detail counts as correct.
  - Run dummy-answer sanity checks.
  - Test whether the order of the references changes the verdict.

**Caveat:** short factual QA with very high human agreement. Real judging is harder, as the authors
note (App. A), and the models are from 2023–24.

#### 2 · Bavaresco et al. (2025) — *LLMs instead of Human Judges?* · [PDF](papers/2024_Bavaresco_LLMs-instead-of-Human-Judges_ACL2025.pdf) · [arXiv:2406.18403](https://arxiv.org/abs/2406.18403)

**Setup**
- **Judges:** 11, including GPT-4o, Gemini-1.5, Llama-3.1 8B/70B, Mixtral and Command R(+).
- **Data:** JUDGE-BENCH — 20 human-annotated datasets and more than 70k items, spanning reasoning,
  planning, toxicity/safety, dialogue, MT, summarisation and instruction following.
- **Metrics:** Cohen's κ for categorical labels, Spearman ρ for graded ones.

**Findings**
- **No single best judge.** "no single model demonstrates a clear superiority over others across all
  properties; instead, different quality dimensions are better assessed by different models" (p.5).
- **The open–closed gap is small.** Mean κ is 0.28 for GPT-4o and 0.28 for Llama-3.1-70B
  (Table 1, p.3). Open models beat GPT-4o on some datasets.
- **Performance depends on the task.**
  - Instruction following is judged well: LLMBar-natural κ = 0.84 for GPT-4o.
  - Toxicity and safety are judged worst: DICES-990 κ = −0.24 for GPT-4o (p.3–4). The authors partly
    blame guardrails.
- **Machine text is harder to judge.** Every judge agrees with humans better on human-written text
  than on machine-generated text (p.5).
- **Prompting tricks are not reliable fixes.** CoT, few-shot and paraphrased prompts "do not
  consistently improve agreement" (p.5). CoT helped on LLMBar-adversarial and hurt on DICES.

**Implication**
- **Q1:** choose by measured agreement on *this* task, not by reputation. "we recommend validating
  LLM judges against task-specific human annotations before deploying them for any particular task"
  (p.2, p.5).
- **Q2:**
  - Use the original human annotation guidelines as the judge prompt.
  - Constrain the output to a single label, decoded greedily.
  - Compare against a human upper bound computed from inter-annotator agreement.

**Caveat:** pointwise only, with 2024 judges. Invalid outputs were replaced by a *random* label,
which is a choice to avoid.

#### 3 · Huang et al. (2025) — *Fine-tuned Judge Model is not a General Substitute for GPT-4* · [PDF](papers/2024_Huang_Fine-tuned-Judge-not-Substitute-for-GPT4_ACLFindings2025.pdf) (local only, see [Files](#7-files)) · [arXiv:2403.02839](https://arxiv.org/abs/2403.02839)

**Setup**
- **Fine-tuned judges:** JudgeLM-7B, PandaLM-7B, Auto-J-13B and Prometheus-7B/13B.
- **Compared against:** GPT-3.5, GPT-4 and DeepSeek-V3.
- **Tests:** each fine-tuned judge is run on the others' test sets, on MT-Bench (multi-turn), on
  LLMBar-adversarial, and on aspect-specific sets (HaluEval, ToxicChat).

**Findings**
- **Each fine-tuned judge wins only at home.** Prometheus-13B beats GPT-4 on its own test set (Pearson
  0.864 vs 0.742) but scores 24.58 on JudgeLM's (Table 2, p.2).
- **Multi-turn MT-Bench:** the fine-tuned judges score 48.7–55.2 against 66.9 for GPT-4.
- **LLMBar-adversarial:** the fine-tuned judges score 16.5–46.8 against 64.2–76.6 for GPT-4
  (Table 4, p.3).
- **Prompting does not help them.** Aspect-specific prompts, CoT and in-context examples do not
  improve the fine-tuned judges, and in-context examples hurt JudgeLM (Table 7, p.4).
- **The core claim:** "the fine-tuned judge model inherently operates as a task-specific classifier"
  (p.1).

**Implication**
- **Q1:** do not substitute an off-the-shelf fine-tuned judge (Prometheus, JudgeLM, …) when the
  evaluation scheme differs from its training data. All three of ours do.
- **Q2:** prompt engineering only pays off with general models.

**Caveat:** two of the four test sets were labelled by GPT-4, which partly flatters GPT-4.

#### 4 · Tan et al. (2025) — *JudgeBench* · [PDF](papers/2024_Tan_JudgeBench_ICLR2025.pdf) (local only) · [arXiv:2410.12784](https://arxiv.org/abs/2410.12784)

**Setup**
- **Data:** 350 hard response pairs in knowledge, reasoning, math and code.
- **How the pairs are built:** a strong model answers the same question several times, and one
  correct and one incorrect answer are kept. Style and length are therefore matched.
- **Labels:** objective correctness. No human preference is involved.
- **Scoring:** each pair is judged twice with the order swapped, and an inconsistent verdict counts
  as wrong.

**Findings**
- **Accuracy with the Arena-Hard prompt** (Table 2, p.7):

  | Judge | Accuracy |
  |---|---:|
  | o3-mini (high) | 80.86 |
  | o1-preview | 75.43 |
  | DeepSeek-R1 | 73.14 |
  | Claude-3.5-Sonnet | 64.29 |
  | GPT-4o | 56.57 |
  | Gemini-1.5-pro | 47.14 |

- **GPT-4o is no better than chance with a vanilla prompt.** It scores 50.86 (p.8).
- **Fine-tuned judges and debate fall below the "random" line.** Most fine-tuned judges score there,
  e.g. PandaLM 13.14. The multi-agent debate judge ChatEval scores 34.00 (Table 1, p.7).
- **Judging tracks solving.** "the ability of the judge to verify the solution pairs is highly
  correlated with its ability to solve the problem itself" (p.10). This rests on four models.
- **Prompt matters.** The Arena-Hard prompt, in which the judge answers the question first, beats the
  vanilla prompt.

**Implication**
- **Q1:** where a criterion is about correctness, prefer a reasoning model that can do the underlying
  task.
- **Q2:**
  - Swap order and count inconsistency as failure.
  - Let the judge solve the item before judging it.
  - Note that reasoning models "may not have respected … the zero temperature" (p.16).

**Caveat:** pairwise and objective tasks only. *(Our inference:)* under its double-trial scoring a
coin-flipping judge scores about 25%, not 50%, so some "below random" scores reflect order
inconsistency rather than worse-than-chance discrimination.

#### 5 · Panickssery et al. (2024) — *LLM Evaluators Recognize and Favor Their Own Generations* · [PDF](papers/2024_Panickssery_LLM-Evaluators-Favor-Own-Generations_NeurIPS2024.pdf) · [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)

**Setup**
- **Models:** GPT-4, GPT-3.5 and Llama-2-7b-chat, each both writing and judging summaries.
- **Data:** XSUM and CNN/DailyMail summarisation.
- **Fine-tuning:** used to change how well a model recognises its own text.

**Findings**
- **Recognition drives preference.** "By fine-tuning LLMs, we discover a linear correlation between
  self-recognition capability and the strength of self-preference bias" (p.1). The paper shows this
  in plots and reports no coefficient.
- **GPT-4 recognises itself out of the box.** "GPT-4 is 73.5% accurate distinguishing itself from
  two other LLMs and humans" (p.2).
- **The strongest model has the strongest self-preference.** GPT-4's pairwise self-preference is
  0.705 on XSUM and 0.912 on CNN (Table 7, p.14).
- **Source labels move the verdict.** GPT-4's XSUM self-preference is 0.73 with correct labels and
  0.32 with swapped labels (Table 6, p.13).
- **Order reverses many preferences.** Swapping the two options reverses 25% of GPT-4's preferences
  and 89% of Llama's (p.4).

**Implication**
- **Q1:** a judge should not score its own outputs, and the most capable judge is the most exposed.
- **Q2:**
  - Hide authorship and normalise formatting.
  - Average both orderings.

**Caveat:** only identical generator/judge pairs are tested, not same-family pairs. No human quality
baseline. Summarisation only.

#### 6 · Verga et al. (2024) — *Replacing Judges with Juries* (PoLL) · [PDF](papers/2024_Verga_Replacing-Judges-with-Juries_PoLL.pdf) (local only) · [arXiv:2404.18796](https://arxiv.org/abs/2404.18796)

**Setup**
- **Panel:** "three models being drawn from three disparate model families (Command R, Haiku, and
  GPT-3.5)" (p.3).
- **Aggregation:** max voting on binary QA; average pooling on 1–5 scores.
- **Compared against:** a single GPT-4 judge.
- **Data:** KILT QA (NQ, TriviaQA, HotpotQA), multi-hop QA, and Chatbot Arena Hard.

**Findings**
- **Agreement with humans on QA** (Table 1, p.4):

  | Judge | κ |
  |---|---|
  | PoLL | 0.763 / 0.906 / 0.867 |
  | GPT-4 | 0.627 / 0.841 / 0.830 |

  - Haiku alone beats PoLL on HotpotQA. The authors' reading: "there is not a single 'best' judge
    across all settings, while PoLL performs well consistently" (p.6).
- **Ranking on Arena Hard.** Kendall τ is 0.778 for PoLL against 0.667 for GPT-4 (Table 2).
- **Bias and spread.** "the highest positive delta for each individual model being scored occurs when
  it is judged by itself". PoLL's score spread has SD 2.2, against 6.1 for GPT-3.5 (p.5).
- **Prompt fragility.** GPT-4's κ ranges from 0.518 to 0.725 across prompts. The prompt tuned for
  GPT-4 *hurt* the other judges: GPT-3.5 fell from 0.729 to 0.509 (Table 4, p.10).
- **Cost.** The panel is "seven to eight times less expensive than running a single GPT-4 judge"
  (p.6).

**Implication**
- **Q1:** a cross-family panel is a credible alternative to one big judge, and it dilutes
  self-preference.
- **Q2:**
  - Draw few-shot examples from human-labelled items, including hard negatives.
  - Validate the *prompt and judge as a pair*, since a prompt tuned to one judge can hurt another.
  - Report κ.

**Caveat:**
- The QA task is "essentially a fuzzy string matching exercise" (p.5).
- One panel composition only, and Command R is the authors' own model.
- Not peer-reviewed.

### Part B — building the pipeline

#### 7 · Gu et al. (2026) — *A Survey on LLM-as-a-Judge* · [PDF](LLM_as_judge.pdf) · [arXiv:2411.15594](https://arxiv.org/abs/2411.15594)

**Definitions (p.2)**
- A judge is E ← P_LLM(x ⊕ C).
  - E is the evaluation: a score, choice, label or sentence.
  - x is the object judged, and C is the context, usually a prompt template.
- A *reliable* judge adds R ← f_R(P_LLM, x, C). Here f_R is "a series of constraints and validation
  methods": bias mitigation, variability control and adversarial robustness.

**The four-stage framework (Fig. 2A, p.4)**
1. **In-context learning.**
   - Input design: single, pairwise or batch.
   - Prompt format: a score (1–3, 1–5, 1–10, 0–100), yes/no, pairwise, or multiple choice.
2. **Model selection.**
   - A general LLM. Caveat: weak instruction following or reasoning "may significantly affect" the
     judge.
   - A fine-tuned judge. It "often" wins on its own test set but generalises poorly (p.4–5).
3. **Post-processing.**
   - Options: token extraction, constrained decoding to JSON, logit normalisation, sentence
     selection.
   - Rule-based extraction is "brittle", with "silent errors" (Box 5, p.7).
4. **The evaluation pipeline itself.**
   - Closed models raise cost and give "low reproducibility due to potential changes in models behind
     the API" (p.6).

**Findings**
- **"Quick practice" loop (p.7).**
  - Decide what to evaluate and how humans evaluate it.
  - Write the prompt with scoring dimensions, relative comparison and examples.
  - Choose "a large-scale model with strong reasoning and instruction-following abilities".
  - Standardise the output.
  - Retest.
- **Their meta-evaluation** (Table 1, p.15; LLMEval2 and EvalBiasBench):
  - Alignment with humans: o3-mini 61.66, GPT-4-turbo 61.54, Qwen2.5-7B 56.54, Llama3-8B 50.72.
  - Position consistency: GPT-4-turbo 80.31 vs Llama3-8B 38.85.
  - Their reading: "there was not much of a difference in alignment with humans among different LLMs".
- **Improvement strategies mostly disappoint** (Tables 2–3, p.16–17).
  - Asking for an explanation alongside the score *lowered* GPT-3.5's position consistency from 68.78
    to 48.97.
  - Self-validation had "minimal effectiveness".
  - A majority vote over 5 runs gave modest gains; mean-of-5 and best-of-5 gave none.
- **Panel composition matters.** Swapping one panel member raised position consistency from 32.28 to
  70.98 (p.16).
- **On self-judging:** "we should avoid using the same model as the evaluator", and anonymise model
  names (p.14).

**Implication**
- **Q2:** the framework above is the skeleton for §4. The concrete advice:
  - run a small meta-evaluation before choosing the model (p.16);
  - aggregate by majority vote, not by mean;
  - don't co-generate explanation and score by default;
  - set acceptance thresholds for your own context (p.13);
  - watch for "evaluation drift" across model versions (p.22).

**Caveat:**
- The experiment is small and pairwise only, with dated judges and no Claude model.
- Bias subsets are 24–34 items.
- The prose contradicts the tables in places: the "large margin" claim on p.15 is not borne out by
  Table 1.

#### 8 · Ye et al. (2025) — *Justice or Prejudice?* (CALM) · [PDF](papers/2024_Ye_Justice-or-Prejudice_CALM_ICLR2025.pdf) (local only) · [arXiv:2410.02736](https://arxiv.org/abs/2410.02736)

**Setup**
- **12 bias types:** position, verbosity, compassion-fade (model names shown), bandwagon,
  distraction, fallacy-oversight, authority, sentiment, diversity (identity), chain-of-thought,
  self-enhancement, refinement-aware.
- **Method:** each bias is injected by perturbing an answer, and the paper measures whether the
  verdict survives (the robustness rate, RR).
- **Judges:** 6 — GPT-3.5, GPT-4-Turbo, GPT-4o, Claude-3.5-Sonnet, GLM-4 and Qwen2-72B.

**Findings**
- **Position bias.** RR ranges from 0.566 (ChatGPT) to 0.832 (Claude-3.5) (Table 4, p.8). With 3–4
  options, "most models scor[e] below 0.5".
- **Which biases are worst** *(our calculation from Table 4)*. Averaged over the six judges:
  bandwagon ≈ 0.69, sentiment ≈ 0.69, position ≈ 0.76. Verbosity is mild at ≈ 0.92.
- **The newest model is not always safest.** "avoid assuming that the most advanced model will always
  be the most reliable" (p.7).
- **Subjective data is worse.** Biases are stronger on subjective alignment data than on fact data.
- **Self-enhancement.** Qwen2 gives its own answer 7.64 and others' identical answer 6.58 (Table 5,
  p.9). Quote: "the importance of using separate models for answer generation and evaluation" (p.8).
- **Some bias is invisible in the rationale.** Refinement-aware bias raises scores without ever being
  mentioned in the judge's explanation (p.10). Reading the explanations does not reveal it.
- **Instructions alone do not remove bias** *(our observation)*. Their baseline prompt already told
  the judge to avoid position, length and name bias (Fig. 13, p.25), and position RR was still
  0.566–0.832.

**Implication**
- **Q1:** choose by measured robustness on the biases that matter for the task, and separate the
  judge from the models being judged.
- **Q2:**
  - Randomise order.
  - Strip model names and identity cues.
  - Prefer a reference where one exists. Quote: "without a reference answer, it can be challenging
    for LLM judges to provide an objective score" (p.5).
  - Audit the prompt template for bias before use.

**Caveat:**
- No human ground truth.
- Temperature 0.7.
- The perturbations were written by models that are also judges.

#### 9 · Krumdick et al. (2026) — *No Free Labels* · [PDF](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) · [arXiv:2503.05061](https://arxiv.org/abs/2503.05061)

**Setup**
- **Data:** BFF-Bench, 160 finance questions, plus the MT-Bench math/reasoning items. 1,200 responses
  each carry 3 expert labels.
- **Judges:** GPT-4o, Llama 3.3 70B, Phi-4, Qwen 2.5 7B and Yi 1.5 34B.
- **References given to the judge:** none; Self (the judge's own answer); Human; Wrong (a human answer
  edited to be incorrect); Random (from another question).

**Findings**
- **The central result.** LLM-as-a-judge agrees well with humans only when "the LLM Judge (1) can
  already answer the underlying question or (2) is provided with a correct reference. Having
  confidence in a judge model requires meeting at least one of these two conditions" (p.9).
- **GPT-4o, pairwise κ** (Table 4, p.8):

  | Reference given | Questions GPT-4o answers wrongly | Questions it answers correctly |
  |---|---:|---:|
  | None | 0.30 | 0.78 |
  | Self | 0.16 | 0.86 |
  | Human | **0.83** | 0.92 |

- **What matters is that the reference is correct** (Table 3, p.7):

  | Reference | κ |
  |---|---:|
  | Human | 0.69 |
  | Verified GPT-4o | 0.61 |
  | Random | 0.50 |
  | None | 0.46 |
  | Wrong | 0.21 |

  - A slightly wrong reference can be worse than none.
- **A small judge with a reference beats a big one without.** Qwen 2.5 7B with a human reference
  reaches κ 0.63, against 0.47 for GPT-4o with none (Table 6, p.23).
- **A judge's own reference makes self-preference worse** (p.10).
- **References themselves need checking.** 37.5% of MT-Bench's GPT-4 reference answers were wrong
  (p.4).
- **Juries and decoding settings do not change the picture.**
  - A 5-judge jury does not escape the limit: κ is 0.07 with no reference and 0.51 with a human
    reference, on the questions the majority got wrong (p.27–28).
  - Temperature and CoT made no significant difference.
- **The pattern holds for current frontier judges.** Opus 4.7, GPT-5.4 and Gemini 3.1 show it too,
  against proxy labels (Table 12, p.29).

**Implication**
- **Q1:** judge size matters far less once verified references exist. Without them, the judge must be
  able to solve the items.
- **Q2:** "we strongly recommend practitioners (at minimum) verify their reference responses" (p.10).
  Never use the judge's own answer as the reference.

**Caveat:** one domain, finance, plus math, where answers are clearly right or wrong. Subjective
quality was not studied.

#### 10 · Norman et al. (2026) — *Reliability without Validity* · [PDF](papers/2026_Norman_Reliability-without-Validity.pdf) · [arXiv:2606.19544](https://arxiv.org/abs/2606.19544)

**Setup**
- **Judges:** 21 from 9 providers, current as of April 2026. They include GPT-4o/4.1/5.4, Gemini
  2.5/3.1 Pro, Claude Haiku 4.5/Sonnet 4.6/Opus 4.6, DeepSeek V3.2, Kimi K2.5, GLM-5 and Qwen 3 8B.
- **Data:** MT-Bench (expert preference), JudgeBench (correctness) and RewardBench.
- **Scale:** about 541,000 judgments.
- **Settings:** temperature 0, with thinking suppressed.

**Findings**
- **Exact match overstates agreement.** On MT-Bench, exact match exceeds Cohen's κ by 33.8–41.3 points
  for every judge. Quote: "a judge reporting "85% agreement" on MT-Bench has κ ≈ 0.48" (p.5).
- **Rankings do not transfer.**
  - Judge rankings shift by up to 14 positions between benchmarks (abstract; the body says 15 in
    places).
  - Only Gemini 3.1 Pro and Claude Opus 4.6 stay in the top 3 on all three benchmarks (p.5).
- **Stability is not validity.** Qwen 3 8B has test–retest 0.992 but position bias 0.192.
  Quote: "Reporting test-retest alone stands to present a misleading picture" (p.6).
- **Family differences on JudgeBench.**
  - The three Anthropic judges average κ = 0.770 on JudgeBench, with mean position bias 0.020.
  - The OpenAI flagships average 0.467; GPT-4o is at 0.309 and GPT-5.4 at 0.606.
  - Kimi K2.5 has position bias 0.004 and κ 0.720 at a fraction of the cost (p.7).
- **A data-loader trap.** RewardBench's default loader puts every chosen answer in position A, which
  makes κ 0 for every judge until positions are randomised per item (p.21).
- **The Minimum Viable Validation Protocol** (p.8, verbatim headings):
  1. **Chance-correct.** Report Cohen's κ or Krippendorff's α as the headline number.
  2. **Swap positions.** Report |P(A wins) − 0.5| from AB+BA runs.
  3. **Replicate.** At least 3 runs at temperature 0, with caching disabled.
  4. **Cross-validate.** At least 2 benchmarks, spanning preference-style and correctness-style labels.
  5. **Audit the paradox.** If test–retest exceeds 0.95, check that position bias is below 0.10
     before claiming reliability.

**Implication**
- **Q1:** a leaderboard on a different construct is not evidence for our task. Validate on data whose
  label structure matches ours.
- **Q2:** the protocol above.

**Caveat:**
- A preprint, with several internal number inconsistencies.
- Pairwise only.
- Thinking was suppressed.
- No confidence intervals.

---

## 3. Synthesis — Q1: choosing the judge model

| Option | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| **The top general model** | Only the largest judges approach humans (Thakur). General models beat fine-tuned ones (Huang). The survey's "quick practice" (Gu p.7). | No judge is best everywhere (Bavaresco). GPT-4 was among the weakest on KILT QA (Verga). "avoid assuming that the most advanced model will always be the most reliable" (Ye). Rankings shift by up to 14 places (Norman). | **Necessary, not sufficient.** It defines the shortlist, not the choice. |
| **A reasoning model** | o3-mini 80.86 against GPT-4o 56.57 on hard correctness pairs (Tan). | Gains on human-alignment tasks are "not as pronounced as expected" (Gu p.17). Temperature 0 may not be honoured (Tan p.16). | **Prefer it for correctness-type criteria**: soundness, rubric satisfaction. |
| **Chosen by task, validated on our data** | The explicit recommendation of Bavaresco and Gu (p.16). Rankings do not transfer (Norman). Reliability depends on the items and on references (Krumdick). | Costs human labelling. | **The core of the procedure.** |
| **A fine-tuned judge** (Prometheus, JudgeLM, Auto-J) | Privacy and a fixed version (Gu). | Collapses off its home distribution (Huang). Below random on hard pairs (Tan). JudgeLM π = 65 (Thakur). | **Don't.** |
| **A panel of judges from different families** | Higher κ, 7–8× cheaper, less self-bias (Verga). Majority vote helps a little (Gu). | Does not fix a missing reference (Krumdick). Composition swings results from 32 to 71 (Gu p.16). Debate-style multi-agent judging scored 34 (Tan). | **A good complement.** Choose members by validation, not by name. |
| **A judge outside the evaluated families** | Self-preference (Panickssery). Self-enhancement (Ye). The self-delta finding (Verga). Gu p.14. | "only a stopgap" when evaluating the very best models (Gu p.14). | **A hard constraint wherever possible.** |

**What this means for our model set *(our inference)*.**
- **The families we evaluate** are listed in [PLAN.md](../PLAN.md): Google (Gemini, Gemma), OpenAI,
  xAI, Alibaba (Qwen) and DeepSeek.
- **Anthropic and Moonshot (Kimi) are both outside that set.** In Norman et al., the most recent
  multi-provider comparison here:
  - Anthropic judges had the highest JudgeBench κ and the lowest position bias.
  - Claude Opus 4.6 was one of two judges in the top 3 on every benchmark.
  - Kimi K2.5 was close behind at much lower cost.
- **A shortlist this review would justify:**
  - a current Claude model as the primary judge;
  - a second out-of-family model, such as Kimi, as a cross-check or second panel member.
  - Both are validated on our own human-labelled samples before either is adopted.
- **A Google or OpenAI judge** — Gemini 3.1 Pro scored well — would be judging its own family on every
  sheet.
- **Disclosure.** This review was written with Claude (Anthropic). The recommendation follows from the
  family-exclusion argument and Norman's numbers. It should be settled by the validation step in §4,
  not taken on trust.

---

## 4. Synthesis — Q2: building the pipeline

Each step cites the papers it rests on. Steps 1–6 build the judge; steps 7–8 decide whether to trust
it.

1. **Fix what is measured.**
   - Write down the criteria, the scale and its *direction*, and who wrote the references.
   - Ask how humans would judge it (Gu's "thinking" stage).
   - The fields to record are in [JUDGE_DOCUMENTATION_RULE.md](JUDGE_DOCUMENTATION_RULE.md).
2. **Ground the judge.**
   - Give it a verified reference answer or a per-item rubric wherever one exists (Krumdick; Ye p.5).
   - Never give it a reference the judge or another LLM wrote unchecked (Krumdick p.10).
   - Audit the references: 37.5% of MT-Bench's were wrong (Krumdick p.4).
3. **Write the prompt.**
   - Start from the human annotation guidelines (Bavaresco; Thakur).
   - Add explicit edge-case rules, e.g. under-specified = incorrect (Thakur App. G).
   - Use one criterion per call (Gu, criteria decomposition).
   - Draw few-shot examples from human-labelled items, including hard negatives (Verga).
   - Validate the *prompt and judge as a pair*: a prompt tuned for one judge can hurt another
     (Verga, Table 4).
4. **Hide authorship.**
   - Strip model names and normalise formatting (Panickssery; Ye, compassion-fade; Gu p.14).
5. **Constrain the output.**
   - A single label or score in a fixed format, or structured/JSON output (Gu, post-processing;
     Bavaresco).
   - Log invalid outputs and refusals as their own category. Never impute them (Bavaresco imputed
     randomly; Tan retried once).
   - Don't ask for an explanation with the score by default (Gu p.16).
   - If you use CoT, measure it: it helped on adversarial items and hurt on safety (Bavaresco).
6. **Remove order effects.**
   - Swap or shuffle any ordered input — candidates, or a list of reference steps — and count
     disagreement as inconsistency (Tan; Panickssery; Thakur; Norman).
7. **Choose decoding and replication.**
   - Temperature 0 with at least 3 replicate runs, and test–retest reported (Norman).
   - Alternatively, a majority vote over 5 runs (Gu p.16; Krumdick). Majority, never mean or
     best-of.
   - Record the setting when a reasoning model will not honour temperature 0 (Tan p.16).
8. **Validate before trusting a number.**
   - Use a human-labelled sample from *each* benchmark (Bavaresco; Gu p.16; Norman).
   - Report raw agreement **and** a chance-corrected statistic — Cohen's κ, Scott's π or
     Krippendorff's α — with confidence intervals (Thakur; Norman).
   - Compare against inter-annotator agreement as the human upper bound (Bavaresco).
   - Check that the label distribution is not degenerate before trusting κ (Norman p.7, p.21).
   - Measure leniency, the share judged positive (Thakur).
   - Run dummy-answer sanity checks (Thakur).
   - Audit the biases relevant to the task (Ye).
   - Then **freeze** the model snapshot, prompt and parser, and record them, because hosted models
     drift (Gu p.22).

---

## 5. Applied to our three judged benchmarks *(our inference)*

The judge setups themselves are in [JUDGE_RECORD.md](JUDGE_RECORD.md); only the literature's bearing
on each is noted here.

- **Wonderbread QA** (1–3 per criterion; a human reference for two of four criteria).
  - "Completeness" is exactly Thakur's under-specified-answer weak spot, where recall was 23–34%. The
    validation sample should deliberately include incomplete answers.
  - Clarity and compactness have no reference, so they sit in the zone Ye and Krumdick flag as most
    exposed to style and sentiment bias.
  - The SOP-Generation judge matches one line against an indexed list, so test whether shuffling the
    list order changes its verdict.
- **MultiChallenge** (binary YES/NO against a human-written per-item rubric).
  - The rubric plays the role of Krumdick's "correct reference", which is what makes a judge
    trustworthy. The residual risks are rubric errors and leniency (Thakur).
  - Binary labels with a lopsided pass rate can make κ unstable. Check the balance first (Norman).
- **AwareBench** (60 rows; binary; no reference).
  - This is the worst case in the literature: reference-free, subjective and value-adjacent. It is
    where Bavaresco found judges near or below κ = 0, and where neither of Krumdick's two conditions
    can be met.
  - The published 28-point swing between two evaluator prompts ([JUDGE_SUMMARY.md](JUDGE_SUMMARY.md)
    §6) is what Ye and Gu would predict.
  - Sixty rows is small enough to label every row by hand. Do that before any judge number is
    reported.

---

## 6. Further reading

These were not downloaded or read in full; each summary comes from the abstract.

- **Jung et al., *Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement*** —
  ICLR 2025, [arXiv:2407.18370](https://arxiv.org/abs/2407.18370).
  - Cascaded selective evaluation: a cheap judge first, escalating to a stronger one only when
    confidence is low, with a provable human-agreement guarantee.
  - A further Q1 option.
- **Li et al., *Preference Leakage: A Contamination Problem in LLM-as-a-judge*** — ICLR 2026,
  [arXiv:2502.01534](https://arxiv.org/abs/2502.01534).
  - Judges favour models related to them — the same model, an inheritance relationship, or the
    **same family** — and this is "harder to detect" than other biases.
  - Extends Panickssery from same-model to same-family.
- **Dorner et al., *Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the
  data*** — ICLR 2025, [arXiv:2410.13341](https://arxiv.org/abs/2410.13341).
  - When the judge is no more accurate than the evaluated model, no debiasing method can cut the
    required ground-truth labels by more than half.
- **Shi et al., *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge*** —
  AACL-IJCNLP 2025, [arXiv:2406.07791](https://arxiv.org/abs/2406.07791).
  - 15 judges and over 150k instances.
  - Position bias is not random, varies by judge and task, and grows as the quality gap between
    answers shrinks.
- **Yamauchi et al., *An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation
  Reliability*** — 2025, [arXiv:2506.13639](https://arxiv.org/abs/2506.13639).
  - Explicit evaluation criteria are critical.
  - Non-deterministic sampling improved alignment with humans.
  - CoT adds little when the criteria are clear.
- **Salinas et al., *Tuning LLM Judge Design Decisions for 1/1000 of the Cost*** — ICML 2025,
  [arXiv:2501.17178](https://arxiv.org/abs/2501.17178).
  - Searches judge hyperparameters (model, prompt, decoding) jointly.
  - Finds open-weight judges that are competitive.
- **Han et al., *Judge's Verdict*** — 2025 preprint,
  [arXiv:2510.09738](https://arxiv.org/abs/2510.09738).
  - 54 LLMs scored against humans on judging the accuracy of RAG and agentic answers, with a
    κ-based tiering.
- **Feng et al., *Are We on the Right Way to Assessing LLM-as-a-Judge?* (Sage)** — 2025 preprint,
  [arXiv:2512.16041](https://arxiv.org/abs/2512.16041).
  - Measures judges without human labels, using preference stability and transitivity.

---

## 7. Files

Licences were checked on each paper's arXiv abstract page, 2026-10-05.

**Committed** (open licences):

| PDF | Licence |
|---|---|
| [LLM_as_judge.pdf](LLM_as_judge.pdf) (Gu et al., journal version) | CC BY-NC-ND |
| [papers/2024_Thakur_…](papers/2024_Thakur_Judging-the-Judges_GEM2025.pdf) | CC0 |
| [papers/2024_Bavaresco_…](papers/2024_Bavaresco_LLMs-instead-of-Human-Judges_ACL2025.pdf) | CC BY 4.0 |
| [papers/2024_Panickssery_…](papers/2024_Panickssery_LLM-Evaluators-Favor-Own-Generations_NeurIPS2024.pdf) | CC BY 4.0 |
| [papers/2025_Krumdick_…](papers/2025_Krumdick_No-Free-Labels_COLM2026.pdf) | CC BY-NC-SA 4.0 |
| [papers/2026_Norman_…](papers/2026_Norman_Reliability-without-Validity.pdf) | CC BY 4.0 |

**Local only:**
- **Which:** Huang, Tan, Verga and Ye.
- **Why:** they carry arXiv's default non-exclusive licence, which grants no right to redistribute,
  and both remotes of this repo are public. They are listed in [papers/.gitignore](papers/.gitignore).
- **On a fresh clone:** re-download them from the arXiv links above.
