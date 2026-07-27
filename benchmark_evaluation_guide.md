# Human–AI Benchmarks: Evaluation Methods and Requirements

> Source: `Human_AI_Documentation.pdf`. This guide covers the seven unfinished benchmarks. DocVQA, BIG-Bench Hard, and MMLU have already been evaluated objectively and are therefore excluded. EmoBench appears twice in the PDF but is counted only once.

## Overview

| Benchmark | Core task | Metrics/scoring | LLM-as-a-Judge? | Required evaluation assets |
|---|---|---|---|---|
| AWAREBENCH | Binary, multiple-choice, and open-ended awareness questions | Accuracy for objective questions; GPT-4 evaluates human alignment and generation quality for open-ended mission-awareness questions | **Yes—open-ended mission awareness only** | AWAREEVAL data, question types, gold answers, model responses, GPT-4 judge, judge prompt/rubric, response parser, and aggregation scripts |
| Multi-party Goal Tracking | Fill `[MASK]` annotations for goal tracking or intent-slot recognition in multi-party dialogue | Exact Match, Correct, and Partial (at least 60% of masks correct) | **No** | 29 annotated conversations, mask construction logic, gold annotation strings, zero/few-shot prompts, parser, and three-level scoring rules; manual review when functional equivalence cannot be rule-based |
| PlanBench | Plan generation, optimal planning, plan verification, execution reasoning, and replanning | A VAL/PDDL validator checks executability and goal completion; optimal tasks also compare plan cost | **No** | PDDL domains/problems, natural-language translator, prompts/examples, model plans, plan extractor, VAL or equivalent validator, reference planner/optimal costs, and metric scripts |
| Wonderbread | SOP generation, demo segmentation, workflow QA, demo validation, SOP ranking, and SOP improvement | Semantic Precision/Recall, ARI, GPT-4 QA scores, F1/Accuracy, Kendall τ/Spearman, and rubric-based improvement scores | **Yes—explicitly for QA**; the document also notes “GPT as a judge & human alignment” | Videos/keyframes, action traces, human Gold SOPs, task intents, QA context, validation labels, human rankings, GPT-4 judge/rubric, and metric implementations |
| MultiChallenge | Answer the final turn of a conversation of up to ten turns while retaining instructions, user facts, versions, and self-consistency | An LLM judge answers an instance-specific binary rubric; results are aggregated as Accuracy | **Yes—all final responses** | Full histories, final user turns, human-written binary rubrics, model responses, judge model/prompt, binary parser, and Accuracy script |
| NegotiationToM | Identify Desire, Belief, and Intention in negotiation dialogues | Desire/Belief Exact Match, Intention micro/macro F1, All score, and Consistency score | **No** | 395 CaSiNo dialogues, about 13.8K questions, answer choices, gold labels, multi-label intention annotations, zero/few-shot/CoT prompts, parser, and metric scripts |
| EmoBench | Multiple-choice Emotional Understanding (EU) and Emotional Application (EA) | Accuracy after repeated sampling, majority voting, and averaging across four answer-order permutations | **No** | 400 English/Chinese MCQs, choices, gold labels, EU/EA categories, option shuffler, repeated inference and voting logic, heuristic parser, and Accuracy script |

## LLM-as-a-Judge Summary

### Required by the original benchmark

1. **AWAREBENCH:** GPT-4 judges only the open-ended mission-awareness responses. It evaluates whether the answer prioritizes human needs and assesses generation quality. Binary and multiple-choice items use objective Accuracy.
2. **Wonderbread:** GPT-4 scores Question Answering on soundness, completeness, clarity, and compactness, each on a 1–3 scale. The document also describes the benchmark generally as using “GPT as a judge & human alignment.”
3. **MultiChallenge:** Each item has a human-written yes/no rubric. The judge applies that rubric to the model’s final response, and the binary results are aggregated into Accuracy.

### Not required

- **PlanBench** uses a symbolic planning validator; an LLM judge should not replace VAL.
- **NegotiationToM** and **EmoBench** have discrete gold labels and are scored objectively.
- **Multi-party Goal Tracking** is scored against gold annotations. Its “functionally identical” and broad `Correct` categories may require deterministic equivalence rules or manual review, but this is not LLM-as-a-Judge.

## Detailed Evaluation Procedures

### 1. AWAREBENCH

AWAREBENCH evaluates:

- **Introspective Awareness:** Capability Awareness and Mission Awareness.
- **Social Awareness:** Emotion, Culture, and Perspective Awareness.

Procedure:

1. Send AWAREEVAL binary, multiple-choice, and open-ended items to the target model using the official prompt format.
2. Parse the final choice for binary and multiple-choice items, compare it with the gold label, and calculate overall and per-dimension Accuracy.
3. For open-ended mission-awareness items, send the question and target-model response to a fixed GPT-4 judge version.
4. Apply the fixed rubric to obtain a human-alignment judgment and generation-quality score, preferably in a strict structured format.
5. Report objective Accuracy separately from judge-based scores.

Fix and record the judge model version, temperature, prompt, retry rules, and raw judge responses. A manually reviewed sample is advisable for reliability auditing.

### 2. Multi-party Goal Tracking with LLMs

The dataset contains 29 multi-party conversations and 774 turns involving patients, companions, and the ARI robot. It includes:

- **Goal Tracking:** Predict goal states such as `G` (Established), `AG` (Answered), and `CG` (Closed), including the relevant speaker or group.
- **Intent-slot Recognition:** Predict an intent and its associated slots/values.

Procedure:

1. Construct inputs containing `[MASK]` using the official context-window rules.
2. Run zero-shot and few-shot settings separately. The reproduced few-shot setting uses approximately 7% of the corpus as examples.
3. Extract the annotation string for every mask.
4. Assign one of three outcomes:
   - **Exact Match:** Identical to the gold annotation or equivalent under predefined normalization rules.
   - **Correct:** Every mask is semantically correct, although a prediction may use a broader slot category.
   - **Partial:** At least 60% of masks are correct.
5. Report Goal Tracking and Intent-slot Recognition separately.

The main reproducibility risk is the boundary of `Correct`. Define executable rules for synonyms, parent slots, plurality, and formatting before evaluation; use manual review only for unresolved cases.

### 3. PlanBench

PlanBench primarily uses Blocksworld and Logistics. It covers plan generation, cost-optimal planning, plan verification, execution reasoning, goal reformulation, plan reuse, replanning, and generalization.

Procedure:

1. Convert each PDDL instance into natural language with the domain-specific translator.
2. Provide a lifted domain description, one or more examples, and the target instance in the prompt.
3. Extract and normalize the action sequence from the response. If no valid plan can be extracted, mark the instance incorrect.
4. Submit the sequence to VAL or an equivalent PDDL plan validator.
5. Count plan generation as successful only when the plan is executable and reaches the goal state.
6. For cost-optimal tasks, compare the generated plan cost with the optimal cost produced by a reference planner.
7. Report obfuscated domains, shuffled goals, and other robustness settings separately from standard settings.

Scoring should remain programmatic and verifiable. A plausible natural-language explanation does not pass if the extracted plan fails validation.

### 4. Wonderbread

Wonderbread contains 2,928 expert demonstrations covering 598 workflows. A demonstration may include a full-screen recording, click/keyboard/scroll traces, and a human-written SOP.

| Subtask | Input | Prediction | Scoring |
|---|---|---|---|
| SOP Generation | Intent plus video/keyframes/action log | SOP steps | Semantic step matching against the Gold SOP; Precision and Recall |
| Demo Segmentation | Long video and event sequence | Frame-to-workflow segmentation | Adjusted Rand Index |
| Question Answering | Multimodal workflow context and question | Free-text answer | GPT-4 scores soundness, completeness, clarity, and compactness from 1–3 |
| Demo Validation | Demonstration plus goal/SOP | Success or strict SOP-compliance label | F1 and Accuracy |
| SOP Ranking | Multiple demonstrations of the same task | Ranking | Kendall τ and Spearman correlation against human rankings |
| SOP Improvement | Demonstration plus a low-quality SOP | Improved SOP | A task-specific 1–5 rubric |

Run and report each subtask independently because their input modalities and metrics are not directly comparable. The document explicitly identifies GPT-4 as the QA judge. It states that SOP Improvement uses a 1–5 rubric but does not clearly identify the scorer in that section; confirm this against the official Wonderbread evaluator before implementation.

### 5. MultiChallenge

Its four challenge categories are Instruction Retention, Inference Memory, Reliable Versioned Editing, and Self-Coherence.

Procedure:

1. Send the full conversation history of up to ten turns and the final user message to the target model.
2. Retain the human-written, instance-specific binary rubric for every test case—for example, “Does any suggested recipe contain nuts?”
3. Give the dialogue, final user request, model answer, and instance rubric to the judge.
4. Require a strict yes/no or structured Boolean output and map it to pass/fail.
5. Calculate overall Accuracy and per-category Accuracy.

A generic “Is this a good answer?” rubric is not valid. The benchmark depends on its instance-level rubrics; omitting them changes the evaluation.

### 6. NegotiationToM

NegotiationToM uses real two-party CaSiNo negotiations over food, water, and firewood to test mental-state tracking.

Procedure:

1. Load dialogue histories, questions, choices, and labels from the official split.
2. Run zero-shot, few-shot, and Chain-of-Thought settings separately.
3. For Desire and Belief, require all three preference levels—high, medium, and low—to be correct simultaneously for Exact Match.
4. Treat Intention as multi-label classification and calculate micro-F1 and macro-F1.
5. Calculate the **All** score by requiring Desire, Belief, and Intention to be correct for the same information unit.
6. Calculate Consistency across the full dialogue to measure whether the model tracks mental-state changes correctly.

Do not replace the grouped Desire/Belief Exact Match with ordinary per-choice Accuracy, as that would overestimate performance.

### 7. EmoBench

The 400 English and Chinese MCQs cover:

- **Emotional Understanding (EU):** Identify emotions and causes, including complex emotions, cues, beliefs/experiences, and perspective-taking.
- **Emotional Application (EA):** Choose the most effective action in personal/social and self/other scenarios.

Procedure described in the document:

1. Use the original option order plus three random reorderings, producing four permutations per question.
2. Query the model five times for each permutation.
3. Extract the selected option using heuristic parsing rules.
4. Apply majority voting across the five responses to obtain the final answer for that permutation.
5. Calculate Accuracy for each permutation and report the mean of the four runs.
6. Also report results by language, EU/EA, and subcategory when possible.

The phrase “prompt each LLM five times (5-shot)” is ambiguous. Because the next step uses majority voting, it appears to mean five repeated samples rather than five demonstrations in a prompt. Confirm the exact behavior in the official implementation and document it explicitly.

## Minimum Implementation Checklist

- A target-model API endpoint or local inference server, with an exact model version.
- Official datasets and splits, including inputs, gold labels, instance rubrics, multimodal files, and PDDL files where applicable.
- Official prompts/templates and a declared zero-shot, few-shot, CoT, or multi-turn context policy.
- Fixed inference settings: temperature, top-p, maximum tokens, seed, concurrency, and retry policy.
- Complete logs: prompts, raw responses, parsed answers, errors, and retries.
- Objective scorers: Accuracy, Exact Match, F1, ARI, Kendall τ, Spearman, Consistency, and majority voting.
- Specialized tools: a PDDL planner/VAL for PlanBench and video/keyframe/action-log loaders for Wonderbread.
- Judge service: a fixed GPT-4 version, or a precisely documented substitute, for the applicable AWAREBENCH, Wonderbread, and MultiChallenge tasks.
- Judge configuration: complete rubrics, structured-output schema, parse-failure policy, caching, and manual audit procedure.
- Disaggregated reports by benchmark, task, category, language, and prompt setting.

## Recommended Result Schema

```text
benchmark
subset / category
sample_id
model_name
model_version
prompt_setting
inference_config
raw_output
parsed_answer
gold_answer_or_rubric
objective_score
judge_model
judge_prompt_version
judge_raw_output
judge_score
error_type
```

Populate the `judge_*` fields only for AWAREBENCH open-ended questions, Wonderbread judge-based subtasks, and MultiChallenge. Leave them empty for objectively scored benchmarks.
