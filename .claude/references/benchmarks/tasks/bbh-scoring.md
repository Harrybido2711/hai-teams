# BBH — the scorer, and every time it was wrong

<!-- size-budget: 6000 -->

Split out of [bbh.md](bbh.md) on 2026-08-29, when that page acquired a second job. The card
describes the benchmark; this describes the one matcher every model here is judged by, and the five
times reading real rows showed it was penalising a correct answer.

`bbh_eval_core.py::score_response` is the only scorer in this benchmark and every runner imports it.
**This is the benchmark the project's one-scorer rule came from** (`CLAUDE.md`;
`script-skeleton.md` rule 7).

## The lesson, stated once

**Four of the five versions came from looking at rows, not from thinking about the matcher.** Every
one was a correct answer scored 0 for how it was written. The strongest single tell:

> **A task at exactly 0.000 with `no_marker=0` and `empty=0` is a scorer bug, not a result.**

Luna hit 0.000 on two whole tasks — a binary task at 0/250 is not something a model does — because
it answered `8 musical instruments` and `No, Ka does not tell the truth.`

**Every candidate branch is measured over all stored rows before being kept**, reporting rows gained
and rows lost. Branches are additive and tried in order, so a new one can only gain — which means
the real risk is a *false* gain, and that is what the samples are read for.

## Version history

| Version | What it added | Measured |
|---|---|---|
| **v1** | the original six branches: exact; letter for letter; option text for letter; comma-vs-space; the `dyck` prefix splice; both sides comma-normalised | — |
| **v2** | strips markdown emphasis (`**bold**`) | +31 rows, 0 lost |
| **v3** | four packaging branches: LaTeX math, number + unit noun, restated closed-set answer, option letter at the end | **+766 rows, 0 lost, 5 models** |
| **v4** | brackets without separators, `})>` for `} ) >` | +190 rows, 0 lost, 6 models |
| **v5** | closed-set synonyms, `True` for `Yes` | +18 rows, 0 lost |

v2 moved 10 task-cells; v3 moved Luna 0.8102 → 0.9213 and 4o-mini's `web_of_lies` 0.716 → 0.944;
v4 moved Kimi's `dyck_languages` 0.572 → 0.944.

**The v3 branches fire only when the model emitted the `Final Answer:` marker.** Without it,
`final_answer` is a scrape of the whole response, and "the gold letter appears somewhere in the
reasoning" is not an answer — the first draft of branch 10 credited a Llama row whose reasoning
mentioned `(D)` once, for an answer it never gave.

`SCORER_VERSION` is written onto every result file. A v1 number cannot be compared with a v5 one
without rescoring.

## The line not crossed

48 Luna rows answer `Elanor does not tell the truth.` for gold `No`. Crediting them needs negation
parsing specific to `web_of_lies`: measured at **+92 rows, 0 losses**, but it would make the matcher
*task-aware* rather than packaging-tolerant, which is a different kind of thing from every branch
above. **Not adopted — the user's call.**

The prompt route was tried instead and came up short: `--prompt v2` states that the answer must
stand alone, and moved Luna's `web_of_lies` only 0.728 → 0.780, because the model keeps restating.
**Prompting is a weaker lever here than it looks.**

## The prompt is part of the config

`--prompt v1|v2`, recorded on every row. The resume guard refuses to mix two prompt versions in one
result set; archive to `results_archive_promptv1_<ts>/` rather than resuming across a change.
Until this existed the guard could not see a prompt change at all.
