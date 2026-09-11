# DocVQA — the scorer, and the branch that was measured and refused

<!-- size-budget: 5500 -->
<!-- Split out of docvqa.md on 2026-09-11, the same way bbh-scoring.md was split out of bbh.md and
     for the same reason: the card describes the benchmark, this describes the one matcher every
     model here is judged by. It exists mostly to hold the refusal — without the measurement behind
     it, the next reader will re-propose fuzzy matching and it will look reasonable. -->

`docvqa_eval_core.py` holds two numbers per row and they are not the same kind of thing.

- **ANLS is the reported metric** and it is *graded*: a near-miss scores its similarity, not zero.
  It is the challenge's own metric and it takes the **maximum over the question's answer list**
  with a **0.5 threshold** below which a row contributes nothing.
- **`score` is a binary accuracy column** this project added. It is the one that has a version, and
  the one that was wrong.

**Reading the ANLS row first prevents most arguments.** The two rows a reader is most likely to
query — a model answering `Grocery Manufacturers of America, Inc.` and a model expanding a pronoun —
already score 0.947 and 0.811 in ANLS. Only the binary column called them zero.

## Version history

| Version | What it added | Measured over all 32,094 stored rows |
|---|---|---|
| **v1** (`docvqa_anls_v1`) | the four branches the standalone runners each carried: exact; comma-vs-space; punctuation stripped; either side contained in the other | — |
| **v2** (`docvqa_lenient_v2`) | **5** — spacing *and* punctuation removed together | **+311 rows, 6 models** |
| | **6** — one adjacent transposition, on strings of 8 characters or more | **+12 rows** |

Branch 5 is what catches `JAN 17 '68` against `jan17'68`, `DR.W.J.DARBY` against `Dr. W. J. Darby`,
`3. 28-2001` against `3-28-2001`. v1 normalised spacing **or** punctuation but never both at once,
which is exactly where those fall through.

Branch 6 exists because **the gold answers contain typos**. All twelve rows it credits were read
individually, and they are three distinct questions: `Grocrey` for `Grocery`, `laways` for
`always`, `Cigfil` for `Cigifl`. An adjacent swap is the signature of a typing slip; a substitution
is the signature of a misreading, and those are left wrong.

**The 8-character floor is load-bearing.** Without it the branch credits `14` for `41` — measured,
that was its single false gain across all six models, and the floor removes it while keeping all
twelve real ones.

## The line not crossed: no fuzzy matching

Three fuzzy variants were measured on every stored row, and **all three were refused**:

| Candidate | Gained | Why refused |
|---|---|---|
| normalised edit distance ≤ 0.10 | +451 | credits `Paul Saliman` for `Paul Saltman`, `912-281-0092` for `912-281-0012`, `Larry McChee` for `Larry McGhee` |
| ≤ 0.20 | +775 | adds `$69,634.39` for `$89,434.39` and `50522 9283` for `50572 6283` |
| ANLS ≥ 0.5 as a binary credit | +1,148 | adds `38` for `18`, `70` for `10`, `More` for `none`, `Dowl` for `Done` |

**This is a document-reading benchmark. Misreading a name, a phone number or a dollar amount is the
error class it exists to measure**, and a matcher that forgives it is not lenient, it is broken.
Short strings make the arithmetic worse: two characters differing by one is a distance of 0.5.

The distinction every adopted branch respects is bbh's:
[bbh-scoring.md](bbh-scoring.md) states it as **normalise how an answer is written, never interpret
what it means**. Branches 5 and 6 change the writing. Fuzzy distance tolerates being wrong.

**The pronoun case is the boundary, and it is left to ANLS.** `Whether soft drinks cause teeth to
erode` against gold `whether they cause teeth to erode` is the model resolving a pronoun — right in
substance, and no branch above credits it. ANLS scores it 0.811 and that is the answer: the graded
metric is where "nearly right" belongs, and forcing the binary column to agree is what produced
every bad candidate in the table.

## Rescoring

`rescore_docvqa.py`, run from the DocVQA directory on either side. It re-derives `score` from the
stored `model_response`, rewrites the result files and their overall CSVs, and stamps
`SCORER_VERSION`. It never touches ANLS, and it is idempotent. **A v1 number cannot be compared
with a v2 one** — both sides were rescored on 2026-09-11, and Quest was rescored too so that a
merge there could not quietly reassert a v1 figure under a v2 core.
