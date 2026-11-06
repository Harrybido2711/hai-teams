<!-- One job: how to read and rebuild Results/results.db. -->
# Results database

`Results/results.db` is a SQLite copy of every recorded score and the six criteria, made at the
user's request on 2026-11-06 because a workbook is awkward for an agent to read. **Query it instead
of parsing the Excel files.** It needs nothing but `sqlite3`:

```bash
sqlite3 -header -column Results/results.db "SELECT * FROM final_scores WHERE benchmark = 'EmoBench'"
```

**It is derived, so never edit it by hand.** The workbooks stay what the user reads and what a result
is recorded into; `.claude/scripts/build_results_db.py` rebuilds the database from them and from the
criteria files. A `Final_result` cell that differs from `Tempo_results` stops the build.

```bash
python3 .claude/scripts/build_results_db.py          # rebuild, after any change to a source
python3 .claude/scripts/build_results_db.py --check  # exit 1 if a source changed since the build
```

## Tables

| Table | One row per | Holds |
|---|---|---|
| `benchmarks` | benchmark, all ten | process group and workbook sheet; a NULL sheet has no reported result yet |
| `runs` | benchmark × workbook column | `model_config`; `selected` = 1 if the column is in `Final_result` |
| `scores` | benchmark × sub-benchmark × column | `value`; `level` is `sub`, `group`, `overall` or `standalone`; `parent` names the group; `position` is the sheet order |
| `criteria` | benchmark × selected model × task × metric | `value` or `value_text`, `reason` when blank, `source` file |
| `criteria_inputs` | benchmark × model with criteria | source file, rows, scorer, price source |
| `provenance` | row of a workbook's `Provenance` sheet | where a column's numbers came from, and its caveats |
| `sources` · `meta` | input file · build fact | md5 of each input, which `--check` compares |

Views: `final_scores` (selected models only, the numbers to report) and `gaps` (criteria with no
number, and why).

## Reading it correctly

- **NULL is "no usable number", never a zero.** `Awareness`/`Qwen` perspective is a real 0.0.
- **`scores` holds every model.** Use `final_scores` for anything reported.
- **`criteria` holds the selected models only**, and `task = 'ALL'` is the benchmark's row: tasks
  pooled item-weighted, McNemar on the summed counts. Other `task` values are one task's own numbers,
  on the repeat subsample where the metric comes from repeats.
- **`level` matters when averaging.** A `group` or `overall` row is already an aggregate of `sub` rows.
- **`standalone` is never averaged.** `All_EM` is the one such row, so NegotiationToM has no `overall` row ([why](benchmarks/interpersonal/negotiationtom.md)).
- **`value_text`** holds a non-numeric answer such as `not reported`.

```sql
-- one model across all six criteria on one benchmark
SELECT criterion_no, metric, value FROM criteria
WHERE benchmark = 'NegotiationToM' AND model_column = 'Gemini' AND task = 'ALL' ORDER BY 1, 2;

-- what is still missing
SELECT benchmark, model_column, criterion, reason FROM gaps ORDER BY 1, 2;
```

Criteria definitions: [evaluation-criteria.md](evaluation-criteria.md) and
[criteria-runs.md](criteria-runs.md).
