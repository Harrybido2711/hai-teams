"""Wonderbread — shared core for the judge pilot: the 10% samples, the human-labelled anchor sets,
and upstream's own message builders, judge prompts and scorers. Identical for every judge.

Scope: the two judged subtasks (LLM_as_judge/JUDGE_SELECTION.md §7.3-7.5) — Question Answering (Judge 1:
four 1-3 scores, 1 = BEST) and SOP Generation (Judge 2: line-to-line matching, tallied into
precision / recall / ordering). SOP Improvement's scorer does not execute (§7.5) and is out.

**Nothing upstream is re-typed.** Every prompt, message builder and scorer is imported from the
vendored package, and generation runs upstream's own `generate_response` / `run` functions with
only their transport (`_fetch_completion`) replaced — so what the model sees is what upstream built.
`wonderbread/helpers.py` imports anthropic, google-generativeai, pydrive2 and together at module
level; the functions used here call none of them, so a missing one is replaced by a stand-in that
raises if anything ever touches it (`_absent`). Quest's env lacks anthropic and pydrive2.

Declared deviations from upstream (LLM_as_judge/JUDGE_SELECTION.md §7.3-7.4):
- judges run at temperature 0 where the route allows it — upstream's SOP judge ran at the API
  default 1.0 (D3); 3 replicates, not 1;
- QA scores are parsed into {1,2,3} with a parse-fail status — upstream stored the raw string;
- SOP indices are parsed from JSON with fences stripped (Claude fences JSON mode on OpenRouter;
  upstream's bare json.loads would raise) and an index outside the list is a parse failure, not a
  match;
- `preprocess_sop`'s strip-up-to-the-first-"." (B4) is KEPT, for comparability;
- the tested model is gpt-5.6-luna at its settled config, temperature unsettable — upstream's
  GPT-4 ran at temperature 0.
"""

import collections
import importlib
import json
import os
import re
import sys
import types

WB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(WB_ROOT))
import judge_core as jc  # noqa: E402

DATA_DIR = os.path.join(WB_ROOT, "data")
# upstream's own location for demos (df_rankings.csv: path_to_demo_folder = data/demos/<folder>).
# Filled by fetch_sample_demos.py with only the sampled demos; gitignored.
DEMOS_DIR = os.path.join(DATA_DIR, "demos")
QA_CSV = os.path.join(DATA_DIR, "qa_dataset.csv")
RANKINGS_CSV = os.path.join(DATA_DIR, "df_rankings.csv")
_HC = os.path.join(DATA_DIR, "experimental_results")
ANCHOR_QA_HUMAN = os.path.join(_HC, "question_answering", "human_correlation", "qna-human-sample-final.csv")
ANCHOR_QA_GPT4 = os.path.join(_HC, "question_answering", "human_correlation", "qna-GPT4-sample-final.csv")
ANCHOR_SOP = os.path.join(_HC, "sop_generation", "human_correlation", "sample_human_eval.csv")
SAMPLE_PATH = os.path.join(WB_ROOT, "sample_uids.json")
ENV_PATH = os.path.join(WB_ROOT, ".env")

EXPECTED_QA = 120
EXPECTED_RANK1 = 162
EXPECTED_ANCHOR = 30
QA_CRITERIA = ("completeness", "soundness", "clarity", "compactness")
WITH_REFERENCE = ("completeness", "soundness")        # the two that see the Human Label (A2)
# upstream run_experiments.py for SOP generation: three non-pairwise ablations
SOP_ABLATIONS = {"td": dict(is_td=True, is_kf=False, is_act=False),
                 "td_kf": dict(is_td=True, is_kf=True, is_act=False),
                 "td_kf_act": dict(is_td=True, is_kf=True, is_act=True)}
HEADER_DEMARCATOR = "----------------------------------------"   # collect_results.py
SCORER_VERSION = "wb_upstream_v1"
ANCHOR_PRODUCER = "anchor-human-sample"   # upstream's GPT4 / GeminiPro / Claude3 responses


# ---------------------------------------------------------------- upstream, imported


def _absent(name):
    module = types.ModuleType(name)
    module.__path__ = []                       # importable as a package, so `a.b` resolves
    module.__absent__ = True                   # judge_core's clients check this: never a real client

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def unavailable(*args, **kwargs):
            raise RuntimeError(f"{name}.{attr} is not installed here, and the judge pilot never "
                               f"calls it — something reached an upstream path it should not")
        return unavailable
    module.__getattr__ = __getattr__
    return module


_UPSTREAM_CLIENT_LIBS = ("anthropic", "openai", "together", "google", "google.generativeai",
                         "pydrive2", "pydrive2.auth", "pydrive2.drive")
_upstream = {}


def upstream():
    """The vendored modules this pilot uses, imported once."""
    if _upstream:
        return types.SimpleNamespace(**_upstream)
    stubbed = []
    for name in _UPSTREAM_CLIENT_LIBS:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = _absent(name)
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, sys.modules[name])
            stubbed.append(name)
    if WB_ROOT not in sys.path:
        sys.path.insert(0, WB_ROOT)
    base = "wonderbread.benchmark.tasks"
    _upstream.update(
        helpers=importlib.import_module("wonderbread.helpers"),
        qa_prompts=importlib.import_module(f"{base}.knowledge_transfer.question_answering.prompts"),
        qa_gen=importlib.import_module(f"{base}.knowledge_transfer.question_answering.generate_responses"),
        sop_main=importlib.import_module(f"{base}.documentation.sop_generation.main"),
        sop_eval=importlib.import_module(f"{base}.documentation.sop_generation.eval"),
        eval_prompts=importlib.import_module(f"{base}.documentation.sop_generation.eval_prompts"),
        metrics=importlib.import_module(f"{base}.documentation.sop_generation.metrics"),
        stubbed=stubbed,
    )
    return types.SimpleNamespace(**_upstream)


# ---------------------------------------------------------------- data and the samples


def _pd():
    import pandas as pd
    return pd


def load_qa():
    df = _pd().read_csv(QA_CSV)
    if len(df) != EXPECTED_QA:
        raise SystemExit(f"{QA_CSV}: expected {EXPECTED_QA} rows, found {len(df)}")
    return {f"qa{i:03d}": row for i, row in df.iterrows()}


def task_site(task_id):
    task = upstream().helpers.get_webarena_task_json(int(task_id))
    if task is None:
        raise SystemExit(f"no WebArena task json for task_id {task_id}")
    return task["sites"][0]


def load_rank1():
    """The 162 rank-1 gold demos SOP Generation runs on (`run_experiment(is_use_rank_1_df=True)`)."""
    df = _pd().read_csv(RANKINGS_CSV)
    df = df[df["rank"] == 1]
    if len(df) != EXPECTED_RANK1:
        raise SystemExit(f"{RANKINGS_CSV}: expected {EXPECTED_RANK1} rank-1 demos, found {len(df)}")
    return {row["folder_name"]: int(row["task_id"]) for _, row in df.iterrows()}


def build_sample():
    """QA stratified by question template (11 strata), SOP Generation by website (4) — 10%, seeded,
    drawn once and committed."""
    qa = load_qa()
    qa_pick = jc.stratified_sample(list(qa), lambda u: qa[u]["Question Template"])
    qa_sizes = collections.Counter(r["Question Template"] for r in qa.values())
    rank1 = load_rank1()
    site = {demo: task_site(t) for demo, t in rank1.items()}
    sop_pick = jc.stratified_sample(list(rank1), site.get)
    sop_sizes = collections.Counter(site.values())
    doc = {"seed": jc.SAMPLE_SEED, "fraction": jc.SAMPLE_FRACTION,
           "qa": {"note": "uid = qa<row index in data/qa_dataset.csv>; stratum = Question Template",
                  "strata": {k: {"population": qa_sizes[k], "sampled": len(v), "uids": v}
                             for k, v in qa_pick.items()}},
           "sop": {"note": "uid = rank-1 demo folder (data/df_rankings.csv); stratum = WebArena site",
                   "strata": {k: {"population": sop_sizes[k], "sampled": len(v), "uids": v}
                              for k, v in sop_pick.items()}}}
    with open(SAMPLE_PATH, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return doc


def load_sample(task):
    if not os.path.exists(SAMPLE_PATH):
        raise SystemExit(f"{SAMPLE_PATH} missing — run `python wb_eval_core.py --build-sample` once")
    with open(SAMPLE_PATH) as fh:
        strata = json.load(fh)[task]["strata"]
    return [u for k in sorted(strata) for u in strata[k]["uids"]]


def demos_needed():
    """Every demo folder the sample touches: QA items' Task ID(s) plus the SOP demos."""
    qa = load_qa()
    need = set(load_sample("sop"))
    for uid in load_sample("qa"):
        need.update(x.strip() for x in str(qa[uid]["Task ID(s)"]).split(","))
    return sorted(need)


def check_demo(folder):
    """What upstream's path helpers assert, checked up front so a gap fails before any call."""
    h = upstream().helpers
    path = os.path.join(DEMOS_DIR, folder)
    problems = []
    if not os.path.isdir(path):
        return [f"{folder}: not on disk under {DEMOS_DIR}"]
    if h.get_path_to_sop_txt(path, is_no_assert=True) is None:
        problems.append(f"{folder}: no SOP*.txt")
    if h.get_path_to_trace_json(path, is_no_assert=True) is None:
        problems.append(f"{folder}: no trace .json")
    shots = os.path.join(path, "screenshots")
    if not os.path.isdir(shots) or not os.listdir(shots):
        problems.append(f"{folder}: no screenshots/")
    return problems


# ---------------------------------------------------------------- anchor sets (human-labelled)


def load_anchor_qa():
    """Upstream's 30-item human sample: responses from its own models, scored by humans; its GPT-4
    judge's scores on the same rows sit beside them (LLM_as_judge/JUDGE_SELECTION.md §7.3, A3)."""
    pd = _pd()
    human = pd.read_csv(ANCHOR_QA_HUMAN)
    gpt4 = pd.read_csv(ANCHOR_QA_GPT4)
    if len(human) != EXPECTED_ANCHOR or len(gpt4) != EXPECTED_ANCHOR:
        raise SystemExit(f"anchor QA: expected {EXPECTED_ANCHOR} rows in each file")
    if not (human["Response"].fillna("") == gpt4["Response"].fillna("")).all():
        raise SystemExit("anchor QA: the human and GPT-4 files do not hold the same responses in order")
    out = {}
    for i, row in human.iterrows():
        out[f"anchor{i:02d}"] = {
            "question": row["Question Instantiation"], "human_label": row["Human Label"],
            "response": row["Response"], "template": row["Question Template"],
            "human": {c: row[f"{c}_score"] for c in QA_CRITERIA},
            "upstream_gpt4": {c: gpt4.loc[i, f"{c}_score"] for c in QA_CRITERIA}}
    return out


def load_anchor_sop():
    """Upstream's 30-pair human sample: a generated SOP, its gold SOP (already through
    add_gold_sops), and human precision/recall beside GPT-4's (`auto_eval_*`)."""
    df = _pd().read_csv(ANCHOR_SOP)
    if len(df) != EXPECTED_ANCHOR:
        raise SystemExit(f"{ANCHOR_SOP}: expected {EXPECTED_ANCHOR} rows, found {len(df)}")
    return {f"anchor{i:02d}": {
        "demo": row["demo_name"], "ablation": row["ablation"], "pred_sop": row["pred_sop"],
        "gold_sop": row["gold_sop"],
        "human": {"precision": row["manual_eval_precision"], "recall": row["manual_eval_recall"]},
        "upstream_gpt4": {"precision": row["auto_eval_precision"], "recall": row["auto_eval_recall"],
                          "ordering": row["ordering"]}} for i, row in df.iterrows()}


# ---------------------------------------------------------------- generation: upstream builders


def gold_sop(demo):
    """upstream eval.add_gold_sops for one demo: the SOP file, title line dropped."""
    df = _pd().DataFrame([{"demo_name": demo}])
    return upstream().sop_eval.add_gold_sops(df, DEMOS_DIR)["gold_sop"].iloc[0]


def generated_sop(path_to_file):
    """collect_results.py: the SOP is everything after the header demarcator."""
    with open(path_to_file) as fh:
        contents = fh.read()
    return contents[contents.index(HEADER_DEMARCATOR) + len(HEADER_DEMARCATOR):].strip()


def count_images(messages):
    return sum(1 for m in messages for c in (m["content"] if isinstance(m["content"], list) else [])
               if c.get("type") == "image_url")


# ---------------------------------------------------------------- judging: upstream prompts


def qa_judge_messages(criterion, question, human_label, response):
    """eval.py get_<criterion>_score → _fetch_completion(…, 'GPT4') → _fetch_openai_completion,
    which prepends SYSTEM_PROMPT. Same message shapes as upstream sent."""
    up = upstream()
    fn = getattr(up.qa_prompts, f"prompt__{criterion}_score")
    prompt = fn(question, human_label, response) if criterion in WITH_REFERENCE else fn(question, response)
    return [{"role": "system", "content": up.helpers.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}]


def qa_skip(response):
    """Upstream writes "NA" without calling the judge when the response is NaN or < 5 characters."""
    return response is None or (isinstance(response, float) and response != response) \
        or len(str(response)) < 5


_SCORE_LEAD = re.compile(r"^\W*(?:score|rating)?\W*([123])(?!\d)", re.IGNORECASE)
_INTEGER = re.compile(r"(?<![\d.])\d+(?!\d|\.\d)")      # "1." is the integer 1; "1.5" is not


def parse_qa_score(text):
    """1, 2 or 3 — or None. Upstream asks for the bare number. Accepted: the number leading the
    answer with no rival on its first line ("2", "**2**", "Score: 2", "2 - mostly complete",
    "Rating: 2" then an explanation); otherwise exactly one integer in the whole answer, and it in
    range. Two candidates ("1 or 2") is ambiguous, never a guess."""
    t = jc.strip_fences(text).strip()
    m = _SCORE_LEAD.match(t)
    if m and len(_INTEGER.findall(t.split("\n", 1)[0])) == 1:
        return int(m.group(1))
    numbers = _INTEGER.findall(t)
    if len(numbers) == 1 and numbers[0] in ("1", "2", "3"):
        return int(numbers[0])
    return None


def sop_lines(sop):
    return upstream().sop_eval.preprocess_sop(sop)


def sop_judge_messages(query_line, list_of_lines):
    """metrics.PairwiseComparison → eval_completion.get_completion → helpers.fetch_openai_json_completion:
    SYSTEM_PROMPT, then the prompt as a plain-string user turn, JSON mode."""
    up = upstream()
    prompt = up.eval_prompts.map_query_to_one_prompt(query_line, list_of_lines)
    return [{"role": "system", "content": up.helpers.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


SOP_RESPONSE_FORMAT = {"type": "json_object"}


def make_index_parser(n_lines):
    def parse(text):
        obj = jc.first_json_object(text)
        if obj is None or "index" not in obj:
            return None
        try:
            value = float(obj["index"])
        except (TypeError, ValueError):
            return None
        if value != int(value):
            return None
        index = int(value)
        return index if index == -1 or 0 <= index < n_lines else None
    return parse


def sop_metrics(pred_indices, gold_indices):
    """Upstream's own precision / recall / ordering (metrics.PairwiseComparison), fed our parsed
    indices instead of fetching; F1 as run_sop_generation.py computes it. None where upstream
    would divide by zero."""
    if not pred_indices or not gold_indices or None in pred_indices or None in gold_indices:
        return None
    PC = upstream().metrics.PairwiseComparison
    pc = PC.__new__(PC)                            # __init__ would call the API
    pc.pred_sop_len, pc.ref_sop_len = len(pred_indices), len(gold_indices)
    pc.completions = {**{f"pred_line_{i}": {"index": v} for i, v in enumerate(pred_indices)},
                      **{f"gold_line_{j}": {"index": v} for j, v in enumerate(gold_indices)}}
    p, r, o = pc.precision(), pc.recall(), pc.ordering()
    return {"precision": p, "recall": r, "ordering": o,
            "f1": (2 * p * r / (p + r)) if (p + r) else None,
            "n_lines_pred_sop": len(pred_indices), "n_lines_gold_sop": len(gold_indices)}


if __name__ == "__main__":
    if "--build-sample" in sys.argv:
        if os.path.exists(SAMPLE_PATH) and "--force" not in sys.argv:
            raise SystemExit(f"{SAMPLE_PATH} exists; the sample is drawn once. --force to redraw.")
        doc = build_sample()
        for task in ("qa", "sop"):
            for k, v in doc[task]["strata"].items():
                print(f"{task:3s} {v['sampled']:2d} of {v['population']:3d}  {k[:80]}")
        print(f"demos needed: {len(demos_needed())}")
    elif "--demos-needed" in sys.argv:
        print("\n".join(demos_needed()))
    else:
        print(__doc__)
