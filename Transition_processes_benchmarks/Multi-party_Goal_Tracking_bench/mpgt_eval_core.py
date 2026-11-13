"""mpgt-eval — shared core: data, upstream's six prompts, the generation loop every tested model runs
(`run_cli`), and the one lenient scorer. Identical for every model; each runner in `MPGT_<Model>/`
holds only its client and parameters (`judge_core.Generator`).

What upstream does, and what this keeps (`.claude/references/benchmarks/transition/mpgt.md`):
- condition: **zero-shot**, DST-only and goal-only — the user's choice for the first run,
  2026-11-09. Data are the files upstream's two zero-shot scripts read, 29 dialogues each;
- prompt: upstream's six `generate_*_prompt(src, few)` builders, taken from the vendored scripts'
  source (`_upstream_builders`) so not a character can differ, sent as one user turn with no system
  prompt — as upstream's `predict` does. **All six styles run; the headline is their mean** (the
  user, 2026-11-09);
- decoding: each model's settled config, not upstream's `gpt-3.5-turbo` at temperature 0.4 — a
  declared deviation, the same one every benchmark here makes;
- scoring: upstream ships no scorer, and the paper's metric — exact match on the gold `[MASK]`
  strings with the misses reviewed by a person — is replaced by **one automatic lenient matcher**
  (the user, 2026-11-09). It is the project's rule: one scorer per benchmark, the lenient one.

**The call layer is `Action_processes_benchmarks/judge_core.py`**, imported rather than ported a
fourth time: halts, the SIGALRM watchdog, retries on an empty body, per-row usage, parameter
negotiation, the one-writer checkpoint. **A sync of this benchmark to Quest therefore includes
`judge_core.py`**, and a change to it is a change to MultiChallenge, Wonderbread and this.

    python mpgt_eval_core.py --summarize MPGT_*/results/zero-shot/*.jsonl   # score stored rows
"""

import argparse
import ast
import collections
import difflib
import filecmp
import glob
import json
import os
import re
import shutil
import sys
import time

MPGT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(MPGT_ROOT)),
                                "Action_processes_benchmarks"))
import judge_core as jc  # noqa: E402

ENV_PATH = os.path.join(MPGT_ROOT, ".env")
CONDITION = "zero-shot"
DATA_DIR = os.path.join(MPGT_ROOT, "Preprocessing", "data", "split-diogs", "zero-shot")

# Verified 2026-11-09 from the local tree (mpgt.md § Expected counts). A different count means the
# data or the slot extraction changed — stop, do not score.
TASKS = {
    "dst": {"data": os.path.join(DATA_DIR, "dst-only", "dst-only.jsonl"),
            "script": os.path.join(MPGT_ROOT, "gpt", "zero_shot_dst_only_gpt.py"),
            "marker": "#", "rows": 29, "slots": 389},
    "goal": {"data": os.path.join(DATA_DIR, "goal-only", "goals-only.jsonl"),
             "script": os.path.join(MPGT_ROOT, "gpt", "zero_shot_goal_only_gpt.py"),
             "marker": "@", "rows": 29, "slots": 225},
}
# style -> upstream builder; upstream's `predict` maps the same six names to the same six functions
STYLES = {"basic": "generate_basic_prompt", "specific": "generate_specific_prompt",
          "story": "generate_story_prompt", "roleplay": "generate_roleplay_prompt",
          "annotation": "generate_annotation_instructions_prompt",
          "reasoning": "generate_reasoning_prompt"}
SPEAKERS = ("ari", "rp", "rc", "lc", "lp")     # every speaker tag in both files, 2026-11-09
SCORER_VERSION = "mpgt_lenient_v4"   # 2026-11-11: v2 read by turns, v3 three readings, v4 dialogue-first


# ---------------------------------------------------------------- data and prompts


def _upstream_builders(script):
    """The six `generate_*_prompt` functions from a vendored script, by parsing its source — importing
    it would run upstream's whole experiment at module level (and needs openai<1.0 and pandas).
    Only those six function definitions are compiled; each is a string concatenation."""
    with open(script) as fh:
        tree = ast.parse(fh.read())
    wanted = set(STYLES.values())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in defs} != wanted:
        raise SystemExit(f"{script}: expected builders {sorted(wanted)}, found "
                         f"{sorted(n.name for n in defs)}")
    namespace = {}
    exec(compile(ast.Module(body=defs, type_ignores=[]), script, "exec"), namespace)
    return {style: namespace[name] for style, name in STYLES.items()}


_BUILDERS = {}


def build_prompt(task, style, src):
    """Exactly what upstream's zero-shot `predict` sends for this style: `few` is False."""
    if task not in _BUILDERS:
        _BUILDERS[task] = _upstream_builders(TASKS[task]["script"])
    return _BUILDERS[task][style](src, False)


def load_rows(task):
    """The task's dialogues in file order, each with its gold annotations — counts checked against
    TASKS, and every gold must start with the task's marker."""
    spec = TASKS[task]
    with open(spec["data"]) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for idx, row in enumerate(rows):
        row["idx"] = idx
        row["gold"] = gold_slots(row)
    slots = sum(len(r["gold"]) for r in rows)
    if len(rows) != spec["rows"] or slots != spec["slots"]:
        raise SystemExit(f"{spec['data']}: expected {spec['rows']} rows / {spec['slots']} [MASK] "
                         f"slots, found {len(rows)} / {slots}")
    bad = [(r["idx"], g[:3]) for r in rows for g in r["gold"] if not g or g[0] != spec["marker"]]
    if bad:
        raise SystemExit(f"{spec['data']}: gold slots not led by {spec['marker']!r}: {bad[:5]}")
    return rows


def units(tasks, styles, limit=0):
    """(uid, task, style, row) in run order: task, then dialogue, then style — so `--limit N` is N
    dialogues under every style."""
    for task in tasks:
        rows = load_rows(task)
        for row in rows[:limit] if limit else rows:
            for style in styles:
                yield f"{task}/{style}/{row['idx']:02d}", task, style, row


# ---------------------------------------------------------------- the lenient scorer
# Upstream masks with `dst_mask` / `goal_mask` (Preprocessing/dst_and_goal_masking.py): inside the
# [start]...[end] window, each turn's annotation — from its '#' (DST) or '@' (goal) to the turn's
# end — becomes one [MASK]; the utterances are untouched, and `tgt` is the window unmasked. So a
# token alignment of window against `tgt` gives every gold slot exactly (all 614 checked).
#
# A RESPONSE is read dialogue-first (v4, 2026-11-11, after five reviews: A1–A6, F1–F6, G1–G6,
# H1–H6, R1–R5 — the last one safe-to-run, with the limits in mpgt.md § Traps):
#   1. turns — copies of the window among the response's turns (`_copy`: the window's turns matched
#      in order to response turns of the same speaker, as one contiguous stretch). A stretch the
#      context explains as well, or a turn whose neighbours are echoed context, is the context
#      echoed and never read (`_by_turns`). An annotation right after a turn's utterance is a
#      STRONG answer (`_answer`); copies covering at least half the masked turns are read in order,
#      a later one's strong answers replacing an earlier one's (counted as `revised`; a later blank
#      replaces nothing). An annotation found after prose is WEAK, looked for only where the turn
#      opens its own line; a turn quoted inside prose answers nothing;
#   2. fillers — weak answers, smaller copies, then a numbered LIST (`_listed`: the last run counted
#      from 1 with one item per slot, at least half answered) or, where nothing is numbered, a ROW
#      (markers outside the dialogue's turns, each shaped like the scheme, exactly one per slot)
#      only fill slots still unread. A filler that disagrees with a kept answer is a `conflict`,
#      never applied; a list or row with any conflict fills nothing (`_predict`).
# An annotation is read by its grammar (`_read`), with any other joined to it in the same paragraph
# as their union (`# x or # y`); two with words between them (`# x rather than # y`) are unread.

TOKEN = re.compile(r"\n[ \t\r]*\n|\n|\[MASK\]|\[start\]|\[end\]|[A-Za-z0-9_'\-]+|[^\sA-Za-z0-9_'\-]")
PARA = "[para]"                      # a blank line: an answer never runs across one
NL = "[nl]"                          # a line break: an item never runs across one
REF = "[ref]"                        # `#3`, `@2`: a numbered reference, skipped
MARKERS = ("#", "@")
_THINK_END = re.compile(r"</think>", re.IGNORECASE)
_SKIP = {"*", "`", '"', ".", NL, REF, "[start]", "[end]", "[mask]"}  # never part of an answer
_JOIN = {";", ",", "and", "or", "/", "&"}                       # items of ONE answer, a hedge too
_STOP_OPEN = {'"', "`", "|", "[", "]", "{", "}", PARA, NL, "[start]", "[end]"}  # a line, string,
#                                                     code span, bracket or table cell ended
_BARE_END = ({None, PARA, NL, "[start]", "[end]", "[mask]", "`", "*", '"', ".", "]", ")", "}", "|"}
             | _JOIN | set(MARKERS))                          # what may follow a bare `# thank`
_BARE_JOINED_END = _BARE_END - {"."} - set(MARKERS)         # ... a bare `bye` in `# x ( ) ; bye`
_TAG_DECORATION = {"*", "]", "`", '"'}                          # `**ARI**:`, `[ARI]:`
_IN_BRACKETS = {"(", ")", ";", ",", "+", "*"} | _SKIP | set(MARKERS)  # every sign a gold bracket holds
_NAME = re.compile(r"[a-z0-9_'\-]*[a-z][a-z0-9_'\-]*$")      # an item's name has a letter
_WORDY = re.compile(r"[a-z0-9]")
_ITEM = re.compile(r"(?m)^[ \t]*(?:[*_]{1,2})?(?:(\d+)[.):]|\[mask\][ \t]*(\d+))", re.IGNORECASE)
MATCH_MIN = 0.5      # share of a source utterance's words a response turn must hold to be that turn
GAP = 0.2            # cost of a response turn skipped inside one copy of the window
SKIP_COST = 0.01     # cost of a window turn the copy leaves out: of two turns that read the same
#                      (ARI's `No .` twice), the one in sequence is matched
MAX_COPIES = 3       # copies of the window looked for in one response
CHECK_SHARE = 0.2    # a style with more of its slots unread than this is flagged by the summary


def _tokens_at(text):
    """(tokens, character offset of each). Curly apostrophes are straightened first."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    toks, offs = [], []
    for mt in TOKEN.finditer(text):
        t = mt.group(0)
        if t.isdigit() and toks and toks[-1] in MARKERS and offs[-1] == mt.start() - 1:
            toks[-1] = REF                       # `Mask #3`: a number, not an annotation's marker
            continue
        toks.append((PARA if t.strip(" \t\r") != "\n" else NL) if t.startswith("\n") else t.lower())
        offs.append(mt.start())
    return toks, offs


def tokens(text):
    return _tokens_at(text)[0]


def window(src):
    start, end = src.find("[start]"), src.find("[end]")
    if start < 0 or end <= start:
        raise ValueError("no [start]...[end] window")
    return src[start + len("[start]"):end]


def _align(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def gold_slots(row):
    a, b = tokens(window(row["src"])), tokens(row["tgt"])
    gold = []
    for op, i1, i2, j1, j2 in _align(a, b):
        if "[mask]" in a[i1:i2]:
            if op != "replace" or a[i1:i2] != ["[mask]"]:
                raise SystemExit(f"gold alignment: {op} over {a[i1:i2][:6]} — the data are not "
                                 f"what dst_and_goal_masking.py writes")
            gold.append(b[j1:j2])
    return gold


# ---- reading one annotation

def _norm(name):
    return name.replace("_", "").replace("-", "")


def _read(toks, start=0, one=False):
    """(raw, items, end) for the annotation at toks[start:], read by its grammar: items
    `name ( ... )` joined by `;`, `,`, `and`, `or`, `/`, a marker, or nothing (up to two filler
    words after a joiner: `, and also x ( )`). A bare `name` is an item only right after a marker,
    or right after a joiner where the annotation then ends (`# thank`, `# x ( ) ; bye`) — never a
    word of prose (`here the`, `hmm ,`, `so # x`, `, probably .`, `, I think .`); with neither marker
    nor brackets, only a name of the annotation scheme standing alone (`-> affirm`, `_vocab`) is
    read. Skipped: decoration, a left-over [MASK], markers, a `)` with
    nothing open, labels (`Intent:`, `"2":`), and before anything is read, arrows and brackets. An
    item left
    open is closed at the end, a marker, a line break, or where a string, code span, bracket or
    table cell ends. `one`: a second marker ends it. `raw` is the tokens read, for `exact`."""
    raw, items, cur, depth, state = [], [], [], 0, "start"     # at depth 0: start | name | closed
    marked = filler = 0     # a marker came just before; words skipped since the last joiner
    j, n = start, len(toks)
    while j < n:
        t = toks[j]
        nxt = toks[j + 1] if j + 1 < n else None
        if depth:
            if (t not in _JOIN and _NAME.match(t) and nxt == "."
                    or not _WORDY.search(t) and t not in _IN_BRACKETS):
                # inside a gold bracket there is never a word before a full stop, nor any sign but
                # `( ) ; , +`: prose after an item left open (`, probably .`, `— maybe`) — close
                # it there, and read on as after a closed item, so `, and also x ( )` still adds
                # its item. Words in a row are NOT prose here: `closing time` is `closing_time`
                joined = bool(cur) and cur[-1] in _JOIN
                while cur and cur[-1] in _JOIN:
                    cur.pop()
                    raw.pop()
                items.append(cur + [")"] * depth)
                cur, depth, state, filler = [], 0, "start" if joined else "closed", 0
                continue
            if t in _STOP_OPEN or t in MARKERS:
                items.append(cur + [")"] * depth)
                cur, depth, state = [], 0, "closed"
                if t not in MARKERS:
                    break
            else:
                if t not in _SKIP:
                    cur.append(t)
                    raw.append(t)
                    depth += (t == "(") - (t == ")")
                    if not depth:
                        items.append(cur)
                        cur, state = [], "closed"
                j += 1
                continue
        if t == PARA:
            break
        if t in MARKERS:
            if one and raw:
                break
            if state == "name":
                items.append(cur)
                cur = []
            if raw and raw[-1] not in _JOIN:
                raw.append(";")                  # `# thank # bye` reads again as `thank ; bye`
            state, marked, filler = "start", 1, 0
            j += 1
            continue
        if t == "." and (state == "name" or state == "closed" and nxt and _NAME.match(nxt)
                         and _norm(nxt) in _vocab()):
            # `# thank . bye`, `# inform ( x ) . offer-help`: a full stop before another name of
            # the scheme joins, as `;` would
            if state == "name":
                items.append(cur)
            cur, state, filler = [], "start", 0
            raw.append(";")
            j += 1
            continue
        if t in _SKIP or t == ")":
            j += 1
            continue
        if t in _JOIN:
            if state == "name":
                items.append(cur)
                cur = []
            if raw:
                raw.append(t)
            state, filler = "start", 0
        elif state == "name" and t == "(":
            cur.append(t)
            raw.append(t)
            depth = 1
        else:
            if state == "name":                  # a bare name, closed by what follows it
                items.append(cur)
                cur, state = [], "closed"
            sig = next((x for x in toks[j + 1:j + 4] if x not in ('"', "`", "*")), None)
            span = next((w for w in (3, 2) if j + w <= n and all(_NAME.match(x) for x in toks[j:j + w])
                         and _norm("".join(toks[j:j + w])) in _vocab()
                         and (toks[j + w] if j + w < n else None) in _BARE_END | {"("}), 0)
            if span and state in ("start", "closed"):
                # a name of the scheme written with spaces (`offer help ( )`, `r repet`): one name
                cur, state = list(toks[j:j + span]), "name"
                raw.extend(cur)
                j += span
                marked = 0
                continue
            if _NAME.match(t) and sig == ":":    # a label
                j += 1
            elif _NAME.match(t) and (nxt == "(" or state == "start" and (
                    marked and nxt in _BARE_END
                    or raw and not marked and not filler and (
                        nxt in _BARE_JOINED_END or nxt == "." and _norm(t) in _vocab())
                    or not raw and not marked and nxt in _BARE_END and nxt not in MARKERS
                    and _norm(t) in _vocab())):
                cur, state = [t], "name"
                raw.append(t)
            elif not raw and not _WORDY.search(t) and t != "|" and (t != "(" or nxt in MARKERS):
                pass                             # `->`, `—`, `[`, `(` before the answer
            elif raw and state == "start" and _NAME.match(t) and filler < 2:
                filler += 1                      # `, and also x ( )`, `or possibly x ( )`
            else:
                break
        marked = 0
        j += 1
    if depth:
        items.append(cur + [")"] * depth)       # left open, as nine gold annotations are
    elif state == "name":
        items.append(cur)
    return (raw, items, j) if items else None


def read_annotation(toks, one=False):
    """(raw, items) for the annotation at the head of `toks`, or None — see `_read`."""
    got = _read(toks, 0, one)
    return (got[0], got[1]) if got else None


def _answer(toks, marker, rescue=False):
    """One turn's (or one list item's) answer: (raw, items), or None. The annotation at its head —
    or, with `rescue` (a list item, which is about one slot), at the paragraph's first marker when
    prose comes first — and every other annotation in the same paragraph joined to it (`# x or # y`,
    `# x # y`) as their union, so a hedge is never the answer. Two annotations with WORDS between
    them (`# x (or possibly # y)`, `# x rather than # y`, `# x — not # y`) are a hedge or a
    rejection the reader cannot settle: no answer, so the summary's CHECK sees it. A blank line
    before anything is read means no answer, unless the next paragraph opens with the marker."""
    n, j = len(toks), 0
    while j < n and toks[j] in _SKIP:
        j += 1
    if j < n and toks[j] == PARA:
        while j < n and (toks[j] == PARA or toks[j] in ("*", "`", '"')):
            j += 1
        if j + 1 < n and _NAME.match(toks[j]) and toks[j + 1] == ":":     # `Annotation:`
            j += 2
            while j < n and toks[j] in ("*", "`", '"'):
                j += 1
        if j >= n or toks[j] != marker:
            return None
    end = next((q for q in range(j, n) if toks[q] in (PARA, "[end]", "[start]")), n)
    got = _read(toks[:end], j)
    if got is None:
        if not rescue:
            return None
        first = next((q for q in range(j, end) if toks[q] == marker), None)
        got = _read(toks[:end], first) if first is not None else None
        if got is None:
            return None
    raw, items, k = list(got[0]), list(got[1]), got[2]
    while True:
        q = next((q for q in range(k, end) if toks[q] == marker), None)
        if q is None:
            break
        if any(_NAME.match(t) and t not in _JOIN for t in toks[k:q]):
            return None
        more = _read(toks[:end], q, one=True)
        if more:
            raw += [";"] + more[0]
            items += more[1]
        k = max(more[2] if more else 0, q + 1)
    return raw, items


_PLUS = re.compile(r"[a-z0-9]+(?:\+[a-z0-9]+)+")


_SPK_PAIR = re.compile(r"(?<![a-z])(ari|rp|rc|lc|lp)_?(?=(ari|rp|rc|lc|lp)(?![a-z]))")


def _item_key(item):
    # `lc_rp` and `lc rp` are `LC+RP` (goal 19's gold writes `lc_rp`) — before `_` is dropped
    s = _SPK_PAIR.sub(r"\1+", "".join(item)).replace("_", "").replace("-", "")
    if "(" not in s:
        s += "()"                                # `thank` is `thank ( )`
    else:                                        # a nested `help ( )` is `help`: the gold never
        head, inner = s.split("(", 1)            # writes both, so no two answers are merged
        s = head + "(" + inner[:-1].replace("()", "") + inner[-1:]
    return _PLUS.sub(lambda m: "+".join(sorted(m.group(0).split("+"))), s)


def canonical(ann):
    """One annotation — gold or predicted, a token list — as the set of its items. Lenient about
    form: spacing, case, the marker, `_` against `-`, the order of items and of speakers in
    `LC+RP`, an empty `( )` left off or added, unbalanced brackets (the gold has eleven). Strict
    about content: every item, slot and speaker must match, and an extra item — a second intent,
    or a hedge `X or Y` — makes it a different set."""
    toks = list(ann)
    if not toks or toks[0] not in MARKERS:      # read as an annotation, as it was read the first
        toks = ["#"] + toks                      # time — a leading bare `thank` is not prose here
    got = read_annotation(toks)
    return frozenset(_item_key(i) for i in got[1]) if got else frozenset()


# ---- turns

def _tag_end(b, j):
    """Index after the `:` if a speaker tag starts at b[j] (`ARI :`, `**ARI**:`, `[ARI]:`), else None."""
    if b[j] not in SPEAKERS:
        return None
    k = j + 1
    while k < len(b) and k <= j + 3 and b[k] in _TAG_DECORATION:
        k += 1
    return k + 1 if k < len(b) and b[k] == ":" else None


def _turn_spans(toks):
    """[(speaker, body start, body end)] at every speaker tag; what precedes the first is no turn."""
    tags = [(j, e) for j in range(len(toks)) for e in [_tag_end(toks, j)] if e]
    return [(toks[j], e, tags[i + 1][0] if i + 1 < len(tags) else len(toks))
            for i, (j, e) in enumerate(tags)]


def _split_turns(toks):
    return [(spk, toks[s:e]) for spk, s, e in _turn_spans(toks)]


def _utterance(body):
    """A turn's words before its annotation, or its [MASK]."""
    for k, t in enumerate(body):
        if t in MARKERS or t in ("[mask]", "[start]", "[end]"):
            return body[:k]
    return body


_CLITIC = re.compile(r"^(.*?[a-z])(n't|'[a-z]+)$")


def _words(toks):
    """The words of an utterance, contractions split as the data writes them (`do n't`)."""
    out = collections.Counter()
    for t in toks:
        if t.startswith("[") or not _WORDY.search(t):
            continue
        m = _CLITIC.match(t)
        out.update(m.groups() if m else (t,))
    return out


def _after_utterance(utt, body):
    """What a response turn says after its utterance — where its annotation is."""
    blocks = [b for b in difflib.SequenceMatcher(None, utt, body, autojunk=False)
              .get_matching_blocks() if b.size]
    if not blocks:
        return body
    end = blocks[-1].b + blocks[-1].size
    for k in range(blocks[0].b, end):             # a match that ran into the annotation: cut back
        if body[k] in MARKERS or body[k] == "[mask]":
            return body[k:]
    return body[end:]


def _source_turns(src):
    """Every turn of the dialogue: speaker, utterance, its words, whether it is inside the window,
    and its slot number if it carries a [MASK]."""
    toks = tokens(src)
    start, end = toks.index("[start]"), toks.index("[end]")
    turns, slot = [], 0
    for speaker, s, e in _turn_spans(toks):
        body = toks[s:e]
        utt = _utterance(body)
        words = _words(utt)
        has_slot = "[mask]" in body
        turns.append({"speaker": speaker, "utt": utt, "words": words, "n": sum(words.values()),
                      "window": start < s < end, "slot": slot if has_slot else None})
        slot += has_slot
    return turns, slot


def _share(s, words):
    """How much of source turn s's utterance a response turn holds — the test for being that turn."""
    return sum((s["words"] & words).values()) / s["n"] if s["n"] else 0.0


def _close(s, words):
    """How alike the two utterances are, both ways — to choose between two source turns."""
    m = s["n"] + sum(words.values())
    return 2 * sum((s["words"] & words).values()) / m if m else 0.0


def _copy(W, R, sim, banned):
    """The best local alignment of the window's turns W to the response's turns R outside
    `banned`: matched shares summed, less GAP for every response turn skipped INSIDE the copy and
    SKIP_COST for every window turn left out. [(w, r)] in order, or []."""
    n, m, neg = len(W), len(R), float("-inf")
    M = [[neg] * (m + 1) for _ in range(n + 1)]       # best copy whose last pair is (i-1, j-1)
    fresh = [[False] * (m + 1) for _ in range(n + 1)]
    B = [[neg] * (m + 1) for _ in range(n + 1)]       # best copy within W[:i], R[:j]
    how = [[None] * (m + 1) for _ in range(n + 1)]
    best, at = 0.0, None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim.get((i - 1, j - 1))
            if s is not None and j - 1 not in banned:
                M[i][j] = s + max(0.0, B[i - 1][j - 1])
                fresh[i][j] = B[i - 1][j - 1] <= 0
                if M[i][j] >= best:                    # >=: of equal copies, the later
                    best, at = M[i][j], (i, j)
            B[i][j], how[i][j] = max((M[i][j], 0), (B[i][j - 1] - GAP, 1),
                                     (B[i - 1][j] - SKIP_COST, 2),
                                     key=lambda c: c[0])
    pairs = []
    while at:
        i, j = at
        pairs.append((i - 1, j - 1))
        if fresh[i][j]:
            break
        i, j = i - 1, j - 1
        while how[i][j]:
            i, j = (i, j - 1) if how[i][j] == 1 else (i - 1, j)
        at = (i, j)
    return pairs[::-1]


_LINE_LEAD = {"*", "-", "_", ".", ")", "[", "|", ">", "`", '"'}  # bullets, numbering, quote marks


def _opens_line(toks, body_start):
    """The turn whose body starts at `body_start` opens its line — only bullets, numbering or a
    quote mark before its speaker tag (`- LC :`, `> LC :`, `` `LC : ``) — rather than being quoted
    inside prose (after `LC : ...`, this turn is ...)."""
    j = body_start - 1
    while j >= 0 and toks[j] != ":":
        j -= 1
    j -= 1                                       # the `:`; then the speaker and its decoration
    while j >= 0 and toks[j] in _TAG_DECORATION | set(SPEAKERS):
        j -= 1
    while j >= 0 and toks[j] not in (NL, PARA):
        if toks[j] not in _LINE_LEAD and not toks[j].isdigit():
            return False
        j -= 1
    return True


def _echo(S, R, rw):
    """Response turns that are the CONTEXT echoed: closer to some context turn than to any window
    turn of the same speaker."""
    out = set()
    for j, (spk, _) in enumerate(R):
        ctx = max((_close(s, rw[j]) for s in S if not s["window"] and s["speaker"] == spk), default=0)
        win = max((_close(s, rw[j]) for s in S if s["window"] and s["speaker"] == spk), default=0)
        if ctx >= MATCH_MIN and ctx > win:
            out.add(j)
    return out


def _by_turns(S, k, toks, marker):
    """(copies, dialogue, W): each copy of the window found — (character-order key, {slot: (answer
    or None, strong)}) — see the notes at the head of this section; dialogue is every token inside a
    response turn that is the dialogue (a window turn, or echoed context), where the row reading
    must not look; W the window's turns."""
    W = [s for s in S if s["window"]]
    C = [s for s in S if not s["window"]]
    spans = _turn_spans(toks)
    R = [(spk, toks[a:b]) for spk, a, b in spans]
    rw = [_words(_utterance(body)) for _, body in R]
    echo = _echo(S, R, rw)

    def matches(s, j):
        return 0 <= j < len(R) and R[j][0] == s["speaker"] and _share(s, rw[j]) >= MATCH_MIN

    def among_context(w, j):
        """R[j] is as close to a context turn c as to w, and next to it sits an echo of c's own
        context neighbour — the response dropped w and kept c, which reads the same."""
        for f, c in enumerate(S):
            if c["window"] or c["speaker"] != R[j][0] or _close(c, rw[j]) < _close(w, rw[j]):
                continue
            for d in (-1, 1):
                g, h = f + d, j + d
                if (0 <= g < len(S) and not S[g]["window"] and h in echo
                        and R[h][0] == S[g]["speaker"] and _close(S[g], rw[h]) >= MATCH_MIN):
                    return True
        return False

    sim, csim = {}, {}
    for j in range(len(R)):
        for i, w in enumerate(W):
            if j not in echo and matches(w, j):
                sim[i, j] = _share(w, rw[j])
        for i, c in enumerate(C):
            if matches(c, j):
                csim[i, j] = _share(c, rw[j])
    dialogue = {q for j, (_, a, b) in enumerate(spans)
                if j in echo or any((i, j) in sim for i in range(len(W))) for q in range(a, b)}
    copies, banned = [], set()
    for _ in range(MAX_COPIES):
        pairs = _copy(W, R, sim, banned)
        if not pairs:
            break
        span = set(range(pairs[0][1], pairs[-1][1] + 1))
        banned |= span
        # the dialogue repeats itself ("Thank you .", ARI giving the same directions twice), so
        # echoed context can look like a copy of the window: a stretch the context explains as
        # well is not one
        outside = set(range(len(R))) - span
        if sum(sim[p] for p in pairs) <= sum(csim[p] for p in _copy(C, R, csim, outside)):
            continue
        read = {}
        for i, j in pairs:
            w = W[i]
            if w["slot"] is None or among_context(w, j):
                continue
            # an annotation right after the utterance is a STRONG answer. One found after prose is
            # weak — commentary, a note: it may fill an unread slot but never replace an answer —
            # and is looked for only where the turn opens its own line: a turn quoted inside
            # prose (`after \`LC : ...\`, this turn is # x`) is a reference and answers nothing
            tail = _after_utterance(w["utt"], R[j][1])
            got, strong = _answer(tail, marker), True
            if got is None and _opens_line(toks, spans[j][1]):
                got, strong = _answer(tail, marker, rescue=True), False
            read[w["slot"]] = (got[0] if got else None, strong)
        copies.append((spans[pairs[0][1]][1], read))
    return copies, dialogue, W


# ---- answers not written as the dialogue

_VOCAB = set()


def _vocab():
    """Every top-level name in the gold of both tasks (`affirm`, `request`, `g`, `ag`, ...): an
    item WITHOUT a marker is read as an annotation only when it opens with one of these."""
    if not _VOCAB:
        _VOCAB.add("")                           # building: `_read` must not call back in
        for task in TASKS:
            for row in load_rows(task):
                for g in row["gold"]:
                    _VOCAB.update(_item_key(i).split("(")[0] for i in read_annotation(g)[1])
        _VOCAB.discard("")
    return _VOCAB


_PLACEHOLDERS = {"intent", "intents", "slot", "slots", "value", "values", "act", "dialogueact",
                 "tag", "label", "annotation", "action", "parameter", "parameters", "speaker"}
_GOAL_ITEM = re.compile(r"^(g|ag|cg)\((ari|rp|rc|lc|lp)(\+?(ari|rp|rc|lc|lp))*,")


def _plausible(raw, marker):
    """An answer read OUTSIDE the dialogue is taken only if it has the scheme's shape — every goal
    item `G/AG/CG ( speaker , ... )`; a DST annotation whose first item has brackets or is a name of
    the scheme, and is no placeholder — so a format example in prose (`such as @ G ( speaker ,
    action ( parameters ) )`, `# intent ( slot )`) is not an answer, while an intent the model
    invented (`express-disappointment ( )`) still is: it is scored wrong, not dropped."""
    keys = [_item_key(i) for i in read_annotation(["#"] + list(raw))[1]]
    if marker == "@":
        return bool(keys) and all(_GOAL_ITEM.match(k) for k in keys)
    if not keys:
        return False
    name = keys[0].split("(")[0]
    return name not in _PLACEHOLDERS and ("(" in "".join(raw) or name in _vocab())


def _item_turn_ok(item, i, W):
    """A list item that OPENS with a turn of the window (`3. LC : Thank you . # x`) is about that
    turn: it answers slot i only if that turn is the i-th masked one. Speaker labels with no
    utterance, and turns quoted later in the item's prose, are not checked."""
    q = 0
    while q < len(item) and (item[q] in _SKIP or item[q] in _LINE_LEAD):
        q += 1
    end = _tag_end(item, q) if q < len(item) else None
    if not end:
        return True
    words = _words(_utterance(item[end:]))
    hits = [w for w in W if w["speaker"] == item[q] and _share(w, words) >= MATCH_MIN]
    return not hits or any(w["slot"] == i for w in hits)


def _listed(text, toks, offs, marker, S, k, dialogue, W):
    """(how, [(slot, answer or None)]) when the response lists its answers apart from the dialogue,
    else None.
    - "list": numbered items — the LAST run counted from 1 with exactly k items, at least half of
      them answered. An item's answer is its first annotation (`_answer` with rescue); an item
      without the marker counts only if it opens with a name of the scheme (`_vocab`); an item that
      opens with a window turn answers only for that turn (`_item_turn_ok`).
    - "row": only where nothing is numbered — every marker outside the dialogue's turns, in order,
      that has the scheme's shape (`_plausible`): a table, JSON, answers in a row; exactly k.
    Where every answer names its speaker, the speakers must be the masked turns' in order."""
    want = [s["speaker"] for s in S if s["slot"] is not None]
    marks = list(_ITEM.finditer(text))
    runs = []
    for mt in marks:
        if int(mt.group(1) or mt.group(2)) == 1 or not runs:
            runs.append([])
        runs[-1].append(mt)
    for run in reversed([r for r in runs if len(r) == k]):
        got, speakers = [], []
        for i, mt in enumerate(run):
            end = next((x.start() for x in marks if x.start() > mt.start()), len(text))
            item = tokens(text[mt.end():end])
            first = next((q for q, t in enumerate(item) if t == marker), None)
            if first is not None:
                ans = _answer(item[first:], marker, rescue=True)
            else:
                ans = _answer(item, marker)
                if ans and _item_key(ans[1][0]).split("(")[0] not in _vocab():
                    ans = None
            if ans and not _item_turn_ok(item, i, W):
                ans = None
            got.append(ans[0] if ans else None)
            speakers.append(next((item[q] for q in range(first or 0) if _tag_end(item, q)), None))
        if 2 * sum(a is not None for a in got) >= k and (not all(speakers) or speakers == want):
            return "list", list(enumerate(got))
    if marks:
        return None
    got, speakers, q = [], [], 0
    spans = _turn_spans(toks)
    while q < len(toks):
        if toks[q] == marker and q not in dialogue:
            ans = _read(toks, q, one=True)
            if ans:
                if _plausible(ans[0], marker):
                    got.append(ans[0])
                    speakers.append(next((spk for spk, a, b in spans if a <= q < b), None))
                q = max(ans[2], q + 1)
                continue
        q += 1
    if len(got) == k and (not all(speakers) or speakers == want):
        return "row", list(enumerate(got))
    return None


def _predict(src, response, marker):
    """(predictions, n_unaligned, how, conflicts). The dialogue rewritten is the answer: copies of
    the window that cover at least half the masked turns are read in order, and a later one's
    STRONG answers (an annotation right after the utterance) replace an earlier one's; a later blank
    replaces nothing. Everything else only FILLS slots still unread: weak answers (found after
    prose), smaller copies (notes, corrections quoting a turn or two), then a list or row reading.
    Where a filler disagrees with the answer kept, that is a conflict: counted, never applied.
    `how`: slots answered by each reading. Unaligned: slots no reading placed."""
    text = _THINK_END.split(response or "")[-1]       # reasoning a model printed: not the answer
    toks, offs = _tokens_at(text)
    S, k = _source_turns(src)
    copies, dialogue, W = _by_turns(S, k, toks, marker)
    found, best, conflicts, revised = set(), {}, set(), set()

    def differs(slot, ans):
        return slot in best and canonical(best[slot][0]) != canonical(ans)

    def fill(slot, ans, how, replace):
        if ans is None:
            return
        if differs(slot, ans):
            if not replace:
                conflicts.add(slot)
                return
            revised.add(slot)                    # a later full copy changed its mind
        if replace or slot not in best:
            best[slot] = (ans, how)

    copies.sort(key=lambda c: c[0])
    big = [read for _, read in copies if 2 * len(read) >= k]
    for read in big:
        for slot, (ans, strong) in read.items():
            if strong:
                fill(slot, ans, "turns", replace=True)
    for _, read in copies:
        found |= set(read)
        for slot, (ans, strong) in read.items():
            if not (strong and any(read is b for b in big)):
                fill(slot, ans, "turns", replace=False)
    listed = _listed(text, toks, offs, marker, S, k, dialogue, W)
    if listed:
        found |= {slot for slot, _ in listed[1]}
        clash = {slot for slot, ans in listed[1] if ans is not None and differs(slot, ans)}
        conflicts |= clash
        if not clash:                            # a list that disagrees anywhere may be shifted:
            for slot, ans in listed[1]:          # it fills nothing
                fill(slot, ans, listed[0], replace=False)
    preds = [best[s][0] if s in best else None for s in range(k)]
    how = collections.Counter(h for _, h in best.values())
    how["revised"] = len(revised)
    return preds, k - len(found), how, len(conflicts)


def predicted_slots(src, response, marker):
    """(predictions, n_unaligned) — see `_predict`."""
    preds, unaligned, _, _ = _predict(src, response, marker)
    return preds, unaligned


def score_row(row, response, marker):
    """Per-slot counts for one response: `lenient` (canonical sets equal), `exact` (the same tokens,
    up to spacing and case — reported beside, never the headline), and how the answered slots were
    read (`via_turns`, `via_list`, `via_row`), and `conflict`: slots where a list, row or partial
    copy said something other than the answer kept; `revised`: slots a later full copy changed."""
    preds, unaligned, how, conflicts = _predict(row["src"], response, marker)
    lenient = exact = missing = 0
    for g, p in zip(row["gold"], preds):
        if p is None:
            missing += 1
            continue
        exact += p == [t for t in g if t not in MARKERS]
        lenient += canonical(p) == canonical(g)
    return {"slots": len(row["gold"]), "lenient": lenient, "exact": exact, "missing": missing,
            "unaligned": unaligned, "via_turns": how["turns"], "via_list": how["list"],
            "via_row": how["row"], "conflict": conflicts, "revised": how["revised"]}


def summarize(rows):
    """{task: {"styles": {...}, "mean_over_styles", "complete", "check"}} over the `ok` rows. A
    style's accuracy is slots correct / slots — an unread slot counts as wrong — and the task
    headline the plain mean over the styles present. `complete` says whether every dialogue of every
    style was scored: an incomplete mean is not the headline. `check` lists styles with more than
    CHECK_SHARE of their slots unread — the model answered in a shape the reader could not place;
    `disagree` the styles where a response answered a slot twice, differently (`conflict`).
    `via` counts how the answered slots were read (turns, list, row): a style whose answers come
    by an unexpected route is a shape to look at before its number is believed."""
    data = {task: {r["idx"]: r for r in load_rows(task)} for task in TASKS}
    acc = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r.get("status") != "ok":
            acc[(r["task"], r["style"])]["not_ok"] += 1
            continue
        s = score_row(data[r["task"]][r["idx"]], r["response"], TASKS[r["task"]]["marker"])
        acc[(r["task"], r["style"])].update(s, rows=1)
    out = {}
    for task in TASKS:
        styles = {}
        for style in STYLES:
            c = acc.get((task, style))
            if not c or not c["slots"]:
                continue
            styles[style] = {"rows": c["rows"], "not_ok": c["not_ok"], "slots": c["slots"],
                             "lenient": c["lenient"] / c["slots"], "exact": c["exact"] / c["slots"],
                             "missing": c["missing"], "unaligned": c["unaligned"],
                             "unread": c["missing"] / c["slots"],
                             "via": {h: c["via_" + h] for h in ("turns", "list", "row")},
                             "conflict": c["conflict"], "revised": c["revised"]}
        if styles:
            complete = (len(styles) == len(STYLES) and all(
                v["rows"] == TASKS[task]["rows"] and not v["not_ok"] for v in styles.values()))
            out[task] = {"styles": styles, "complete": complete,
                         "mean_over_styles": sum(v["lenient"] for v in styles.values()) / len(styles),
                         "check": [s for s, v in styles.items() if v["unread"] > CHECK_SHARE],
                         "disagree": [s for s, v in styles.items() if v["conflict"] or v["revised"]]}
    return out


def print_summary(label, summary):
    print(f"\n{label}  (scorer {SCORER_VERSION})")
    for task, t in summary.items():
        print(f"  {task}: mean over styles {t['mean_over_styles']:.4f}"
              f"{'' if t['complete'] else '  — INCOMPLETE, not a headline'}"
              + (f"  — CHECK {', '.join(t['check'])}: many slots unread" if t["check"] else "")
              + (f"  — DISAGREE {', '.join(t['disagree'])}: responses answer twice, differently"
                 if t["disagree"] else ""))
        for style, v in t["styles"].items():
            via = " ".join(f"{h}={n}" for h, n in v["via"].items() if n)
            print(f"    {style:10s} rows={v['rows']:2d} slots={v['slots']:3d} lenient={v['lenient']:.4f}"
                  f" exact={v['exact']:.4f} missing={v['missing']} unaligned={v['unaligned']}"
                  f" via: {via or '-'}" + (f" conflict={v['conflict']}" if v["conflict"] else "")
                  + (f" revised={v['revised']}" if v["revised"] else "")
                  + (f" not_ok={v['not_ok']}" if v["not_ok"] else "")
                  + (f"  CHECK: {v['unread']:.0%} unread" if v["unread"] > CHECK_SHARE else ""))


# ---------------------------------------------------------------- the generation loop


def _is_cut(row):
    """A stored answer the output cap cut off and no forced answer has replaced."""
    return (row.get("status") == "ok" and row.get("finish_reason") == "length"
            and not row.get("forced_answer"))


def _rewrite(path, rows):
    """Every line of the file again, atomically — a forced answer replaces its row in place."""
    tmp = f"{path}.tmp.{os.uname().nodename}.{os.getpid()}"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def force_cut(gen, client, model, model_dir, params, config, path, by_uid, smoke, sleep):
    """`--force-cut`: the stored rows that ended at the output cap get the runner's forced answer
    (`gen.force_stored`: the cut text handed back, one more call) and NOTHING else is called — the
    user's rule, 2026-11-13: only the rows that hit the cap, not the run again.

    The forced answer is part of the config, so the stored rows are under the config without it.
    They are accepted under exactly that config and no other, the file is copied first to
    `results_archive_before_forced_answer_<date>/`, and every row is then written under the one
    new config. A forced call that fails leaves its stored row as it was — still cut, picked up by
    the next `--force-cut`. Exit 1 if ANY row of the file is still cut afterwards (a failed call, or
    a `--task`/`--styles` selection that left some out: the whole file now carries the config that
    promises a forced answer), or on what a plain run exits 1 for — a unit with no row, a row that
    is not `ok`, an unknown finish. Exit 2 if a row still ends at `length`: a forced answer that
    filled its own cap (a repetition loop does)."""
    if not os.path.exists(path):
        raise SystemExit(f"--force-cut: {path} does not exist — there are no stored rows to force")
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    want = json.loads(json.dumps(config, sort_keys=True))
    base = {k: v for k, v in want.items() if k != "forced_answer"}
    for r in rows:
        if r.get("config") not in (want, base):
            stored = r.get("config") or {}
            diff = {k: (want.get(k), stored.get(k)) for k in set(want) | set(stored)
                    if want.get(k) != stored.get(k) and k != "forced_answer"}
            raise SystemExit(f"{path}\n  refusing --force-cut: row {r.get('uid')} ran under a "
                             f"different config.\n  (wanted, stored): {diff}")
    last = {r["uid"]: i for i, r in enumerate(rows)}          # the last row per uid is the row
    todo = [i for uid, i in last.items() if uid in by_uid and _is_cut(rows[i])]
    upgrade = [r for r in rows if r.get("config") != want]
    print(f"--force-cut {os.path.basename(path)}: {len(last)} stored rows, {len(todo)} cut at the "
          f"cap to force, {len(upgrade)} row(s) under the config without the forced answer",
          flush=True)
    if upgrade:
        if not smoke:
            archive = os.path.join(model_dir, "results_archive_before_forced_answer_"
                                   + time.strftime("%Y%m%d"), CONDITION, os.path.basename(path))
            held = glob.glob(os.path.join(model_dir, "results_archive_before_forced_answer_*",
                                          CONDITION, os.path.basename(path)))
            if held and not all(filecmp.cmp(h, path, shallow=False) for h in held):
                raise SystemExit(f"{held[0]} already exists, differs from {path}, and that file "
                                 f"still holds rows from before the forced answer — look at both "
                                 f"before going on")
            if not held:           # else: a run that stopped before its first write, same file
                os.makedirs(os.path.dirname(archive), exist_ok=True)
                shutil.copy2(path, archive)
                print(f"  stored rows copied to {archive}", flush=True)
        for r in upgrade:
            r["config"] = want
        _rewrite(path, rows)

    failed = []
    for n, i in enumerate(todo, 1):
        stored = rows[i]
        uid = stored["uid"]
        task, style, row = by_uid[uid]
        marker = TASKS[task]["marker"]
        built = [{"role": "user", "content": build_prompt(task, style, row["src"])}]
        once = gen.force_stored(client, model, gen.messages(built), params, stored)
        text, meta, status = jc.call_with_retries(once, model, model_dir)
        if status == "ok":
            before = score_row(row, stored["response"], marker)
            rows[i] = dict(stored, **meta, response=text, status="ok", config=want,
                           first_latency_s=stored.get("latency_s"),
                           first_n_attempts=stored.get("n_attempts"))
            _rewrite(path, rows)
            after = score_row(row, text, marker)
            note = (f"lenient {before['lenient']} -> {after['lenient']}/{after['slots']} "
                    f"missing {before['missing']} -> {after['missing']}")
        else:
            failed.append(uid)
            note = "stored row left as it was"
        jc.note_outcome(status, model, model_dir)
        print(f"  [{n}/{len(todo)}] {uid:18s} {status} forced finish={meta.get('finish_reason')} "
              f"out={meta.get('forced_completion_tokens')} {meta.get('latency_s')}s  {note}",
              flush=True)
        time.sleep(sleep)

    final = [rows[i] for i in last.values() if rows[i]["uid"] in by_uid]
    forced = [r for r in final if r.get("forced_answer")]
    finish = collections.Counter(r.get("finish_reason") for r in final)
    print(f"\nDONE --force-cut: {len(todo) - len(failed)}/{len(todo)} forced now, {len(forced)} "
          f"forced rows in the file  forced finish="
          f"{dict(collections.Counter(r.get('finish_reason') for r in forced))}  "
          f"all rows finish={dict(finish)}  calls={jc.STATS}", flush=True)
    print_summary(os.path.relpath(path, MPGT_ROOT), summarize(final))
    still = [rows[i]["uid"] for i in last.values() if _is_cut(rows[i])]     # the whole file
    missing = [u for u in by_uid if u not in last]
    not_ok = [r["uid"] for r in final if r.get("status") != "ok"]
    odd = jc.unexpected_finish(final)
    if still:
        print(f"STOP: {len(still)} row(s) of the file are still cut at the cap with no forced answer "
              f"({len(failed)} failed just now): {still[:5]} — run --force-cut again, on every task "
              f"and style.", flush=True)
    if missing or not_ok:
        print(f"STOP: {len(missing)} unit(s) have no row and {len(not_ok)} row(s) are not ok "
              f"{(missing + not_ok)[:5]} — a plain run retries those.", flush=True)
    if odd:
        print(f"STOP: {sum(odd.values())} ok row(s) ended with finish_reason {dict(odd)}, not a known "
              f"complete answer — inspect them before scoring.", flush=True)
    if still or missing or not_ok or odd:
        sys.exit(1)
    if finish.get("length"):
        print(f"STOP: {finish['length']} row(s) still end at `length` — a forced answer that filled "
              f"its own cap. Read them before scoring.", flush=True)
        sys.exit(2)


def run_cli(gen, model_dir, description):
    """One response per (task, style, dialogue). Exit 1 unless every unit has an `ok` row; exit 2 if
    any row hit the output cap (`finish_reason=length`) — the cap bound, and that is decided before
    scoring, not after."""
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=gen.model)
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--styles", default=",".join(STYLES),
                    help="comma-separated subset of " + ",".join(STYLES))
    ap.add_argument("--limit", type=int, default=0,
                    help="first N dialogues per task; writes smoke/, never results/")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="build every prompt, call nothing")
    ap.add_argument("--force-cut", dest="force_cut", action="store_true",
                    help="call only the stored rows that ended at the output cap, and only their "
                         "forced answer (a runner that defines one); every other row is kept")
    args = ap.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    styles = [s for s in args.styles.split(",") if s]
    unknown = [s for s in styles if s not in STYLES]
    if unknown:
        raise SystemExit(f"unknown style(s) {unknown}; choose from {list(STYLES)}")
    todo_units = list(units(tasks, styles, args.limit))
    if args.force_cut and not ("forced_answer" in gen.extra_config and hasattr(gen, "force_stored")):
        raise SystemExit("--force-cut: this runner defines no forced answer for stored rows")
    if args.dry_run and args.force_cut:
        path = jc.out_path(model_dir, args.limit > 0, CONDITION, f"{jc.model_slug(args.model)}.jsonl")
        wanted = {u[0] for u in todo_units}
        cut = [r["uid"] for r in jc.read_rows(path) if r["uid"] in wanted and _is_cut(r)]
        for uid in cut:
            print(uid)
        print(f"{len(cut)} stored row(s) cut at the cap in {path}, nothing called")
        return
    if args.dry_run:
        for uid, task, style, row in todo_units:
            msgs = gen.messages([{"role": "user", "content": build_prompt(task, style, row["src"])}])
            print(f"{uid}: {len(msgs)} message(s), {sum(len(m['content']) for m in msgs)} chars, "
                  f"{len(row['gold'])} slots")
        print(f"{len(todo_units)} units, nothing called")
        return

    load_dotenv(ENV_PATH)
    jc.HARD_CALL_TIMEOUT = gen.call_timeout
    smoke = args.limit > 0
    jc.MARKER_PREFIX = ("smoke_" if smoke else "") + f"{CONDITION}_"
    jc.clear_stale_halt_markers(model_dir)
    client = gen.make_client()
    print(f"negotiating parameter surface for {args.model} ...", flush=True)
    params, notes = jc.negotiate(client, args.model, gen.wanted, model_dir, gen.required, gen.fixed)
    print(f"  accepted: {params}" + (f"  notes: {notes}" if notes else ""), flush=True)

    config = dict(gen.config(args.model, params, notes), condition=CONDITION,
                  input="upstream zero-shot prompt, one user turn, no system prompt"
                  + (" (the runner's prepare adds one: see config)" if gen.prepare else ""))
    path = jc.out_path(model_dir, smoke, CONDITION, f"{jc.model_slug(args.model)}.jsonl")
    jc.lock_output(path)
    if args.force_cut:
        force_cut(gen, client, args.model, model_dir, params, config, path,
                  {u[0]: u[1:] for u in todo_units}, smoke, args.sleep)
        return
    done = jc.load_done(path, config)
    todo = [u for u in todo_units if u[0] not in done]
    print(f"{CONDITION} {tasks} x {styles}: {len(todo_units)} units, {len(todo)} to call -> {path}",
          flush=True)

    for n, (uid, task, style, row) in enumerate(todo, 1):
        built = [{"role": "user", "content": build_prompt(task, style, row["src"])}]
        once = gen.once(client, args.model, gen.messages(built), params)
        text, meta, status = jc.call_with_retries(once, args.model, model_dir)
        jc.append_row(path, {"uid": uid, "task": task, "style": style, "idx": row["idx"],
                             "status": status, "response": text, "config": config, **meta})
        jc.note_outcome(status, args.model, model_dir)
        s = score_row(row, text, TASKS[task]["marker"]) if status == "ok" else None
        print(f"  [{n}/{len(todo)}] {uid:18s} {status} finish={meta.get('finish_reason')} "
              f"out={meta.get('completion_tokens')} reasoning={meta.get('reasoning_tokens')} "
              f"{meta.get('latency_s')}s"
              + (f"  lenient {s['lenient']}/{s['slots']} exact {s['exact']} missing {s['missing']}"
                 if s else "")
              + (f" fallback={meta['fallback']}" if meta.get("fallback") else ""), flush=True)
        time.sleep(args.sleep)

    wanted = {u[0] for u in todo_units}
    rows = [r for r in jc.read_rows(path) if r["uid"] in wanted]
    finish = collections.Counter(r.get("finish_reason") for r in rows)
    print(f"\nDONE {len(rows)}/{len(todo_units)} rows  status={jc.status_counts(rows)}  "
          f"finish={dict(finish)}  calls={jc.STATS}", flush=True)
    print_summary(os.path.relpath(path, MPGT_ROOT), summarize(rows))
    odd = jc.unexpected_finish(rows)
    if odd:
        print(f"STOP: {sum(odd.values())} ok row(s) ended with finish_reason {dict(odd)}, not a known "
              f"complete answer — inspect them before scoring.", flush=True)
    if finish.get("length"):
        print(f"STOP: {finish['length']} row(s) hit the output cap — the cap bound. Decide whether "
              f"to raise it (archive the results first) before scoring.", flush=True)
        sys.exit(2)
    if odd or len(rows) < len(todo_units) or any(r["status"] != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        for p in [a for a in sys.argv[1:] if a != "--summarize"]:
            print_summary(p, summarize(jc.read_rows(p)))
    else:
        print(__doc__)
