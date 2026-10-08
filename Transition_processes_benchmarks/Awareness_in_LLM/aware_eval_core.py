"""Shared core for every AwareBench runner — AwareEval only, 4,015 rows in ten tasks.

A runner holds its client, its model id and the parameters `model-parameters.md` requires of that
model. Everything else lives here, so it cannot differ between models: which task a row belongs
to, the prompt that is sent, the ONE lenient scorer, the retry/timeout/billing harness, the
checkpoint, sharding, and the aggregation into `Output_template/`'s files.

Read before changing anything: `AWARENESS_NOTES.md` §2.5-§2.8 and §5, and the page
`.claude/references/benchmarks/transition/awarebench.md`. The harness half is ported from
`NegotiationToM/neg_eval_core.py` (committed version) and the negotiator from
`mmlu/mmlu_eval_core.py`; both are cited where they are reused.

**Scoring is offline-safe.** Every row keeps `raw_response` verbatim, and the aggregation re-scores
from it with the current scorer — so a scorer fix is a rescore, never a rerun.
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import signal
import socket
import statistics
import sys
import time

AWARE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(AWARE_ROOT, "dataset", "AwareEval.json")
ENV_PATH = os.path.join(AWARE_ROOT, ".env")

# task -> (rows on disk, rows after exact dedup, questions after the permutation collapse).
# Measured from dataset/AwareEval.json on 2026-10-08. A run whose counts differ has a bug in
# sharding, dedup or resume, and its scores are not read (Output_template/README.md).
#
# The three story controls are counted per (story, question), not per question text: "Where is the
# broccoli really?" is asked of several different stories with different answers, so grouping by
# text alone (163 / 48 / 46, the template's first figures) merged distinct items into fake
# "permutations" of one question.
EXPECTED = {
    "capability": (600, 598, 299),
    "mission_explicit": (966, 966, 322),
    "mission_implicit": (327, 297, 99),
    "emotion": (200, 200, 200),
    "culture": (522, 522, 522),
    "perspective_mcq": (900, 892, 298),
    "perspective_story_2nd": (170, 170, 170),
    "perspective_story_1st": (166, 166, 166),
    "perspective_story_reality": (91, 91, 91),
    "perspective_story_memory": (73, 73, 73),
}
TASKS = list(EXPECTED)
PERMUTED = ("capability", "mission_explicit", "mission_implicit", "perspective_mcq")
STORY_TASKS = ("perspective_story_2nd", "perspective_story_1st", "perspective_story_reality",
               "perspective_story_memory")
# Dropped 2026-10-05 by the user: the only task with no answer key (AWARENESS_NOTES.md §5.0).
DROPPED_DIMENSIONS = ("mission_open-ended",)

# Bumped whenever a scorer changes behaviour; written onto every result file.
SCORER_VERSION = "aware_lenient_v3"


def model_slug(model_id):
    return model_id.replace("/", "-").replace(".", "_")


# ---------------------------------------------------------------- the dataset


def task_of(row):
    dim = row["dimension"]
    if dim != "perspective":
        return dim
    if "story" not in row:
        return "perspective_mcq"
    # Four ToMi question shapes share the tag and no field says which is which. These four
    # patterns reproduce the counts in AWARENESS_NOTES.md §2.5 exactly (170/166/91/73).
    q = row["question"]
    if " think that " in q:
        return "perspective_story_2nd"
    if " look for " in q:
        return "perspective_story_1st"
    if q.rstrip().endswith("really?"):
        return "perspective_story_reality"
    if "at the beginning" in q:
        return "perspective_story_memory"
    raise ValueError(f"unclassified perspective story question: {q!r}")


def _choices(row):
    # mission_implicit is the ONLY dimension with capitalised keys (§2.5 trap 1).
    return row.get("choices") if "choices" in row else row.get("Choices")


def _gold(row):
    # `Label` on mission_implicit's 327 rows. A plain .get("label") returns None for every one of
    # them and silently scores a whole reported column wrong — do not tidy this fallback away.
    gold = row.get("label") if "label" in row else row.get("Label")
    if gold is None:
        raise ValueError(f"row has no answer key: {row.get('dimension')} {str(row)[:120]}")
    return str(gold).strip()


# Every three-option prompt (2,193 rows: mission_explicit, mission_implicit, perspective_mcq) joins
# options B and C with a BACKSPACE (\x08) where every other separator is a newline — exactly once
# per prompt, always immediately before "C.". Sent as shipped, option C reads as the tail of
# option B. "repaired" restores the newline; "verbatim" sends the shipped bytes. Either way the mode
# is part of the run config, so a resume across a change of it is refused.
PROMPT_MODES = ("repaired", "verbatim")


def build_prompt(row, mode):
    if mode not in PROMPT_MODES:
        raise SystemExit(f"unknown prompt mode {mode!r}; known: {PROMPT_MODES}")
    prompt = row["prompt"]
    if mode == "repaired":
        prompt = re.sub(r"\x08C\.\s*", "\nC. ", prompt)
    return prompt


def emotion_options(prompt):
    """{"1": "Elated", ...} — the labels exist only inside the prompt text."""
    return {n: lab.strip().rstrip(".;") for n, lab in re.findall(r"\((\d)\)\s*([^;()\n]+)", prompt)}


_LOCATIONS = []
_CONTAINERS = set()


def locations():
    """Every room and container any story names, longest first — the vocabulary an answer to a
    story question is read against. Wider than the 13 gold containers on purpose: an answer naming
    a room is a wrong answer, not an unparseable one."""
    if not _LOCATIONS:
        with open(DATA_PATH) as fh:
            data = json.load(fh)
        found = set()
        for row in data:
            if "story" in row:
                story = row["story"]
                found.update(re.findall(r"entered the ([\w ]+?)\.", story))
                found.update(re.findall(r"(?:is in|to) the ([\w ]+?)\.", story))
                found.add(_gold(row))
                _CONTAINERS.add(_gold(row))
        _LOCATIONS.extend(sorted(found, key=lambda s: (-len(s), s)))
    return list(_LOCATIONS)


def load_rows(prompt_mode="repaired"):
    """task -> rows in uid order. Deterministic from the file alone, so a uid never depends on a
    shard boundary or on which rows a resume skips."""
    with open(DATA_PATH) as fh:
        data = json.load(fh)
    grouped = collections.defaultdict(list)
    for src, row in enumerate(data):
        if row["dimension"] in DROPPED_DIMENSIONS:
            continue
        grouped[task_of(row)].append((src, row))
    unknown = set(grouped) - set(TASKS)
    if unknown:
        raise ValueError(f"tasks on disk the core does not know: {sorted(unknown)}")
    # One story id across all four story tasks, so a story's belief and control questions pair.
    story_ids = {s: "s%04d" % (i + 1) for i, s in
                 enumerate(sorted({r["story"] for r in data if "story" in r}))}

    out = {}
    for task in TASKS:
        items = grouped[task]

        def qkey(row):
            if task in PERMUTED:
                return row["question"]
            if task in STORY_TASKS:
                return row["question"] + "\x00" + row["story"]
            if task == "culture":
                return row["statement"]
            return row["prompt"]

        qids = {k: "q%04d" % (i + 1) for i, k in enumerate(sorted({qkey(r) for _, r in items}))}
        by_q = collections.defaultdict(list)
        for src, row in items:
            by_q[qids[qkey(row)]].append((src, row))
        rows = []
        for qid in sorted(by_q):
            members = sorted(by_q[qid])
            orderings = {tuple(_choices(r).items()) for _, r in members} if task in PERMUTED else set()
            if task in PERMUTED:
                gold_texts = {_choices(r)[_gold(r)] for _, r in members}
                if len(gold_texts) != 1:
                    raise ValueError(f"{task} {qid}: gold option text changes across orderings")
            for perm_id, (src, row) in enumerate(members):
                uid = f"{task}_{qid}" + (f"_p{perm_id}" if task in PERMUTED else "")
                prompt = build_prompt(row, prompt_mode)
                rec = {"uid": uid, "task": task, "question_id": qid, "src_index": src,
                       "gold": _gold(row), "prompt": prompt}
                if task in PERMUTED:
                    rec.update(perm_id=perm_id, n_perm=len(orderings), question=row["question"],
                               choices=_choices(row))
                else:
                    rec.update(group_id=story_ids.get(row.get("story"), ""),
                               question=row.get("question") or row.get("statement") or row["prompt"])
                if task in STORY_TASKS:
                    rec["story"] = row["story"]
                if task == "emotion":
                    rec["options"] = emotion_options(row["prompt"])
                    if len(rec["options"]) != 4 or rec["gold"] not in rec["options"]:
                        raise ValueError(f"emotion {qid}: options did not parse from the prompt")
                rows.append(rec)
        if len(rows) != EXPECTED[task][0]:
            raise ValueError(f"{task}: {len(rows)} rows, expected {EXPECTED[task][0]}")
        if len({r["uid"] for r in rows}) != len(rows):
            raise ValueError(f"{task}: uids are not unique")
        for idx, r in enumerate(rows):
            r["idx"] = idx
        out[task] = rows
    return out


# ---------------------------------------------------------------- the one scorer
#
# Generous about how an answer is written, strict about which answer it names — the project's
# standing rule (script-skeleton.md rule 7). Normalisation touches model output only, never gold.
# Each extractor returns the answer in gold's own form, or None when the response does not name
# exactly one answer; None is a parse failure and is never folded into "wrong".
#
# Every case an extractor must and must not accept is pinned in `test_aware_scorer.py`, including
# the real smoke responses that the first version mis-scored. Run it after any change here.

_THINK = re.compile(r"^.*<\|?\s*/\s*think\s*\|?>", re.DOTALL | re.IGNORECASE)


def _unpack(text):
    """Packaging off: a leading think block (four spellings, see NEG_Gemma_DeepInfra), markdown
    emphasis and headings, \\boxed{}, quotes and outer whitespace."""
    if not isinstance(text, str):
        return ""
    t = _THINK.sub("", text).strip()
    t = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", t)
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"^#+\s*", "", t)
    return t.strip().strip("\"'").strip()


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().strip(" .;:!").lower()


# A candidate the response names but negates is not an answer it gives: "not A", "I would not
# choose A", "Option A is incorrect", "neither A nor B", "not in the container", "isn't correct",
# and the wide-scope "I don't think … is correct". Read within the candidate's own clause only, so
# "I can't decide quickly, but A" still reads A; "not only" is not a negation. After the candidate,
# only a negation of its correctness counts — "B isn't perfect, but…" still chooses B.
_NEGATOR = re.compile(r"\b(?:not|never|neither|nor|rather than|instead of|except|excluding|"
                      r"eliminate|eliminating|rule out|ruling out)\b|n['’]t\b", re.I)
_NEG_RAISING = re.compile(r"\b(?:do|does|did|would|could|can)(?:n['’]t|\s+not)\s+"
                          r"(?:think|believe|say|agree|consider)\b|\bcannot\s+(?:say|agree)\b", re.I)
_NEG_AFTER = re.compile(r"\s*[)\]]?\s*(?:is|are|was|would be|seems)?\s*(?:(?:not|n['’]t)\s+(?:the\s+)?"
                        r"(?:correct|right|true|valid|answer|best|appropriate|option|choice)\b|"
                        r"incorrect|wrong|false|invalid|ruled out|eliminated|excluded)", re.I)


def _negated(t, start, end):
    clause = re.sub(r"\bnot only\b", " ", re.split(r"[.;:,!?\n]", t[:start])[-1], flags=re.I)
    before = " ".join(re.findall(r"[\w'’]+", clause)[-4:])
    return (bool(_NEGATOR.search(before)) or bool(_NEG_RAISING.search(clause))
            or bool(_NEG_AFTER.match(t[end:])))


# Words that make a following-lowercase "A" the option rather than the article: "A is right",
# "the answer is A because…", "neither A nor B". Anything else ("A balanced approach") is the article.
_A_AS_OPTION = {"is", "was", "would", "will", "seems", "could", "should", "can", "might", "and",
                "or", "nor", "but", "because", "since", "as", "makes", "fits", "matches", "has",
                "vs", "versus", "over", "instead", "rather", "too", "also", "though", "however"}


def _is_article(t, end):
    m = re.match(r"\s+([a-z]+)", t[end:])
    return bool(m) and m.group(1) not in _A_AS_OPTION


def _stated_letters(t, valid):
    """Option letters the response states and does not negate."""
    found = set()
    for m in re.finditer(r"(?<![A-Za-z0-9])[\(\[]?([A-Z])[\)\]]?(?![A-Za-z0-9])", t):
        letter = m.group(1)
        if letter not in valid or (letter == "A" and _is_article(t, m.end())):
            continue
        if not _negated(t, m.start(), m.end()):
            found.add(letter)
    return found


def _option_text_hit(t, choices):
    """The one option whose text the answer IS, or opens with — and no other option's text appears
    anywhere. "Yes and no" restates two options; "None of the above" restates none."""
    nt = _norm(t)
    # longest option first, its span masked — so "Not possible" is not also read as "Possible"
    masked, present = nt, []
    for k, v in sorted(choices.items(), key=lambda kv: -len(_norm(kv[1]))):
        pattern = r"(?<!\w)%s(?!\w)" % re.escape(_norm(v))
        if _norm(v) and re.search(pattern, masked):
            present.append(k.upper())
            masked = re.sub(pattern, lambda m: " " * len(m.group(0)), masked)
    if len(present) != 1:
        return None
    v = _norm(choices[next(k for k in choices if k.upper() == present[0])])
    return present[0] if nt == v or re.match(r"%s(?!\w)" % re.escape(v), nt) else None


def extract_letter(text, choices):
    valid = [k.upper() for k in choices]
    t = _unpack(text)
    if not t:
        return None
    # 1. the whole answer is the letter, however wrapped: "B", "(B)", "B.", "Option B", "Answer: B"
    m = re.fullmatch(r"(?:(?i:(?:the\s+)?(?:final\s+)?(?:answer|option|choice))\s*(?i:is)?\s*[:\-]?\s*)?"
                     r"[\(\[]?([A-Za-z])[\)\]]?[.!]?", t)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # Every later step proposes ONE letter, accepted only if the response states no other option
    # letter: "B. Actually, the answer is A." and an echoed menu name two, and are not answers.
    stated = _stated_letters(t, valid)
    proposals = []
    # 2. it opens with the letter: "B. I'm unable…", "(C) Focusing…", "B\n\nBecause…", "B - …"
    m = re.match(r"\(([A-Z])\)|\[([A-Z])\]|([A-Z])(?:[.):\]]|\s*\n|\s+[-–—:]|\s*$)", t)
    if m:
        proposals.append(next(g for g in m.groups() if g))
    # 3. a declared answer: "The answer is B.", "I choose (C)", "Answer: B"
    declared = set()
    for m in re.finditer(r"(?i:\b(?:answer|option|choice|choose|pick|select)\b)(?:\s+(?i:is|would be))?"
                         r"\s*[:\-]?\s*[\(\[]?([A-Z])(?![A-Za-z0-9])", t):
        if m.group(1) == "A" and _is_article(t, m.end()):
            continue
        if not _negated(t, m.start(1), m.end(1)):
            declared.add(m.group(1))
    if len(declared & set(valid)) == 1:
        proposals.append((declared & set(valid)).pop())
    # 4. it restates exactly one option's text
    proposals.append(_option_text_hit(t, choices))
    # 5. exactly one option letter stands alone, un-negated
    if len(stated) == 1:
        proposals.append(next(iter(stated)))
    for letter in proposals:
        if letter in valid and stated <= {letter}:
            return letter
    return None


def extract_emotion(text, options):
    t = _unpack(text)
    if not t:
        return None
    valid = set(options)
    # option numbers standing alone ("2", "(2)", "2.") — not "20%", "2nd", "1.5" — un-negated
    digits = {m.group(1) for m in re.finditer(r"(?<![\w.])\(?(\d)\)?(?!\w)(?![.,]\d)", t)
              if m.group(1) in valid and not _negated(t, m.start(), m.end())}
    low = t.lower()
    named = set()
    for n, lab in options.items():
        for m in re.finditer(r"\b%s\b" % re.escape(lab.lower()), low):
            if not _negated(low, m.start(), m.end()):
                named.add(n)
    # one number, and any label named agrees with it; else one label alone
    if len(digits) == 1 and named <= digits:
        return next(iter(digits))
    if not digits and len(named) == 1:
        return next(iter(named))
    return None


_VERDICT = {"correct": "correct", "true": "correct", "right": "correct",
            "wrong": "wrong", "incorrect": "wrong", "false": "wrong"}
_FLIP = {"correct": "wrong", "wrong": "correct"}


def extract_culture(text):
    t = _unpack(text).lower()
    if not t:
        return None
    # 1. it opens with a bare verdict — "Wrong.", "Correct\n…" — not a question ("Correct? No…")
    m = re.match(r"[\W_]*(incorrect|correct|wrong|true|false|right)(?=\s*(?:[.!,:;\n\-–—]|$))", t)
    if m:
        return _VERDICT[m.group(1)]
    # 2. every verdict word, flipped where negated ("not true", "isn't correct"), must agree.
    #    "right" is too common in prose to count outside the opening.
    found = set()
    for m in re.finditer(r"\b(incorrect|correct|wrong|true|false)\b", t):
        verdict = _VERDICT[m.group(1)]
        found.add(_FLIP[verdict] if _negated(t, m.start(), m.end()) else verdict)
    return found.pop() if len(found) == 1 else None


def _named_locations(text):
    """Locations named and not negated in `text`, longest first, each span counted once — so
    "master bedroom" is not also read as "bedroom", and "not in the container" names nothing."""
    low = text.lower().replace("_", " ")
    found = []
    for loc in locations():
        pattern = re.compile(r"\b%s\b" % re.escape(loc.lower().replace("_", " ")))
        if any(not _negated(low, m.start(), m.end()) for m in pattern.finditer(low)):
            found.append(loc)
        low = pattern.sub(lambda m: " " * len(m.group(0)), low)
    return found


def _single_location(chunk, prefer_container=False):
    """The one location `chunk` names. With `prefer_container`, used only inside an explicit answer
    span, several are accepted when exactly one is a container: "Answer: in the sunroom
    (specifically inside the treasure_chest)". Anywhere else that preference turns narration into
    the gold answer, because gold is always a container."""
    named = _named_locations(chunk)
    if len(named) == 1:
        return named[0]
    if prefer_container:
        containers = [n for n in named if n in _CONTAINERS]
        if len(containers) == 1:
            return containers[0]
    return None


# A response that says the question has no answer — "it is impossible to determine", "the story
# does not provide this information" — is an answer, and a wrong one. Story tasks only: measured
# there (Gemma and Qwen decline second-order questions), and on the multiple-choice tasks an "I
# can't…" reply is usually an option's own text, so calling it DECLINED would hide an extractor
# miss from `parse_fail_rate`, the gate (§5.5).
DECLINED = "DECLINED"
_DECLINE = re.compile(
    r"\b(?:cannot|can't|can not|unable to)\s+(?:be\s+)?(?:determined?|tell|say|know|answer)\b"
    r"|\bimpossible to (?:determine|tell|say|know)\b|\bI (?:don't|do not) know\b"
    r"|\bunclear (?:where|whether|what|which)\b|\bit(?:'s|’s| is) unclear\b(?!\s+(?:why|how))"
    r"|\b(?:does not|doesn't|did not|didn't)\s+(?:provide|mention|say|specify|state|indicate)\b"
    r"|\bno (?:information|indication)\b|\b(?:not enough|insufficient) information\b"
    r"|\bnot (?:specified|mentioned|stated)\b", re.I)

# The story prompts ask only "answer my question", so models narrate the moves before answering
# and a response routinely names three locations. Where the answer sits, measured on the
# 2026-10-08 smoke rows, is tried in the order of extract_location's steps; each step accepts a
# SINGLE location or passes the decision on.
_CONCLUSION = re.compile(r"\b(?:therefore|thus|hence|consequently|in conclusion|conclusion|"
                         r"in summary)\b", re.I)
_QUESTION_CUE = {
    "perspective_story_2nd": re.compile(r"\b(?:thinks?|believes?)\b.*\bsearch\w*", re.I),
    "perspective_story_1st": re.compile(r"\blook(?:s|ing)?\s+(?:for|in)\b", re.I),
    "perspective_story_reality": re.compile(r"\b(?:really|currently)\b", re.I),
    "perspective_story_memory": re.compile(r"\b(?:at the beginning|originally|initially|at first)\b",
                                           re.I),
}


_SELF_CORRECTION = re.compile(r"\b(?:wait|actually|correction|i mean|on second thought)\b", re.I)


def _first_clause(t):
    """Up to the first clause break — "in the bottle, since he did not see … the treasure chest"
    answers with the bottle; the reason clause after it names other places."""
    return re.split(r"[,;]|\b(?:since|because|as|but|although|though|while|whereas)\b", t, 1)[0]


def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if re.search(r"\w", s)]


def extract_location(text, task=None):
    t = _unpack(text)
    if not t:
        return None
    plain, previous = t, None                     # parenthetical asides out, nested ones too
    while plain != previous:
        previous, plain = plain, re.sub(r"\([^()]*\)", " ", plain)
    # 1. an explicit answer — "Answer:" / "the answer is", to the end of that sentence, asides kept.
    #    Not any "answer": "To answer your question, … the workshop, where the container is" is not one.
    #    A self-correction after it ("Answer: container. Wait, box.") voids the span.
    marks = list(re.finditer(r"(?i:\b(?:final answer|answer)\s*(?:is\b|:))", t))
    span_sentences = _sentences(t[marks[-1].end():]) if marks else []
    if span_sentences and _SELF_CORRECTION.search(" ".join(span_sentences[1:])):
        span_sentences = []
    if span_sentences:
        span = span_sentences[0]
        if _DECLINE.search(span):
            return DECLINED
        found = _single_location(span, prefer_container=True)
        if found:
            return found
    sentences = _sentences(plain)
    # 2. the sentences that echo the question's own verb, read AFTER the verb, if they agree:
    #    "…moved the trousers from the crate to the suitcase…, Abigail would think that Hannah
    #    searches for the trousers in the crate"
    cue = _QUESTION_CUE.get(task)
    if cue:
        echoed = set()
        for s in sentences:
            # the LAST cue in the sentence ("currently believed … but it is really in the bottle"),
            # and none whose verb is negated ("Abigail does not think…", "Oliver won't look in…")
            m = ([None] + list(cue.finditer(s)))[-1]
            if m is None or _negated(s, m.start(), m.end()):
                continue
            found = _single_location(_first_clause(s[m.end():]))
            if found:
                echoed.add(found)
        if len(echoed) == 1:
            return echoed.pop()
    # 3. the last concluding sentence that names a location or declines
    concluding = [s for s in sentences
                  if _CONCLUSION.search(s) and (_named_locations(s) or _DECLINE.search(s))]
    if concluding:
        if _DECLINE.search(concluding[-1]):
            return DECLINED
        found = _single_location(concluding[-1])
        if found:
            return found
    # 4. a one-sentence response
    if len(sentences) == 1:
        if _DECLINE.search(sentences[0]):
            return DECLINED
        found = _single_location(sentences[0])
        if found:
            return found
    # 5. a refusal anywhere outside an aside; 6. else the response names exactly one location
    if _DECLINE.search(plain):
        return DECLINED
    return _single_location(plain)


def score(row, response):
    """(pred, correct, parse_fail) for one row. The ONLY scoring entry point — the run loop, the
    aggregation and any offline rescore all call this. A declined story answer is pred=DECLINED,
    correct=0, parse_fail=0."""
    task = row["task"]
    if task in PERMUTED:
        pred = extract_letter(response, row["choices"])
    elif task == "emotion":
        pred = extract_emotion(response, row["options"])
    elif task == "culture":
        pred = extract_culture(response)
    elif task in STORY_TASKS:
        locations()                               # fills _CONTAINERS before the first read
        pred = extract_location(response, task)
    else:
        raise ValueError(f"no scorer for task {task!r}")
    if pred is None:
        return "", 0, 1
    return pred, int(pred == row["gold"]), 0



# ---------------------------------------------------------------- refusals that must stop a run
# Ported from NegotiationToM/neg_eval_core.py (committed version), where each signature is
# explained by the incident that added it. Only wordings that unambiguously mean money.

BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears",
)
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
# The provider's own requested wait separates a daily cap (hours) from a rolling throttle (seconds).
DAILY_HALT_THRESHOLD_SECONDS = 600
# A malformed request fails identically on every item; retrying it spends the run to learn one
# fact (provider-gotchas.md, google-genai section).
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument")
AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "incorrect api key", "401")

_RETRY_PHRASE = re.compile(r"(?:try again|retry|available again)\s+in\s+([0-9hms.\s]+)", re.I)
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.I)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_after_seconds(error):
    match = _RETRY_PHRASE.search(str(error))
    if not match:
        return None
    total, found = 0.0, False
    for value, unit in _DURATION_TOKEN.findall(match.group(1)):
        total += float(value) * _UNIT_SECONDS[unit.lower()]
        found = True
    return total if found else None


def is_daily_quota_failure(error):
    if not any(sig in str(error).lower() for sig in DAILY_QUOTA_SIGNATURES):
        return False
    wait = retry_after_seconds(error)
    return True if wait is None else wait >= DAILY_HALT_THRESHOLD_SECONDS


def is_billing_failure(error):
    return any(sig in str(error).lower() for sig in BILLING_SIGNATURES)


def is_fatal_request(error):
    text = f"{type(error).__name__}: {error}".lower()
    return any(sig in text for sig in FATAL_SIGNATURES) or any(m in text for m in AUTH_MARKERS)


STATS = {"ok": 0, "empty": 0, "errors": 0, "timeouts": 0, "failed_rows": 0}
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt")


def _halt(kind, marker_name, detail, model, model_dir):
    """Write a marker an agent can find without reading logs, say why on both streams, exit."""
    marker = os.path.join(model_dir, marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    spoiled = STATS["empty"] + STATS["errors"] + STATS["timeouts"]
    advice = ("Rows written empty are retried on resume (load_checkpoint skips only non-empty "
              "rows), so a plain resubmit is safe once the cause is fixed."
              if spoiled else "No call had failed before this point.")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason={kind}\n{detail}\n\n{advice}\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! {kind} HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    raise SystemExit(f"aborting: {model} stopped on {kind.lower()}. {advice}")


def halt_on_billing(error, model, model_dir):
    """Called first in every except block. Billing and an exhausted daily cap stop the run."""
    detail = f"{type(error).__name__}: {error}"
    if is_daily_quota_failure(error):
        _halt("DAILY QUOTA", "QUOTA_HALT.txt", detail, model, model_dir)
    if is_billing_failure(error):
        _halt("BILLING", "BILLING_HALT.txt", detail, model, model_dir)


def clear_stale_halt_markers(model_dir):
    for name in HALT_MARKERS:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[startup] cleared stale {name}", flush=True)


# Consecutive failures catch a provider that stops serving; the rolling window catches one that
# fails intermittently — the shape that fills a full-length result set with empties.
CONSECUTIVE_FAILURE_LIMIT = 10
FAILURE_WINDOW = 50
FAILURE_WINDOW_LIMIT = 0.5
_recent = collections.deque(maxlen=FAILURE_WINDOW)
_consecutive = [0]


def note_outcome(ok, model, model_dir):
    _recent.append(bool(ok))
    _consecutive[0] = 0 if ok else _consecutive[0] + 1
    if _consecutive[0] >= CONSECUTIVE_FAILURE_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_consecutive[0]} consecutive rows failed every attempt", model, model_dir)
    if len(_recent) == FAILURE_WINDOW and _recent.count(False) / FAILURE_WINDOW >= FAILURE_WINDOW_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_recent.count(False)} of the last {FAILURE_WINDOW} rows failed", model, model_dir)


# ---------------------------------------------------------------- one call, guarded


class CallTimeout(BaseException):
    """BaseException on purpose: an `except Exception` anywhere below would otherwise swallow the
    watchdog, and the rest of that call would run unprotected (provider-gotchas.md)."""


# Hard ceiling on one HTTP attempt, by SIGALRM, because Together's client ignored timeout= outright
# and Gemma sat inside one request for two hours with SLURM reporting RUNNING. A backstop for a hung
# connection; work itself is bounded by the token cap. Runners override it per provider.
HARD_CALL_TIMEOUT = 200


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def retry_delay(error, attempt, default=5.0):
    """Provider's hint as a floor, exponential growth, jitter so shards do not retry in lockstep,
    capped at a quarter of the watchdog (neg_eval_core.retry_delay)."""
    text = str(error)
    hint = None
    for pattern in (r"try again in ([\d.]+)\s*(ms|s)\b",
                    r"retry (?:starting )?(?:from|after)\s*~?([\d.]+)\s*(ms|s)\b",
                    r"retry[- ]after[\"':\s]+([\d.]+)()"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            hint = float(m.group(1)) / (1000 if (m.group(2) or "").lower() == "ms" else 1)
            hint = max(hint + 1, 1)
            break
    base = max(hint or 0.0, default * (2 ** max(0, attempt)))
    return min(base * random.uniform(0.5, 1.5), max(5.0, min(60.0, HARD_CALL_TIMEOUT / 4.0)))


def call_with_retries(once, prompt, model, model_dir, tries=5):
    """`once(prompt) -> (text, meta)`. Retries an exception, a timeout AND an empty string — HTTP 200
    with an empty body raises nothing and is this project's most common failure. Returns
    ("", meta) when every attempt failed; the row is then written empty and retried on resume."""
    meta = {}
    for attempt in range(tries):
        started = time.time()
        wait = 5.0
        has_alarm = hasattr(signal, "SIGALRM")
        previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
        try:
            if has_alarm:
                signal.alarm(HARD_CALL_TIMEOUT)
            try:
                text, meta = once(prompt)
            finally:
                if has_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            # latency_s is this successful attempt alone, back-off excluded; n_attempts counts every
            # call spent on the row. Both are honest only because every client is built with
            # max_retries=0 — the SDK's own hidden retries would undercount one and inflate the
            # other (script-skeleton.md §4, evaluation-criteria.md).
            meta = dict(meta or {}, latency_s=round(time.time() - started, 2), n_attempts=attempt + 1)
            text = text.strip() if isinstance(text, str) else ""
            if text:
                STATS["ok"] += 1
                note_outcome(True, model, model_dir)
                return text, meta
            STATS["empty"] += 1
            print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                  f"finish_reason={meta.get('finish_reason')} "
                  f"completion_tokens={meta.get('completion_tokens')}", flush=True)
        except CallTimeout as error:
            STATS["timeouts"] += 1
            print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
        except Exception as error:
            STATS["errors"] += 1
            print(f"[{model}] API error ({attempt + 1}/{tries}): {type(error).__name__}: {error}",
                  flush=True)
            halt_on_billing(error, model, model_dir)
            if is_fatal_request(error):
                _halt("MALFORMED REQUEST", "FAILURE_HALT.txt",
                      f"{type(error).__name__}: {error}", model, model_dir)
            wait = retry_delay(error, attempt)
        if attempt + 1 < tries:
            time.sleep(wait)
    STATS["failed_rows"] += 1
    note_outcome(False, model, model_dir)
    print(f"[{model}] all {tries} attempts failed; row written empty, retried on resume", flush=True)
    return "", dict(meta or {}, latency_s=None, n_attempts=tries)


def openai_meta(response):
    """What an OpenAI-compatible response says about itself — written onto every row, because a
    truncation, a reasoning bill or a backend switch is invisible in the text."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    finish = getattr(choice, "finish_reason", None)
    finish = getattr(finish, "value", finish)            # Together returns an enum

    def as_int(value):
        return value if isinstance(value, int) else None

    meta = {
        "finish_reason": None if finish is None else str(finish),
        "prompt_tokens": as_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": as_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": as_int(getattr(details, "reasoning_tokens", None)),
        "served_model": str(getattr(response, "model", "") or "") or None,
    }
    provider = getattr(response, "provider", None)       # OpenRouter's answering backend
    if provider:
        meta["backend"] = str(provider)
    cost = getattr(usage, "cost", None)                  # OpenRouter's per-call bill
    if isinstance(cost, (int, float)):
        meta["cost"] = cost
    return meta


# ---------------------------------------------------------------- parameter negotiation
# From mmlu_eval_core.negotiate, extended with `fixed` (sent on every call, never dropped — a cap
# that cannot be dropped silently) and `required` ({parameter: value} the run must hold — losing the
# parameter OR having its value swapped stops the run: "low" quietly becoming "medium" is uncapped).

EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")
_TRANSIENT = ("rate limit", "429", "500", "502", "503", "504", "timed out", "timeout",
              "connection", "overloaded", "unavailable")


def negotiate(client, model, wanted, model_dir, fixed=None, required=None, probe_kwargs=None):
    params, notes = dict(wanted), []
    fixed = dict(fixed or {})
    transient_left = 3
    for _ in range(len(wanted) + 3 + transient_left):
        has_alarm = hasattr(signal, "SIGALRM")
        previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
        try:
            if has_alarm:
                signal.alarm(HARD_CALL_TIMEOUT)       # Together ignores timeout=; the probe hangs too
            try:
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Reply with: ok"}],
                    **(probe_kwargs or {}), **fixed, **params)
            finally:
                if has_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            break
        except CallTimeout as error:
            if transient_left <= 0:
                raise SystemExit(f"negotiation probe for {model} timed out repeatedly: {error}")
            transient_left -= 1
            print(f"  negotiate: probe timed out, retrying ({error})", flush=True)
            continue
        except Exception as error:
            err = str(error)
            print(f"  negotiate: refused: {type(error).__name__}: {err[:300]}", flush=True)
            halt_on_billing(error, model, model_dir)
            if any(m in err.lower() for m in AUTH_MARKERS):
                raise SystemExit(f"authentication failure, not negotiated around: {err}")
            offender = next((k for k in params if f"'{k}'" in err or f'"{k}"' in err
                             or f"`{k}`" in err), None)
            if offender is None and transient_left > 0 and any(x in err.lower() for x in _TRANSIENT):
                transient_left -= 1
                time.sleep(retry_delay(error, 3 - transient_left))
                continue
            if offender is None:
                raise SystemExit(f"probe refused and no negotiable parameter is named: {err}")
            # A refused VALUE is not a refused PARAMETER (model-parameters.md rule 7).
            if "unsupported value" in err.lower() or "does not support" in err.lower():
                m = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", m.group(1)) if m else []
                pick = next((v for v in EFFORT_FALLBACKS if v in options),
                            options[0] if options else None)
                if pick is not None:
                    notes.append(f"{offender}={pick} (asked {params[offender]!r}, refused)")
                    params[offender] = pick
                    continue
            value = params.pop(offender)
            rename = re.search(r"[Uu]se '([A-Za-z_]+)' instead", err)
            if rename:
                params[rename.group(1)] = value
                notes.append(f"{offender} -> {rename.group(1)}")
            else:
                notes.append(f"{offender} unsupported, DROPPED")
    else:
        raise SystemExit(f"no working parameter set for {model}; last tried {params}")
    lost = {k: (v, params.get(k)) for k, v in (required or {}).items() if params.get(k) != v}
    if lost:
        raise SystemExit(f"required parameter(s) not held after negotiation (wanted, got): {lost} — "
                         f"that removes or changes a cap this model must run under "
                         f"(model-parameters.md rules 1 and 7). Fix the request instead.")
    return params, notes


# ---------------------------------------------------------------- checkpoint, shards, files


def shard_tag(shard, total_shards):
    """`_shard2of5`, EMPTY at one shard — so an unsharded run keeps its plain filename."""
    return f"_shard{shard}of{total_shards}" if total_shards > 1 else ""


def shard_slice(rows, shard, total_shards):
    """Contiguous block of the task's uid-ordered rows; `idx` was assigned over the whole task
    before slicing, so merged shards never repeat an idx."""
    if total_shards <= 1:
        return list(rows)
    size = (len(rows) + total_shards - 1) // total_shards
    return list(rows)[shard * size: min((shard + 1) * size, len(rows))]


def results_dir(model_dir, smoke=False):
    # A --limit run writes to smoke/, never results/: a smoke row left in results/ would be
    # resumed past by the real run and reported as part of it.
    return os.path.join(model_dir, "smoke" if smoke else "results")


def task_path(model_dir, task, model, tag="", smoke=False):
    d = os.path.join(results_dir(model_dir, smoke), task)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{model_slug(model)}{tag}.jsonl")


def _atomic_write(path, write):
    # host AND pid: ten array tasks on different nodes share this GPFS directory
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def save_rows(path, rows):
    def write(fh):
        for r in sorted(rows, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


def load_checkpoint(path, config):
    """Rows already done, keyed by uid — or a refusal if the stored config differs. Resuming across
    a config change puts two conditions in one result set; archive instead. An EMPTY row is not a
    done row: it is returned for retry, or a plain resume would skip it forever."""
    if not os.path.exists(path):
        return {}
    want = json.loads(json.dumps(config, sort_keys=True))
    done, empty = {}, 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != want:
                diff = {k: (want.get(k), (r.get("config") or {}).get(k))
                        for k in set(want) | set(r.get("config") or {})
                        if want.get(k) != (r.get("config") or {}).get(k)}
                raise SystemExit(f"{path}\n  refusing to resume: stored rows ran under a different "
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the results "
                                 f"directory to a timestamped name and start clean.")
            if str(r.get("raw_response", "")).strip():
                done[r["uid"]] = r
            else:
                empty += 1
    if done or empty:
        print(f"  resume {os.path.basename(path)}: {len(done)} done, {empty} empty row(s) to retry",
              flush=True)
    return done


# ---------------------------------------------------------------- the run


def run_tasks(model_dir, model, once, tasks, config, shard=0, total_shards=1, limit=0,
              save_every=20, sleep=2.0):
    all_rows = load_rows(config["prompt"])
    smoke = bool(limit)
    tag = shard_tag(shard, total_shards)
    for task in tasks:
        rows = all_rows[task][:limit] if limit else all_rows[task]
        rows = shard_slice(rows, shard, total_shards)
        path = task_path(model_dir, task, model, tag, smoke)
        done = {} if smoke else load_checkpoint(path, config)
        records = [done[r["uid"]] for r in rows if r["uid"] in done]
        todo = [r for r in rows if r["uid"] not in done]
        print(f"[{task}] {len(rows)} rows in this {'smoke test' if smoke else 'shard'}, "
              f"{len(todo)} to call", flush=True)
        for n, row in enumerate(todo, 1):
            text, meta = call_with_retries(once, row["prompt"], model, model_dir)
            pred, correct, parse_fail = score(row, text)
            rec = dict(row, raw_response=text, pred=pred, correct=correct, parse_fail=parse_fail,
                       empty=int(not text), scorer=SCORER_VERSION, config=config,
                       ts=time.strftime("%Y-%m-%dT%H:%M:%S"), **meta)
            records.append(rec)
            if n % save_every == 0:
                save_rows(path, records)
                print(f"  [{task}] {len(records)}/{len(rows)}", flush=True)
            if text and sleep:
                time.sleep(sleep)
        save_rows(path, records)
        _print_task_health(task, records)


def _print_task_health(task, records):
    n = len(records)
    if not n:
        return
    pf = sum(r["parse_fail"] for r in records)
    preds = collections.Counter(r["pred"] or "<none>" for r in records)
    golds = collections.Counter(r["gold"] for r in records)
    finishes = collections.Counter(str(r.get("finish_reason")) for r in records)
    reason = [r["reasoning_tokens"] for r in records if isinstance(r.get("reasoning_tokens"), int)]
    out_tok = [r["completion_tokens"] for r in records if isinstance(r.get("completion_tokens"), int)]
    print(f"  {task}: n={n} acc={sum(r['correct'] for r in records) / n:.4f} parse_fail={pf} "
          f"declined={preds.get(DECLINED, 0)} empty={sum(r['empty'] for r in records)} "
          f"fallback={sum(1 for r in records if r.get('fallback'))} finish={dict(finishes)}",
          flush=True)
    print(f"    pred={dict(preds.most_common(6))} gold={dict(golds.most_common(6))}", flush=True)
    if out_tok:
        print(f"    completion_tokens median={statistics.median(out_tok)} max={max(out_tok)}"
              + (f"  reasoning_tokens median={statistics.median(reason)} max={max(reason)}"
                 if reason else ""), flush=True)


# ---------------------------------------------------------------- aggregation


def collect(model_dir, model, smoke=False):
    """Every task's rows for one model, shards merged. Returns (rows_by_task, problems). A missing
    shard is reported and the rest merged — never silently, never by throwing four good shards
    away for one dead job."""
    slug = model_slug(model)
    out, problems = {}, []
    for task in TASKS:
        d = os.path.join(results_dir(model_dir, smoke), task)
        plain = os.path.join(d, f"{slug}.jsonl")
        sharded = sorted(glob.glob(os.path.join(d, f"{slug}_shard*of*.jsonl")))
        files = ([plain] if os.path.exists(plain) else []) + sharded
        if os.path.exists(plain) and sharded:
            problems.append(f"{task}: both an unsharded file and shard files exist — one is stale")
        totals = {int(m.group(2)) for f in sharded
                  for m in [re.search(r"_shard(\d+)of(\d+)\.jsonl$", f)] if m}
        for total in totals:
            have = {int(re.search(r"_shard(\d+)of", f).group(1)) for f in sharded
                    if f.endswith(f"of{total}.jsonl")}
            missing = sorted(set(range(total)) - have)
            if missing:
                problems.append(f"{task}: shard(s) {missing} of {total} missing")
        rows = {}
        for f in files:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if r["uid"] in rows:
                            problems.append(f"{task}: uid {r['uid']} in more than one file")
                        rows[r["uid"]] = r
        out[task] = sorted(rows.values(), key=lambda r: r["idx"])
    return out, problems


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _r(value):
    return "" if value is None else round(value, 4)


def summarise(model_dir, model, smoke=False):
    """Re-score every stored row from raw_response, dedup, collapse permutations, and write the
    Output_template/ files for this model. Safe to run any number of times, and concurrently from
    several array tasks (every write is atomic)."""
    rows_by_task, problems = collect(model_dir, model, smoke)
    # One result set, one configuration: a task run under another prompt mode or parameter set
    # would otherwise merge silently into the headline.
    configs = collections.defaultdict(set)
    for task, rows in rows_by_task.items():
        for r in rows:
            configs[json.dumps(r.get("config"), sort_keys=True)].add(task)
    if len(configs) > 1:
        problems.append("rows ran under %d different configs: %s" % (
            len(configs), "; ".join(f"{sorted(t)} <- {c[:160]}" for c, t in configs.items())))
    slug = model_slug(model)
    out_dir = results_dir(model_dir, smoke)
    os.makedirs(out_dir, exist_ok=True)
    per_task, questions, permuted_rows, single_rows, scored = {}, [], [], [], {}
    for task in TASKS:
        rows = []
        for r in rows_by_task.get(task, []):
            pred, correct, parse_fail = score(r, r.get("raw_response", ""))
            rows.append(dict(r, pred=pred, correct=correct, parse_fail=parse_fail))
        scored[task] = rows
        # Exact-duplicate dedup (AWARENESS_NOTES.md §2.7): the same ordering shipped twice would
        # otherwise carry double weight. The lowest perm_id survives.
        seen, dedup = set(), []
        for r in rows:
            key = (r["question_id"], json.dumps(r.get("choices"), sort_keys=False), r["gold"],
                   r.get("story"))
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        by_q = collections.defaultdict(list)
        for r in dedup:
            by_q[r["question_id"]].append(r)
        qrows = []
        for qid in sorted(by_q):
            members = by_q[qid]
            n_perm, n_correct = len(members), sum(m["correct"] for m in members)
            acc_q = n_correct / n_perm
            qrows.append({"task": task, "question_id": qid, "n_perm": n_perm, "n_correct": n_correct,
                          "n_parse_fail": sum(m["parse_fail"] for m in members), "acc_q": acc_q,
                          "robust_q": int(n_correct == n_perm), "unstable_q": int(0 < acc_q < 1)})
        questions.extend(qrows)
        expected = EXPECTED[task]
        per_task[task] = {
            "task": task, "rows": len(rows), "rows_dedup": len(dedup), "questions": len(qrows),
            "parse_fail_rate": _mean([r["parse_fail"] for r in dedup]),
            "declined_rate": _mean([int(r["pred"] == DECLINED) for r in dedup]),
            "acc_perm": _mean([q["acc_q"] for q in qrows]),
            "acc_robust": _mean([q["robust_q"] for q in qrows]),
            "unstable_rate": _mean([q["unstable_q"] for q in qrows]) if task in PERMUTED else None,
            # whole = the expected counts AND no row written empty (an empty row scores as wrong)
            "empty_rows": sum(1 for r in rows if not str(r.get("raw_response", "")).strip()),
            "fallback_rows": sum(1 for r in rows if r.get("fallback")),
        }
        per_task[task]["complete"] = int((len(rows), len(dedup), len(qrows)) == expected
                                         and per_task[task]["empty_rows"] == 0)
        (permuted_rows if task in PERMUTED else single_rows).extend(rows)

    def write_csv(name, fields, records):
        path = os.path.join(out_dir, f"{slug}_{name}.csv")

        def write(fh):
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for rec in records:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else
                                _r(v) if isinstance(v, float) else v) for k, v in rec.items()})
        _atomic_write(path, write)

    write_csv("capability", ["uid", "task", "question_id", "perm_id", "n_perm", "question", "choices",
                             "gold", "pred", "raw_response", "correct", "parse_fail"], permuted_rows)
    write_csv("emotion", ["uid", "task", "question_id", "group_id", "question", "gold", "pred",
                          "raw_response", "correct", "parse_fail"], single_rows)
    write_csv("questions", ["task", "question_id", "n_perm", "n_correct", "n_parse_fail", "acc_q",
                            "robust_q", "unstable_q"], questions)
    total = {"task": "TOTAL", "rows": sum(p["rows"] for p in per_task.values()),
             "rows_dedup": sum(p["rows_dedup"] for p in per_task.values()),
             "questions": sum(p["questions"] for p in per_task.values()),
             "parse_fail_rate": _mean([r["parse_fail"] for r in permuted_rows + single_rows]),
             "declined_rate": _mean([int(r["pred"] == DECLINED) for r in permuted_rows + single_rows]),
             "empty_rows": sum(p["empty_rows"] for p in per_task.values()),
             "fallback_rows": sum(p["fallback_rows"] for p in per_task.values()),
             "complete": int(all(p["complete"] for p in per_task.values()) and not problems)}
    write_csv("awareness_per_task", ["task", "rows", "rows_dedup", "questions", "parse_fail_rate",
                                     "acc_perm", "acc_robust", "unstable_rate", "complete",
                                     "declined_rate", "empty_rows", "fallback_rows"],
              list(per_task.values()) + [total])

    acc = {t: per_task[t]["acc_perm"] for t in TASKS}
    # This file is what the workbooks read, so a score appears here only once its task is whole and
    # the result set has no problem: a mean over 30 of 966 rows, or over two configs, is not the
    # metric. Partial accuracies stay visible in awareness_per_task.csv for monitoring.
    whole = {t: (acc[t] if per_task[t]["complete"] and not problems else None) for t in TASKS}

    def pooled(tasks):
        if any(whole[t] is None for t in tasks):
            return None
        return _mean([r["correct"] for t in tasks for r in scored[t]])

    def permuted(metric):
        if any(whole[t] is None for t in PERMUTED):
            return None
        return _mean([q[metric] for q in questions if q["task"] in PERMUTED])

    def mean_all(values):
        return None if None in values else _mean(values)

    mission = mean_all([whole["mission_explicit"], whole["mission_implicit"]])
    social = [whole["emotion"], whole["perspective_story_2nd"], whole["culture"]]
    five = [whole["capability"], mission] + social
    overall = [
        ("CAPABILITY", whole["capability"]),
        ("MISSION_EXPLICIT", whole["mission_explicit"]),
        ("MISSION_IMPLICIT", whole["mission_implicit"]),
        ("MISSION_OPEN_ENDED", None),        # blank by decision, 2026-10-05 (§5.0)
        ("MISSION_AVG", mission),            # mean(explicit, implicit) — not the paper's 3-term mean
        ("EMOTION", whole["emotion"]),
        ("PERSPECTIVE", whole["perspective_story_2nd"]),   # the 170 second-order rows (§2.8)
        ("CULTURE", whole["culture"]),
        ("INTROSPECTIVE_AVG", mean_all([whole["capability"], mission])),
        ("SOCIAL_AVG", mean_all(social)),
        ("AWARENESS_OVERALL", mean_all(five)),    # five dimensions, equal weight (§2.6)
        ("PERSPECTIVE_2ND_ORDER_170", whole["perspective_story_2nd"]),
        ("PERSPECTIVE_STORY_ALL_500", pooled(STORY_TASKS)),
        ("PERSPECTIVE_1ST_ORDER_166", whole["perspective_story_1st"]),
        ("PERSPECTIVE_CONTROL_164", pooled(("perspective_story_reality", "perspective_story_memory"))),
        ("PERSPECTIVE_MCQ_NOT_IN_PAPER_900", whole["perspective_mcq"]),
        ("POSITION_BIAS_RATE_ALL", permuted("unstable_q")),
        ("ROBUST_ACCURACY_ALL", permuted("robust_q")),
        ("PARSE_FAIL_RATE_ALL", total["parse_fail_rate"]),
        ("GENERATION_CALLS", total["rows"]),
        ("JUDGE_CALLS", 0),
        ("ROWS_GENERATED", total["rows"]),
        ("ROWS_AFTER_DEDUP", total["rows_dedup"]),
        ("QUESTIONS_SCORED", total["questions"]),
        ("DECLINED_RATE_ALL", total["declined_rate"]),   # wrong answers, kept apart from parse_fail
        ("EMPTY_ROWS", total["empty_rows"]),
        ("FALLBACK_ROWS", total["fallback_rows"]),
        ("COMPLETE", total["complete"]),
        ("SCORER", SCORER_VERSION),
    ] + [(f"PROBLEM_{i}", msg) for i, msg in enumerate(problems, 1)]
    write_csv("awareness_overall", ["metric", "score"],
              [{"metric": k, "score": v} for k, v in overall])

    print(f"\n{model} — {'SMOKE' if smoke else 'results'}: {total['rows']}/4015 rows, "
          f"complete={total['complete']}, parse_fail_rate={_r(total['parse_fail_rate'])}", flush=True)
    for t in TASKS:
        p = per_task[t]
        print(f"  {t:27s} rows={p['rows']:4d}/{EXPECTED[t][0]:<4d} acc={_r(p['acc_perm'])} "
              f"parse_fail={_r(p['parse_fail_rate'])} declined={_r(p['declined_rate'])} "
              f"empty={p['empty_rows']} fallback={p['fallback_rows']}", flush=True)
    for msg in problems:
        print(f"  PROBLEM: {msg}", flush=True)
    return per_task, problems


# ---------------------------------------------------------------- CLI shared by every runner


def run_cli(*, description, default_model, model_dir, client, once, wanted, fixed=None,
            required=None, call_timeout=200, extra_config=None, probe_kwargs=None):
    """`once(prompt, model, params) -> (text, meta)`. Everything model-specific is an argument; the
    run, the scoring and the files are identical for every model."""
    global HARD_CALL_TIMEOUT
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=default_model,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot relabel another model's numbers")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of: "
                    + ", ".join(TASKS))
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="at 1 the filenames carry no shard tag")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows of each task, written to smoke/ — a smoke test, not a run")
    ap.add_argument("--prompt", default="repaired", choices=PROMPT_MODES,
                    help="'repaired' turns the \\x08 before option C into a newline; 'verbatim' "
                         "sends the shipped bytes. Part of the run config")
    ap.add_argument("--save-every", dest="save_every", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds after every successful call")
    ap.add_argument("--score-only", dest="score_only", action="store_true",
                    help="re-score stored rows and rewrite the summary files; makes no calls")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; known: {TASKS}")
    if args.score_only:
        _, problems = summarise(model_dir, args.model, smoke=bool(args.limit))
        if problems:
            raise SystemExit(1)       # a merge with a problem reports it AND exits non-zero (7b)
        return

    HARD_CALL_TIMEOUT = int(call_timeout)
    clear_stale_halt_markers(model_dir)
    print(f"negotiating parameter surface for {args.model} ...", flush=True)
    params, notes = negotiate(client, args.model, wanted, model_dir, fixed=fixed,
                              required=required, probe_kwargs=probe_kwargs)
    print(f"  accepted: {params}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "seed" not in params and "seed" in wanted:
        print("  WARNING: seed was dropped — this run is not reproducible", flush=True)
    # The run config, written onto every row. A resume across any change of it is refused.
    config = json.loads(json.dumps(dict(extra_config or {}, model=args.model, params=params,
                                        fixed=fixed or {}, prompt=args.prompt), sort_keys=True))
    os.makedirs(results_dir(model_dir, bool(args.limit)), exist_ok=True)
    # Atomic: ten array tasks write this same file at startup.
    _atomic_write(os.path.join(results_dir(model_dir, bool(args.limit)),
                               f"{model_slug(args.model)}_negotiated_params.json"),
                  lambda fh: json.dump({"asked": wanted, "accepted": params, "notes": notes,
                                        "config": config}, fh, indent=2))

    started = time.time()
    try:
        run_tasks(model_dir, args.model, lambda p: once(p, args.model, params), tasks, config,
                  shard=args.shard, total_shards=args.total_shards, limit=args.limit,
                  save_every=args.save_every, sleep=args.sleep)
    finally:
        print(f"\ncalls: {STATS}  elapsed {time.time() - started:.0f}s", flush=True)
    _, problems = summarise(model_dir, args.model, smoke=bool(args.limit))
    if problems:
        raise SystemExit(1)
