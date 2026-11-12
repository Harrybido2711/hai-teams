"""Checks on mpgt's lenient scorer, on the real data — run before trusting a number from it:

    python test_mpgt_scorer.py

Three kinds of check, each over every dialogue of both tasks:
- the gold, and the gold in every shape a model was seen to answer in, or plausibly could — a
  [MASK] left beside the answer, the unfilled window echoed before or after, a numbered list,
  the marker or `( )` dropped, items joined by `and` or the marker — scores 100% lenient;
- an answer the response never gave is never credited: an unfilled window, an echoed
  conversation whose context carries annotations, an extra or hedged item, a note after the
  window that quotes an annotation;
- content is not forgiven: one changed intent or speaker costs exactly one slot.
Cases A1–A6 are the reviewer's first review (2026-11-09), F1–F6 its second and G1–G6/U1–U4 its
third, H1–H6/U5–U6 its fourth and R1–R4 its fifth (2026-11-11) — each failed on the scorer it reviewed. Where a
response answers a slot twice and differently, the dialogue's answer stands and the CONFLICT is
counted: those cases check both. Where a response gives no answer, the check is that nothing is
READ (every prediction None, so the summary's CHECK sees it), not merely that nothing is credited.
"""

import re

import mpgt_eval_core as m

SPK = r"(?:ARI|RP|RC|LC|LP) :"


def ann_re(marker):
    """One gold annotation inside `tgt`: from the marker to the next speaker tag or the end."""
    return re.compile(r"\s" + re.escape(marker) + r" .*?(?=\s" + SPK + r"|$)")


def before(src):
    return src[:src.find("[start]")]


def after(src):
    return src[src.find("[end]") + len("[end]"):]


def total(task, make):
    marker = m.TASKS[task]["marker"]
    agg = {"slots": 0, "lenient": 0, "exact": 0, "missing": 0, "unaligned": 0, "conflict": 0,
           "revised": 0}
    for row in m.load_rows(task):
        s = m.score_row(row, make(row, marker), marker)
        for k in agg:
            agg[k] += s[k]
    return agg


def credited(task, make):
    """make(row, marker) -> (response, slot indices whose answer the response does NOT give);
    returns how many of those slots were credited anyway — must be 0."""
    marker = m.TASKS[task]["marker"]
    n = hits = 0
    for row in m.load_rows(task):
        response, targets = make(row, marker)
        if response is None:
            continue
        preds, _ = m.predicted_slots(row["src"], response, marker)
        for i in targets:
            n += 1
            p = preds[i]
            hits += p is not None and m.canonical(p) == m.canonical(row["gold"][i])
    return hits, n


def reformat(tgt):
    """The gold, written the way a model plausibly writes it: annotations squeezed, multi-item
    annotations reversed, `LC+RP` as `RP+LC`, and wrapped in prose and a code fence."""
    def squeeze(match):
        ann = match.group(0)
        parts = [p.strip() for p in re.split(r"\s;\s", ann[1:])]
        return ann[0] + " ; ".join(reversed(parts)).replace(" ", "")
    out = re.sub(r"[#@][^#@]*?(?=\s(?:ARI|RP|RC|LC|LP) :|$)", squeeze, tgt)
    out = re.sub(r"\b(LC|RP|RC|LP)\+(LC|RP|RC|LP)\b", r"\2+\1", out)
    return "Here is the window with every [MASK] replaced:\n```\n" + out + "\n```\nHope this helps."


def numbered(row, marker, speakers=False):
    lines, turns = [], re.split(r"\s(?=" + SPK + ")", m.window(row["src"]).strip())
    masked = [t.split(" :")[0] for t in turns if "[MASK]" in t]
    for i, g in enumerate(row["gold"]):
        label = f"**{masked[i]}:** " if speakers else ""
        lines.append(f"{i + 1}. {label}`{' '.join(g)}`")
    return "The masked annotations, in order:\n\n" + "\n".join(lines)


def bogus(row, marker):
    """The window with every annotation replaced by a wrong one."""
    return ann_re(marker).sub(lambda x: f" {marker} bogus ( )", row["tgt"])


def turns_of(tgt, marker):
    """[(speaker, utterance, annotation or None)] for the window's turns."""
    out = []
    for part in re.split(r"\s(?=" + SPK + ")", " " + tgt.strip()):
        part = part.strip()
        if not part:
            continue
        spk, rest = part.split(" :", 1)
        k = (" " + rest).find(f" {marker} ")
        out.append((spk, rest.strip(), None) if k < 0 else
                   (spk, (" " + rest)[:k].strip(), (" " + rest)[k:].strip()))
    return out


def rebuild(row, marker, turn, sep="\n"):
    """The window rewritten turn by turn: turn(speaker, utterance, annotation or None) -> text."""
    return sep.join(turn(s, u, a) for s, u, a in turns_of(row["tgt"], marker))


def detok(text):
    text = re.sub(r" (n't|'s|'m|'re|'ve|'ll|'d)\b", r"\1", text)
    return re.sub(r" ([.,?!])", r"\1", text)


def masked_turns(row):
    return [p for p in re.split(r"\s(?=" + SPK + ")", m.window(row["src"]).strip()) if "[MASK]" in p]


def blank_last(text, marker):
    last = list(ann_re(marker).finditer(text))[-1]
    return text[:last.start()] + " [MASK]" + text[last.end():]


def analysis_then_key(row, marker):
    """Qwen's story shape: a k-item analysis, each with its tag, then a k-item answer key."""
    a = "\n\n".join(f"{i + 1}.  **Context**: the speaker reacts.\n    *   **Tag**: `{' '.join(g)}`"
                     for i, g in enumerate(row["gold"]))
    k = "\n".join(f"{i + 1}.  `{' '.join(g)}`" for i, g in enumerate(row["gold"]))
    return "Here is the breakdown:\n\n" + a + "\n\n### Final Answer Key\n\n" + k


def closed(ann):
    """An annotation with the gold's slip mended: brackets it leaves open are closed."""
    return ann + " )" * (ann.count("(") - ann.count(")"))


def per_turn(suffix):
    """Every answered turn followed by `suffix` on the same line — the gold's open brackets closed,
    since a model does not copy that slip and an item left open runs into the words after it."""
    return lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" {closed(a)}{suffix}" if a else ""))


def quote_previous(row, marker):
    """k numbered items, each quoting the turn BEFORE its slot's and then giving the slot's tag."""
    mt = [p.replace(" [MASK]", "") for p in masked_turns(row)]
    return "\n\n".join(f"{i + 1}. Coming right after `{mt[i - 1]}`, this turn is `{' '.join(g)}`." if i
                        else f"1. This turn is `{' '.join(g)}`." for i, g in enumerate(row["gold"]))


def bogus_list(row, marker):
    return "\n".join(f"{i + 1}. `{marker} bogus ( )`" for i in range(len(row["gold"])))


def note_on(row, marker, i, offered, tail=""):
    q = masked_turns(row)[i].replace(" [MASK]", "")
    return f"\n\nNote: `{q}` could arguably also be `{offered}`{tail}."


def moved_to_neighbour(row, marker):
    """(response, moved slots): as many annotations as possible, in order, written on a nearby
    UNMASKED turn of the same speaker instead of their own — or (None, []) if under half move."""
    tt = turns_of(row["tgt"], marker)
    ann = [i for i, x in enumerate(tt) if x[2]]
    free = [i for i, x in enumerate(tt) if not x[2]]
    target, last = {}, -1
    for i in ann:
        c = [q for q in free if q > last and tt[q][0] == tt[i][0] and abs(q - i) <= 6
             and tt[q][1] != tt[i][1]]          # a turn that reads the same is no misplacement
        q = min(c, key=lambda q: abs(q - i)) if c else i
        target[q], last = tt[i][2], q
    moved = [n for n, i in enumerate(ann) if i not in target]
    if 2 * len(moved) <= len(ann):
        return None, []
    return "\n".join(f"{s} : {u}" + (f" {target[q]}" if q in target else "")
                     for q, (s, u, a) in enumerate(tt)), moved


def listed_neighbours(row, marker):
    """H5: (response, moved slots) — the annotated turns as a numbered list, as many as possible
    written on a nearby unmasked turn of the same speaker instead of their own."""
    tt = turns_of(row["tgt"], marker)
    ann = [i for i, x in enumerate(tt) if x[2]]
    free = [i for i, x in enumerate(tt) if not x[2]]
    target, last = {}, -1
    for i in ann:
        c = [q for q in free if q > last and tt[q][0] == tt[i][0] and abs(q - i) <= 6
             and tt[q][1] != tt[i][1]]          # a turn that reads the same is no misplacement
        q = min(c, key=lambda q: abs(q - i)) if c else i
        target[q], last = tt[i][2], q
    moved = [n for n, i in enumerate(ann) if i not in target]
    if 2 * len(moved) <= len(ann):
        return None, []
    return "\n".join(f"{n + 1}. {tt[q][0]} : {tt[q][1]} {target[q]}" for n, q in enumerate(sorted(target))), moved


# every shape must score every slot: (name, make(row, marker) -> response)
SAME = [
    ("gold", lambda r, mk: r["tgt"]),
    ("reformatted, in prose and a fence", lambda r, mk: reformat(r["tgt"])),
    ("marker dropped", lambda r, mk: r["tgt"].replace(mk + " ", "")),
    ("A1 [MASK] kept before the answer",
     lambda r, mk: ann_re(mk).sub(lambda x: " [MASK]" + x.group(0), r["tgt"])),
    ("A1 [MASK] kept, marker dropped (Gemini roleplay)",
     lambda r, mk: ann_re(mk).sub(lambda x: " [MASK] " + x.group(0).strip()[2:], r["tgt"])),
    ("A2 unfilled window echoed, then the answer",
     lambda r, mk: "Extracted text:\n" + m.window(r["src"]) + "\n\nWith annotations:\n" + r["tgt"]),
    ("A2 the answer, then the unfilled window quoted",
     lambda r, mk: r["tgt"] + "\n\nOriginal:\n" + m.window(r["src"])),
    ("A2 a wrong draft, then the answer", lambda r, mk: bogus(r, mk) + "\n\nFinal:\n" + r["tgt"]),
    ("A2 a wrong draft ended by </think> alone",
     lambda r, mk: bogus(r, mk) + "\n</think>\n" + r["tgt"]),
    ("whole conversation echoed, window answered",
     lambda r, mk: before(r["src"]) + "[start] " + r["tgt"] + " [end]" + after(r["src"])),
    ("A3 items joined by the marker", lambda r, mk: r["tgt"].replace(" ; ", f" {mk} ")),
    ("A3 items joined by 'and'", lambda r, mk: r["tgt"].replace(" ; ", " and ")),
    ("A5 numbered list", lambda r, mk: numbered(r, mk)),
    ("A5 numbered list with speakers (GPT story)", lambda r, mk: numbered(r, mk, speakers=True)),
    ("A5 answers in a row, nothing else (Gemini goal)", lambda r, mk: " ".join(" ".join(g) for g in r["gold"])),
    ("A6 '( )' left off", lambda r, mk: r["tgt"].replace(" ( )", "")),
    ("A6 '->' instead of the marker", lambda r, mk: r["tgt"].replace(mk + " ", "-> ")),
    ("**bold** answers",
     lambda r, mk: ann_re(mk).sub(lambda x: " **" + x.group(0).strip() + "**", r["tgt"])),
    ("F1 label 'Intent:'", lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" Intent: {a}" if a else ""))),
    ("F1 '[Annotation: # x]'",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" [Annotation: {a}]" if a else ""))),
    ("F1 '[MASK: # x]'", lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" [MASK: {a}]" if a else ""))),
    ("'(# x)'", lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" ({a})" if a else ""))),
    ("the answer on a line of its own after a blank line",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f"\n\n{a}" if a else ""), sep="\n\n")),
    ("utterances detokenised (don't, 'Yes, absolutely.')",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s}: {detok(u)}" + (f" {a}" if a else ""))),
    ("'**ARI**:'", lambda r, mk: rebuild(r, mk, lambda s, u, a: f"**{s}**: {u}" + (f" {a}" if a else ""))),
    ("'[ARI]:'", lambda r, mk: rebuild(r, mk, lambda s, u, a: f"[{s}]: {u}" + (f" {a}" if a else ""))),
    ("F2 table: speaker | utterance | annotation",
     lambda r, mk: "| Speaker | Utterance | Annotation |\n|---|---|---|\n"
     + rebuild(r, mk, lambda s, u, a: f"| {s} | {u} | {a or ''} |")),
    ("F2 table of slots, with a '#' column",
     lambda r, mk: "| # | Speaker | Annotation |\n|---|---|---|\n"
     + "\n".join(f"| {i + 1} | x | {' '.join(g)} |" for i, g in enumerate(r["gold"]))),
    ("F2 JSON object", lambda r, mk: "{\n" + ",\n".join(f'  "{i + 1}": "{" ".join(g)}"'
                                                       for i, g in enumerate(r["gold"])) + "\n}"),
    ("F2 JSON list", lambda r, mk: "[\n" + ",\n".join(
        f'  {{"mask": {i + 1}, "annotation": "{" ".join(g)}"}}' for i, g in enumerate(r["gold"])) + "\n]"),
    ("numbered list, then a note after a blank line",
     lambda r, mk: numbered(r, mk) + f"\n\nNote: item 1 could also be read as {mk} acknowledge ( )."),
    ("'MASK 1: # x' lines", lambda r, mk: "\n".join(f"MASK {i + 1}: {' '.join(g)}" for i, g in enumerate(r["gold"]))),
    ("'[MASK] 1 -> # x' lines", lambda r, mk: "\n".join(f"[MASK] {i + 1} -> {' '.join(g)}" for i, g in enumerate(r["gold"]))),
    ("F1 the answer, then an explanation quoting each turn",
     lambda r, mk: r["tgt"] + "\n\nExplanation:\n" + "\n".join(
         f"- {p.replace(' [MASK]', '')} -> here the speaker is reacting to the previous turn." for p in masked_turns(r))),
    ("Qwen: an analysis list with tags, then the answer key", analysis_then_key),
    ("G3 ', I think.' after the answer", per_turn(", I think.")),
    ("G3 ', probably.' after the answer", per_turn(", probably.")),
    ("G3 ', most likely.' after the answer", per_turn(", most likely.")),
    ("G3 ' and that's it.' after the answer", per_turn(" and that's it.")),
    ("G3 ' — I think.' after the answer", per_turn(" — I think.")),
    ("G4 items bare and joined by the marker",
     lambda r, mk: r["tgt"].replace(" ( )", "").replace(" ; ", f" {mk} ")),
    ("G4 items bare, one per line, each with the marker", lambda r, mk: rebuild(
        r, mk, lambda s, u, a: f"{s} : {u}" + ("\n" + "\n".join(mk + " " + x.strip() for x in
                                                               a[1:].replace(" ( )", "").split(" ; "))
                                              if a else ""))),
    ("G4 items bare, joined by '. '", lambda r, mk: r["tgt"].replace(" ( )", "").replace(" ; ", " . ")),
    ("U4 utterances retyped with curly apostrophes", lambda r, mk: rebuild(
        r, mk, lambda s, u, a: f"{s}: {detok(u).replace(chr(39), chr(8217))}" + (f" {a}" if a else ""))),
    ("the answer in its own paragraph after a label",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f"\n\nAnnotation: {a}" if a else ""), sep="\n\n")),
    ("'[x]': the annotation in brackets, no marker",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" [{a[1:].strip()}]" if a else ""))),
    ("U3 '-> affirm': marker and '( )' dropped, a name of the scheme",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" → {a[1:].strip().replace(' ( )', '')}" if a else ""))),
    ("U2 numbered list without the marker",
     lambda r, mk: "\n".join(f"{i + 1}. {' '.join(g[1:])}" for i, g in enumerate(r["gold"]))),
    ("U2 numbered list without the marker, in code",
     lambda r, mk: "\n".join(f"{i + 1}. `{' '.join(g[1:])}`" for i, g in enumerate(r["gold"]))),
    ("U1 numbered answers, then a numbered reference list of the turns", lambda r, mk: numbered(r, mk)
     + "\n\nFor reference, the masked turns were:\n"
     + "\n".join(f"{i + 1}. {p.replace(' [MASK]', '')}" for i, p in enumerate(masked_turns(r)))),
    ("whole conversation echoed, one turn per line, detokenised", lambda r, mk: detok(re.sub(
        r"\s(?=" + SPK + ")", "\n", before(r["src"]) + "[start] " + r["tgt"] + " [end]" + after(r["src"])))),
    ("G1 the answer, then items each quoting the PREVIOUS turn",
     lambda r, mk: r["tgt"] + "\n\nWhy:\n\n" + quote_previous(r, mk)),
    ("G1 only the items quoting the previous turn", quote_previous),
    ("H2 the answer, then numbered notes each citing a contrasting tag",
     lambda r, mk: r["tgt"] + "\n\nNotes:\n" + "\n".join(
         f"{i + 1}. Unlike the context's `{OTHER[mk]}`, this one needed more thought." for i in range(len(r["gold"])))),
    ("G2 the answer, then a note offering a wrong alternative for one turn — the answer stands",
     lambda r, mk: r["tgt"] + note_on(r, mk, len(r["gold"]) // 2, f"{mk} bogus ( )")),
    ("R2 '[MASK] #3 -> # x': a numbered reference before the answer", lambda r, mk: rebuild(
        r, mk, lambda s, u, a: f"{s} : {u}" + (f" [MASK] {mk}3 -> {a}" if a else ""))),
    ("R2 numbered list 'Mask #1: # x'",
     lambda r, mk: "\n".join(f"{i + 1}. Mask {mk}{i + 1}: {' '.join(g)}" for i, g in enumerate(r["gold"]))),
    ("H6 '_' written as a space inside brackets", lambda r, mk: r["tgt"].replace("_", " ")),
    ("H6 '-' written as a space", lambda r, mk: re.sub(r"(?<=[a-z])-(?=[a-z])", " ", r["tgt"])),
    ("H1 a format example in the preamble, then the answers in a row",
     lambda r, mk: f"Using the format seen above, such as `{FORMAT[mk]}`, the answers are:\n\n"
     + " ".join(" ".join(g) for g in r["gold"])),
    ("F3 the answer, then a copy that leaves half blank — a later blank replaces nothing",
     lambda r, mk: r["tgt"] + "\n\nOnly these I am sure of:\n" + re.sub(
         r"(" + re.escape(mk) + r" [^#@]*?)(?=\s" + SPK + r"|$)",
         lambda x, c=[0]: (c.__setitem__(0, c[0] + 1) or (x.group(1) if c[0] % 2 else "[MASK]")), r["tgt"])),
]

FORMAT = {"#": "# intent ( slot ( value ) )", "@": "@ G ( speaker , action ( parameters ) )"}
OTHER = {"#": "# acknowledge ( )", "@": "@ G ( LC , go-to ( reception ) )"}

# a slot answered twice, differently: the dialogue's answer stands, and every such slot is a
# CONFLICT the summary shows — (name, make, lenient expected as a function of slots)
CONFLICT = [
    ("G5 the right turns, then a wrong numbered final list",
     lambda r, mk: r["tgt"] + "\n\nFinal answers:\n" + bogus_list(r, mk), lambda n: n),
    ("G5 wrong turns, then the right numbered final list",
     lambda r, mk: bogus(r, mk) + "\n\nFinal answers:\n" + numbered(r, mk), lambda n: 0),
    ("H2 the right turns, then a numbered note per mask naming the answer 'rather than' another",
     lambda r, mk: r["tgt"] + "\n\nWhy, mask by mask:\n" + "\n".join(
         f"{i + 1}. `{' '.join(g)}` rather than `{mk} bogus ( )`, given the context." for i, g in enumerate(r["gold"])),
     lambda n: n),
    ("H3 the right turns, then a bulleted note per masked turn offering another tag",
     lambda r, mk: r["tgt"] + "\n\nNotes:\n" + "\n".join(
         f"- {p.replace(' [MASK]', '')} could arguably be {mk} bogus ( ) instead." for p in masked_turns(r)),
     lambda n: n),
]

# shapes where a slot may go UNREAD — two window turns that read the same, and the response quotes
# only one — but none may be read WRONG
UNSURE = [
    ("U6 bulleted walk-through quoting each masked turn, then its tag",
     lambda r, mk: "\n".join(f"- `{p}` — the speaker reacts here. **Tag**: `{' '.join(g)}`"
                             for p, g in zip(masked_turns(r), r["gold"]))),
    ("U5 '> LC :' blockquotes, prose before each answer",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"> {s} : {u}" + (f" -> {s} is speaking, so {a}" if a else ""))),
    ("F1 prose, then the answer in the same line",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" -> {s} is speaking here, so {a}" if a else ""))),
    ("'1)' items quoting speaker and utterance",
     lambda r, mk: "\n".join(f'{i + 1}) {s}: "{u}" -> {a}'
                             for i, (s, u, a) in enumerate(x for x in turns_of(r["tgt"], mk) if x[2]))),
    ("F1 walk-through quoting each masked turn, then its tag",
     lambda r, mk: "Let me go through each [MASK]:\n\n" + "\n".join(
         f"{i + 1}. `{p}` — the speaker's act here is clear. **Tag**: `{' '.join(g)}`"
         for i, (p, g) in enumerate(zip(masked_turns(r), r["gold"])))),
]


def last_blank(row, marker, tail):
    """The window answered except its last slot; `tail(row)` follows the window."""
    if not m.window(row["src"]).rstrip().endswith("[MASK]"):
        return None, []
    last = list(ann_re(marker).finditer(row["tgt"]))[-1]
    return row["tgt"][:last.start()] + tail(row), [len(row["gold"]) - 1]


def extra(sep):
    def make(row, marker):
        add = sep.format(mk=marker)
        return ann_re(marker).sub(lambda x: x.group(0) + add, row["tgt"]), range(len(row["gold"]))
    return make


# no slot in `targets` may be credited: (name, make(row, marker) -> (response, targets))
NEVER = [
    ("unfilled window", lambda r, mk: (m.window(r["src"]), range(len(r["gold"])))),
    ("empty response", lambda r, mk: ("", range(len(r["gold"])))),
    ("whole conversation echoed, window unanswered — context annotations stay context",
     lambda r, mk: (r["src"].replace("[MASK]", ""), range(len(r["gold"])))),
    ("whole conversation echoed, window wrong",
     lambda r, mk: (before(r["src"]) + "[start] " + bogus(r, mk) + " [end]" + after(r["src"]),
                    range(len(r["gold"])))),
    ("only the context echoed", lambda r, mk: (before(r["src"]) + after(r["src"]), range(len(r["gold"])))),
    ("the answer, then a wrong final version — the later is read",
     lambda r, mk: (r["tgt"] + "\n\nCorrected:\n" + bogus(r, mk), range(len(r["gold"])))),
    ("A3 extra item joined by the marker", extra(" {mk} bogus ( )")),
    ("A3 extra item juxtaposed", extra(" bogus ( )")),
    ("A3 extra item joined by 'and'", extra(" and bogus ( )")),
    ("A3 extra item joined by ', and also'", extra(" , and also bogus ( )")),
    ("A3 hedge 'X or Y'", extra(" or bogus ( )")),
    ("A4 last slot blank, context after the window echoed",
     lambda r, mk: last_blank(r, mk, lambda row: " [end]" + after(row["src"]))),
    ("A4 last slot blank, a note after it quotes the answer",
     lambda r, mk: last_blank(r, mk, lambda row: "\n\nNote: the last turn could be "
                              + " ".join(row["gold"][-1]) + " but it is unclear, so I left it.")),
    ("A5 numbered list, one item hedged",
     lambda r, mk: (re.sub(r"`\n", f" or {mk} bogus ( )`\n", numbered(r, mk), count=1), [0])),
    ("F3 the answer as a draft, then a wrong final with one slot blank",
     lambda r, mk: (r["tgt"] + "\n\nFinal answer:\n" + blank_last(bogus(r, mk), mk), range(len(r["gold"]) - 1))),
    ("F4 hedge on the next line", lambda r, mk: (rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (
        f" {a}\n  Alternatively: {mk} bogus ( )" if a else "")), range(len(r["gold"])))),
    ("F4 hedge in brackets", lambda r, mk: (rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (
        f" {a} (or possibly {mk} bogus ( ))" if a else "")), range(len(r["gold"])))),
    ("F4 hedge after a dash", lambda r, mk: (rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (
        f" {a} — maybe {mk} bogus ( )" if a else "")), range(len(r["gold"])))),
    ("F4 'either # x or # y'", lambda r, mk: (rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (
        f" either {a} or {mk} bogus ( )" if a else "")), range(len(r["gold"])))),
    ("H5 a numbered list of the annotated turns, each on a nearby unmasked turn of its speaker",
     lambda r, mk: listed_neighbours(r, mk)),
    ("G2 wrong answers, then a note offering the right one for one turn, 'but I kept the above'",
     lambda r, mk: (bogus(r, mk) + note_on(r, mk, len(r["gold"]) // 2, " ".join(r["gold"][len(r["gold"]) // 2]),
                                          ", but I kept the above"), [len(r["gold"]) // 2])),
    ("G6 each annotation on a nearby unmasked turn of the same speaker", moved_to_neighbour),
]

# responses that give no answer: every prediction must be None, not merely uncredited
UNREAD = [
    ("unfilled window", lambda r, mk: m.window(r["src"])),
    ("empty response", lambda r, mk: ""),
    ("only the context echoed (goal 5: three context annotations for three slots)",
     lambda r, mk: before(r["src"]) + after(r["src"])),
    ("whole conversation echoed, window unanswered", lambda r, mk: r["src"].replace("[MASK]", "")),
    ("F1 musings quoting each masked turn, no tags",
     lambda r, mk: "\n".join(f"- `{p}`: hmm, let me think about this one." for p in masked_turns(r))),
    ("Qwen's markdown musings: `* then [MASK] -> **ARI** says`",
     lambda r, mk: "\n".join(f"* {p}\n* then `[MASK]` -> **ARI** says \"okay\"" for p in masked_turns(r))),
    ("H1 Gemini goal/story: a format example in the preamble, then one numbered answer short",
     lambda r, mk: f"Using the format seen earlier, such as `{FORMAT[mk]}`, here are the masks:\n\n"
     + "\n".join(f"{i + 1}. **[MASK] {i + 1}:** `{' '.join(g)}`" for i, g in enumerate(r["gold"][:-1]))),
    ("H4 '# x rather than # y': a rejection the reader cannot settle",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" {closed(a)} rather than {mk} bogus ( )" if a else ""))),
    ("H4 '# x (not # y)'",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u}" + (f" {closed(a)} (not {mk} bogus ( ))" if a else ""))),
    ("F1 labels and prose where the answers go",
     lambda r, mk: rebuild(r, mk, lambda s, u, a: f"{s} : {u} [Annotation: here the speaker reacts]")),
    ("each answer written on the turn before its own, utterances summarised",
     lambda r, mk: "\n".join(f"{s}: (speaks) {a or ''}" for (s, _, _), (_, _, a) in zip(
         turns_of(r["tgt"], mk), turns_of(r["tgt"], mk)[1:] + [("", "", None)]))),
]


def main():
    for task, slots in (("dst", 389), ("goal", 225)):
        # the gold's own unbalanced brackets: 11 slots, nine left open and two with an extra `)`
        # (dst dialogue 5). The reader skips a `)` with nothing open, so `exact` — the tokens
        # read — misses those two even for the gold itself; `lenient` misses none
        typos = sum(g.count(")") > g.count("(") for r in m.load_rows(task) for g in r["gold"])
        gold = total(task, lambda r, mk: r["tgt"])
        assert gold == {"slots": slots, "lenient": slots, "exact": slots - typos, "missing": 0,
                        "unaligned": 0, "conflict": 0, "revised": 0}, (task, gold, typos)
        for name, make in SAME:
            got = total(task, make)
            assert got["lenient"] == slots and got["missing"] == 0, (task, name, got)
        loose = total(task, lambda r, mk: reformat(r["tgt"]))
        assert loose["exact"] < slots, (task, loose)         # the strict count does see the change
        print(f"{task}: {len(SAME)} shapes of the gold, each {slots}/{slots} lenient")

        for name, make, expect in CONFLICT:
            got = total(task, make)
            assert got["lenient"] == expect(slots) and got["missing"] == 0, (task, name, got)
            # a second answer that differs is counted; a note that names no answer (H2's items
            # hold two tags with words between: unread) is not a second answer
            assert (got["conflict"] > 0) == name.startswith(("G5", "H3")), (task, name, got)
        print(f"{task}: {len(CONFLICT)} responses answering twice, the dialogue's answer kept, "
              f"the disagreement counted")

        for name, make in UNSURE:
            got = total(task, make)
            assert got["lenient"] + got["missing"] == slots and got["missing"] <= slots // 50, \
                (task, name, got)
        print(f"{task}: {len(UNSURE)} shapes with repeated turns, none read wrong")

        for name, make in UNREAD:
            for r in m.load_rows(task):
                preds, _ = m.predicted_slots(r["src"], make(r, m.TASKS[task]["marker"]),
                                             m.TASKS[task]["marker"])
                assert preds == [None] * len(r["gold"]), (task, name, r["idx"], preds)
        print(f"{task}: {len(UNREAD)} responses without answers, nothing read")
        for name, make in NEVER:
            hits, n = credited(task, make)
            assert hits == 0, (task, name, hits, n)
        print(f"{task}: {len(NEVER)} responses that do not give the answer, none credited")

    # R1: a DST table with one intent the model invented loses that slot only
    rows = m.load_rows("dst")
    table = total("dst", lambda r, mk: "| # | Annotation |\n|---|---|\n" + "\n".join(
        f"| {i + 1} | {'# express-disappointment ( )' if i == 0 else ' '.join(g)} |" for i, g in enumerate(r["gold"])))
    assert table["lenient"] == 389 - len(rows) and table["missing"] == 0, table
    # R3: a summary list shifted by one conflicts, so it fills nothing — the blank slot stays unread
    for task in m.TASKS:
        mk = m.TASKS[task]["marker"]
        for r in m.load_rows(task):
            if len(r["gold"]) < 3:
                continue
            s_ = len(r["gold"]) // 2
            anns = list(ann_re(mk).finditer(r["tgt"]))
            body = r["tgt"][:anns[s_].start()] + " [MASK]" + r["tgt"][anns[s_].end():]
            shifted = r["gold"][1:] + r["gold"][:1]
            resp = body + "\n\nSummary:\n" + "\n".join(f"{i + 1}. `{' '.join(g)}`" for i, g in enumerate(shifted))
            preds, _ = m.predicted_slots(r["src"], resp, mk)
            assert preds[s_] is None and m.score_row(r, resp, mk)["conflict"] > 0, (task, r["idx"])
    # R4, a documented misread: a later full copy that is not the answer replaces it — `revised` shows it
    for task, slots in (("dst", 389), ("goal", 225)):
        got = total(task, lambda r, mk: r["tgt"] + "\n\nA common mistake would be:\n" + bogus(r, mk))
        assert got["lenient"] == 0 and got["revised"] == slots, (task, got)
    print("R1 an invented intent costs its slot only; R3 a shifted list fills nothing; R4 shows as revised")

    # content is not forgiven: one wrong intent, one wrong speaker
    row = m.load_rows("dst")[3]
    s = m.score_row(row, row["tgt"].replace("# thank ( )", "# greet ( )", 1), "#")
    assert s["lenient"] == s["slots"] - 1, s
    row = next(r for r in m.load_rows("goal") if "G ( LC ," in r["tgt"].split("@", 1)[1])
    s = m.score_row(row, row["tgt"].replace("G ( LC ,", "G ( RP ,", 1), "@")
    assert s["lenient"] == s["slots"] - 1, s
    assert m.canonical(["#", "request", "(", "a", ")", ";", "inform", "(", "b", ")"]) == \
        m.canonical(["inform", "(", "b", ")", ",", "request", "(", "a", ")"])
    assert m.canonical(["#", "thank"]) == m.canonical(["#", "thank", "(", ")"])
    assert m.canonical(["@", "g", "(", "lc", ",", "eat", ")"]) == \
        m.canonical(["@", "g", "(", "lc", ",", "eat", "(", ")", ")"])
    assert m.canonical(["#", "affirm", "(", ")", "#", "bye", "(", ")"]) != m.canonical(["#", "affirm", "(", ")"])
    # F6: a nested empty `( )` is form — the gold never writes both, so no two answers merge
    assert m.canonical("# greet ( ) ; request ( help ( ) )".split()) == \
        m.canonical("# greet ( ) ; request ( help )".split())
    print("content checks: a changed intent and a changed speaker each cost exactly one slot")

    # the summary flags a style whose slots went unread
    rows = [{"task": "dst", "style": "basic", "idx": r["idx"], "status": "ok", "response": ""}
            for r in m.load_rows("dst")]
    assert m.summarize(rows)["dst"]["check"] == ["basic"]
    print("ALL PASS")


if __name__ == "__main__":
    main()
