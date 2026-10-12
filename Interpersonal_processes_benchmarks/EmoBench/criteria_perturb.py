"""Criteria 5 (robustness) perturbations for EmoBench — .claude/references/evaluation-criteria.md.

EmoBench has no shared core, so this module is the one place the perturbations live; every runner
that takes --perturb imports it rather than copying it, or two models' "shuffle" could drift apart
and their flip rates would stop being comparable.

Only text this project controls moves:
  shuffle    — the order of each choice list. EA has one list; EU has two (emotion, cause), each
               reordered independently. The runner derives the gold letter from the shuffled list
               (`choices.index(label)`), so gold content is never rewritten.
  paraphrase — the instruction paragraph of upstream's system prompt, meaning kept. The scenario,
               the question stems and the JSON answer format are left verbatim, so a flip is a
               changed answer rather than a parse failure.
"""
import random

PERTURBATIONS = ("shuffle", "paraphrase")

# The "# Instructions" paragraph of src/configs/prompts.yaml `sys.en`, reworded. Headers kept, so
# the response-format block the runner appends after "# Output" reads exactly as it does unperturbed.
PARAPHRASE_SYS_EN = (
    "# Instructions\n"
    "\n"
    "You will read a short scenario, followed by a question and several answer options.\n"
    "Put yourself in the position of the person the scenario is about and think through what "
    "happened to them.\n"
    "Then choose the option that best captures how that person sees the situation or how they "
    "would feel.\n"
    "\n"
    "# Output\n"
)

CHOICE_LISTS = {"EA": ("choices",), "EU": ("emotion_choices", "cause_choices")}


def shuffled(values, seed, *key):
    """A seeded reordering that always differs from the original and is the same for every run and
    every model: the seed is the perturbation seed plus the item's own identity, never a counter."""
    rng = random.Random("|".join([str(seed), *map(str, key)]))
    order = list(values)
    while order == list(values):
        rng.shuffle(order)
    return order


def shuffle_sample(sample, task, seed):
    """A copy of `sample` with each of its choice lists independently reordered."""
    out = dict(sample)
    for name in CHOICE_LISTS[task]:
        out[name] = shuffled(sample[name], seed, task, sample["qid"], name)
    return out
