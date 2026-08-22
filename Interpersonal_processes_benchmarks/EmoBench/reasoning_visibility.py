"""Resolve visible-reasoning mode from the benchmark's own README, not from a literal in a runner.

Rule 3 of `.claude/references/model-parameters.md`: whether reasoning is *shown* is the benchmark's
decision, read from its own README. A runner that hardcodes True or False has quietly made that
decision on the benchmark's behalf, and the next benchmark copied from it inherits the wrong answer.
So nothing here ships a default: the value is read from the README at run time, together with the
line it was read from, and both travel onto every result row.

Rule 1 is untouched by any of this. Hidden thinking is capped unconditionally, whatever this returns
-- the two halves have different economics and are decided separately.

When the README does not declare a default, this raises. That is deliberate. Guessing is the failure
this module exists to prevent, and an operator who knows the answer can always pass the flag
explicitly.
"""
import os
import re

# A CoT/visible-reasoning flag, however the upstream project spells it.
_FLAG = re.compile(r"--(?:use[-_]?cot|cot|chain[-_]?of[-_]?thought|reasoning)\b", re.I)
# "Defaults to `False`." / "default: true" / "Default = False"
_DEFAULT = re.compile(r"defaults?\s*(?:to|:|=)\s*[`'\"*]*(true|false)\b", re.I)


class ReasoningVisibilityUndeclared(Exception):
    """The README names no default, so the runner must be told explicitly."""


def resolve(benchmark_root, readme_name="README.md"):
    """Return (use_cot: bool, evidence: str) read from the benchmark folder's README.

    `benchmark_root` is the vendored benchmark directory -- the one holding the upstream README,
    not the per-model runner subdirectory.
    """
    path = os.path.join(benchmark_root, readme_name)
    if not os.path.exists(path):
        raise ReasoningVisibilityUndeclared(
            "no {} in {} -- cannot read the benchmark's decision on visible reasoning. "
            "Pass --use-cot or --no-use-cot explicitly.".format(readme_name, benchmark_root))

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    for i, line in enumerate(lines):
        if not _FLAG.search(line):
            continue
        # The default is normally on the same line; allow the next one for wrapped prose.
        window = line if _DEFAULT.search(line) else " ".join(lines[i:i + 2])
        hit = _DEFAULT.search(window)
        if hit:
            use_cot = hit.group(1).lower() == "true"
            evidence = "{}:{}: {}".format(readme_name, i + 1, line.strip())
            return use_cot, evidence

    raise ReasoningVisibilityUndeclared(
        "{} names no default for a chain-of-thought flag. Read it yourself and pass --use-cot or "
        "--no-use-cot; do not let the runner guess.".format(path))
