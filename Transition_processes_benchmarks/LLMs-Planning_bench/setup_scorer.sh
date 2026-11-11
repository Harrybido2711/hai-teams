#!/usr/bin/env bash
# Builds what plan_scorer.py needs on THIS machine, inside this folder, both gitignored:
#   planner_tools/VAL/build/<os>/validate   — VAL compiled from the vendored source (non-Linux only;
#                                             on Linux the shipped planner_tools/VAL/validate is used)
#   .scorer-env/                            — Python 3.12 with tarski 0.7.0, pyyaml, numpy
#
# Why each piece is needed (planbench.md § Scoring):
#   - The shipped `validate` is a Linux x86-64 ELF. Built here into build/<os>/ — never in place,
#     because the Makefile writes `validate` into its own directory and would overwrite it.
#   - tarski 0.7.0 pins antlr4-python3-runtime 4.7.2, which imports `typing.io`; Python 3.13 removed
#     it. 3.12 is the newest interpreter it runs on. conda -p keeps the env inside this folder.
#
# Then: .scorer-env/bin/python plan_scorer.py --selftest
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
VAL_DIR="$ROOT/planner_tools/VAL"

if [ "$OS" = "linux" ]; then
    echo "linux: plan_scorer.py uses the shipped $VAL_DIR/validate"
else
    B="$VAL_DIR/build/$OS"
    rm -rf "$B"
    mkdir -p "$B"
    cp -R "$VAL_DIR/src" "$VAL_DIR/include" "$VAL_DIR/Makefile" "$VAL_DIR/Make.files" "$VAL_DIR/Make.header" "$B/"
    echo "building VAL in $B ..."
    (cd "$B" && make validate CXX=clang++ CXXFLAGS="-O2 -std=c++98 -w -Iinclude -Isrc/Parser" -j8 > build.log 2>&1) \
        || { echo "VAL build failed — see $B/build.log"; exit 1; }
    "$B/validate" 2>&1 | head -1 || true
    echo "VAL: $B/validate"
fi

ENV="$ROOT/.scorer-env"
if [ ! -x "$ENV/bin/python" ]; then
    if [ -n "${PYTHON:-}" ]; then
        "$PYTHON" -m venv "$ENV"
    elif command -v conda > /dev/null; then
        conda create -y -q -p "$ENV" python=3.12 > /dev/null
    else
        echo "need conda, or PYTHON=/path/to/python3.9-3.12"; exit 1
    fi
fi
"$ENV/bin/python" -m pip install -q tarski==0.7.0 pyyaml numpy
"$ENV/bin/python" -c "import tarski.io, yaml, numpy; print('scorer env ok:', tarski.__version__)"
