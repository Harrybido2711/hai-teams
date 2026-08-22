#!/usr/bin/env bash
# Pull NegotiationToM results down from Quest. Never commits, never pushes.
#
# The hourly pull used to be part of an unattended watcher that also auto-committed and pushed. That
# watcher swept unreviewed files into commits named "watcher checkpoint" and fought with interactive
# work, so it was killed — and the pull went with it, leaving four full runs (56,774 rows) living
# only on the cluster. Those are two different jobs: protecting data against loss should be
# automatic, publishing it should not.
#
#   bash .claude/scripts/pull_quest_results.sh          # pull once
#   bash .claude/scripts/pull_quest_results.sh --watch  # pull every 30 min, log each cycle
#
# Results flow Quest -> local only. Nothing here writes to Quest.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$REPO/Interpersonal_processes_benchmarks/NegotiationToM"
QDIR=/gpfs/projects/p32983/Interpersonal_processes_benchmarks/NegotiationToM
LOG="${PULL_LOG:-$REPO/quest_pull.log}"
INTERVAL="${PULL_INTERVAL:-1800}"

pull_once() {
  local stamp before after transferred
  stamp=$(date '+%Y-%m-%d %H:%M:%S')
  before=$(find "$LOCAL" -path '*/results/*' -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')

  # rsync, not tar: the result tree is ~90 MB and a tar pipe re-sends all of it every cycle, which
  # took over two minutes and would repeat every half hour for the sake of a few new rows. rsync's
  # delta transfer sends only the differing blocks of a changed file, which is exactly right for
  # append-only .jsonl checkpoints.
  #
  # No --append-verify: macOS still ships rsync 2.6.9 (2006), which rejects it. The flag was in here
  # briefly and every pull failed silently, because stderr went to /dev/null and the `if` tested the
  # exit status of the awk at the end of the pipe rather than rsync's.
  #
  # No --delete either: archives created on Quest must not vanish locally, and nothing in this
  # script should be able to remove local data.
  local out rc
  out=$(rsync -az --stats \
        -e "ssh -o BatchMode=yes -o ConnectTimeout=30" \
        --include='NEG_*/' --include='NEG_*/results/***' \
        --include='NEG_*/log_*.txt' --include='NEG_*/*HALT*.txt' --exclude='*' \
        "quest:$QDIR/" "$LOCAL/" 2>&1)
  rc=$?
  out=$(printf '%s\n' "$out" | grep -v libcrypto)

  if [ "$rc" -ne 0 ]; then
    # Loud on failure: a pull that silently stops is indistinguishable from one with nothing to do.
    echo "[$stamp] PULL-FAILED rc=$rc — results still live only on Quest"
    printf '%s\n' "$out" | tail -3
    return 1
  fi

  after=$(find "$LOCAL" -path '*/results/*' -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
  transferred=$(printf '%s\n' "$out" | awk -F': ' '/Total transferred file size/ {print $2}')
  echo "[$stamp] pulled ok — rows $before -> $after (+$((after - before))), transferred ${transferred:-0 bytes}"
}

if [ "${1:-}" = "--watch" ]; then
  echo "pulling every ${INTERVAL}s -> $LOG"
  while true; do
    pull_once >> "$LOG" 2>&1
    sleep "$INTERVAL"
  done
else
  pull_once
fi
