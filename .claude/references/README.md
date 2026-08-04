# References

Detail that agents load **on demand** instead of carrying in every prompt.

An agent definition holds its role, its judgement, and the rules that apply to every one of its
tasks. Everything else lives here. `executor.md` was 13 KB and entered the context in full even when
the task was a one-line `scancel`; splitting it means a `scancel` costs the routing table and
nothing more, while a new runner still gets every line of the provider notes.

**The routing table is not advisory.** Read the reference for every row that matches your task
before you act. These files are not background reading — each line in them was paid for by a run
that failed.

| If the task involves | Read |
|---|---|
| Anything at all, before deciding what is true | [shared-context.md](shared-context.md) — repo layout, and which committed doc is authoritative on what |
| Being handed work by the planner, or handing work on | [handoffs.md](handoffs.md) — what a dispatch must carry, what each agent must return |
| SSH, transferring files, `md5sum`, `sbatch`, `scancel`, partitions, sharding | [quest-cluster.md](quest-cluster.md) |
| Choosing or debugging a provider client, timeouts, empty responses, reasoning budgets | [provider-gotchas.md](provider-gotchas.md) |
| Writing or changing an eval script — retries, checkpoints, normalisation, scoring | [script-skeleton.md](script-skeleton.md) |
| Considering borrowing a pattern from an outside repo, or checking whether one was already screened and rejected | [external-patterns.md](external-patterns.md) — sweep history: adopted/to-adapt proposals (none applied without a human decision), rejected patterns, repos screened out and why, what's still unexamined |

## Keeping these honest

When a run exposes something a reference should have said, edit the reference. A fact that lives
only in a session transcript is lost at the end of that session; a fact in here reaches every future
agent that reads the matching row.

Two failure modes to avoid:

- **A reference nothing routes to.** If you add a file, add its row to the table above in the same
  edit, phrased as the condition an agent can recognise in its own task — not as a topic name.
- **A fact in two places.** When something moves here, delete it from the agent definition rather
  than leaving a summary behind. Two copies drift, and the agent then has to decide which is right.
