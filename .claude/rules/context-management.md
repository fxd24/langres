# Context Management & Parallelization

**Protect the context window. Maximize parallelism.**

## Delegate Output-Heavy Operations to Subagents

When an operation produces large output (git diff, test suite results, large file reads, database query results), **delegate it to a subagent** instead of running it in the main context. The subagent processes the output and returns only a summary. This prevents the main orchestrator's context from filling with data that is only needed transiently.

**Pattern:**
1. Launch a Task agent to perform the output-heavy operation
2. The agent reads/processes the large output in its own context
3. The agent returns a concise summary to the orchestrator
4. The orchestrator acts on the summary, not the raw data

**Examples of operations that should use subagents:**
- `git diff` when diff is expected to be >50 lines
- `make test` output when failures occur (launch agent to analyze and summarize failures)
- Reading multiple large files for context (launch Explore agent to synthesize findings)
- Database query results with many rows (launch agent to analyze and report key findings)

**Anti-pattern:** Running `git diff` 3 times across a workflow, each time dumping 200+ lines into the orchestrator context.

## Parallelize Independent Work

Before starting any multi-step task, evaluate whether it can be decomposed into independent parallel chunks. If pieces of work don't depend on each other, launch them as parallel subagents to minimize wall-clock time.

**Checklist before execution:**
1. Can this task be split into 2+ independent pieces? (e.g., tests + implementation skeleton, or multiple independent file changes)
2. Are there operations that can run concurrently? (e.g., multiple TDD agents for independent tasks in the same wave)
3. Can review agents run in parallel? (internal code-reviewer + external models)

**Anti-pattern:** Running tasks sequentially when they have no dependencies on each other.
