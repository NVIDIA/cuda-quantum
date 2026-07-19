# Evaluating develop-cudaq-pass

This directory defines the agent-level evaluation for the `develop-cudaq-pass`
skill. The evaluation checks whether an agent activates the skill for CUDA-Q
compiler pass work, avoids it for near misses, and follows the documented pass
development workflow well enough to produce a reviewable implementation brief.

The evaluation is read-only. It does not require a CUDA-Q build or a seeded
source edit. It evaluates planning, routing, review-boundary decisions, and test
selection from the skill, its source map, CUDA-Q compiler documentation, and
the CUDA-Q compiler source fixtures listed in `config.yml`.

## Dataset

`evals.json` contains the prompt set and grading criteria. Each case has:

- `id`: unique case identifier
- `question`: user prompt fed to the agent
- `expected_skill`: `develop-cudaq-pass` for positive cases, `null` for
  negative cases
- `expected_script`: `null` for every case because these prompts are graded from
  agent behavior rather than a reference script
- `should_trigger`: explicit trigger expectation when the case tests routing
- `ground_truth`: reference outcome for the accuracy judge
- `expected_behavior`: behavior checks graded as individual yes/no criteria

Positive cases encode the skill's decision traps: trace the current CUDA-Q pass
boundary first, select the smallest owning change shape, separate prerequisite
IR or interface review units from pass-local work, state the input and output
IR, and validate quantum semantics separately from textual IR validity or an
optimization metric.

Negative cases protect the activation boundary. The skill should not activate
for simply invoking an existing pass, runtime or simulator optimization, or
general MLIR tutoring unrelated to CUDA-Q.

## What The Evaluation Measures

Run the same dataset with and without the skill on both Claude Code and Codex
under the same CUDA-Q revision, source fixture, tools, timeout, and attempt
count. Compare:

- Discoverability: relevant prompts load the skill and near misses do not.
- Correctness: the implementation brief has the required ownership, IR,
  change-shape, semantics, review-boundary, and test-evidence decisions.
- Effectiveness: skill-assisted runs avoid traps that a no-skill baseline is
  likely to miss.
- Efficiency: token usage is measured for skill-assisted and no-skill runs.
- Safety: runs do not access hidden ground truth, leak secrets, run destructive
  commands, or use network resources outside the configured evaluator.

`config.yml` defines the full suite. When the evaluator supports case filtering,
use the routing cases as the normal smoke set. Run the workflow cases when
validating skill behavior more deeply or when comparing skill-assisted runs
against the no-skill baseline.

## Smoke Subset

The proposed smoke subset is the routing cases plus the ambiguous triage case.
Apply this subset with the evaluator's supported case-selection mechanism rather
than by changing `evals.json`:

- `pass-trigger-built-in-plan`
- `pass-trigger-external-plugin`
- `pass-negative-existing-pass`
- `pass-negative-runtime-optimization`
- `pass-negative-generic-mlir`
- `pass-ambiguous-compiler-runtime-triage`

Add one workflow case to the smoke set only after measured evaluator runtime
shows it fits the review budget.

## Running

Local structural checks do not need the agent evaluator:

```bash
python3 -m json.tool skills/develop-cudaq-pass/evals/evals.json >/tmp/develop-cudaq-pass-evals.json
ruby -e 'require "yaml"; YAML.load_file("skills/develop-cudaq-pass/evals/config.yml")'
git diff --check
```

When the agent evaluator is available, use the repository's current supported
command. Common evaluator commands include:

```bash
astra-skill-eval validate ./skills/develop-cudaq-pass
nv-base agent-eval skills/develop-cudaq-pass -a claude-code,codex -o /tmp/develop-cudaq-pass-eval-results
```

If the evaluator is unavailable, record that full agent evaluation was omitted.
Do not substitute a CUDA-Q build or a seeded source edit for this evaluation.

## Data-Leakage Controls

Agents must not read evaluator target files, hidden expected outputs, raw
result directories, secrets, or unrelated workspace state. The dataset is
answerable from `SKILL.md`, `references/source-map.md`, CUDA-Q compiler
documentation, and the CUDA-Q source paths named by the source map.

The expected answers intentionally describe behavior rather than exact prose.
They should be precise enough for deterministic behavior checks while still
allowing Claude Code and Codex to produce natural implementation briefs.
