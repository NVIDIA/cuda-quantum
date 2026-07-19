# Evaluating develop-cudaq-pass

This directory defines the agent-level evaluation for the `develop-cudaq-pass`
skill. The evaluation checks whether an agent activates the skill for CUDA-Q
compiler pass work, avoids it for near misses, and follows the documented pass
development workflow well enough to plan or implement a reviewable change.

Most cases are read-only and evaluate planning, routing, review-boundary
decisions, and test selection. One case asks the agent to create and test a
complete out-of-tree pass plugin against a CUDA-Q development environment. It
does not provide starter source or tests.

## Dataset

`evals.json` contains the prompt set and grading criteria. Each case has:

- `id`: unique case identifier
- `question`: user prompt fed to the agent
- `expected_skill`: `develop-cudaq-pass` for positive cases, `null` for
  negative cases
- `expected_script`: `null` because the cases are graded from agent behavior;
  the blocked source-editing pilot did not validate a custom build grader
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

Keep `pass-source-edit-external-hxh` scheduled or manual unless its measured
runtime, resource use, and stability satisfy the source-editing smoke budget.

## Source-Editing Case

`pass-source-edit-external-hxh` asks the agent to create an out-of-tree plugin
under `hxh-plugin/` that registers `cudaq-hxh-to-z` and implements the exact
single-qubit identity `H X H = Z`. The agent must create the project, pass, and
tests without a supplied scaffold. The task deliberately leaves representation,
modifier, control, use-def, and unsupported-input boundaries for the agent to
derive from current CUDA-Q documentation and source.

A source-editing qualification must stage the latest
`ghcr.io/nvidia/cuda-quantum-dev:amd64-gcc12-main` image, resolve its digest
once per run, and use that resolved image for every comparable attempt. The
moving tag is not permanently pinned. Before running an agent, verify privately
that the image can compile and load an external pass without placing that probe
in the agent workspace. The current `config.yml` does not yet wire this CUDA-Q
image into the evaluator environment.

A completed source-editing qualification must add evaluator-owned inputs after
the agent exits and check plugin loading, the expected three-to-one gate rewrite,
exact CircuitCheck equivalence, conservative behavior outside the claimed
support, and the absence of CUDA-Q production-source or pipeline changes. The
grader inputs and expected results must not be present in the agent workspace.
Inspect the trajectory separately for a failing behavioral test before
implementation and for focused agent-authored correctness tests.

### Stage 6 Pilot Evidence

The 2026-07-19 pilot resolved the moving tag to OCI index
`sha256:99a27ebc1177897fa042a731def0e2d29fde0c2d703ebd185ca104ff7790fa65`
and Linux/amd64 manifest
`sha256:026b7205dc92b49d56cb45e9ca00ce89de0c6b53477278e07ac9d31dce985ba2`.
The image was created on 2026-07-17 from CUDA-Q revision
`f33819860c7c65a55683e1a579ff408e9370167c`. Its configuration advertises a
debug build at `/workspaces/cuda-quantum/build`, install prefix
`/usr/local/cudaq`, and ccache rooted at `/root/.ccache`.

Local staging did not complete. The first pull downloaded the other image
layers but did not finish the final warm-build layer. A second pull requested
only that layer and remained incomplete for more than 30 minutes. The unresolved
compressed layer was
`sha256:3e50c1ec6811a94a54e94b92fd2e04d24d4af2b5aff884bf8db294a4305c7ad8`
at 9,156,764,020 bytes. After both attempts, `docker image inspect` still
reported that the tagged image was absent. Both pulls were stopped without
removing existing Docker data.

Because the image was not runnable, the pilot could not verify writable Harbor
staging, external plugin compilation, warm-build or ccache reuse, immutable
grader staging, agent authentication inside the image, or build and test
resource use. No Claude Code or Codex source-editing attempt was run. This is an
infrastructure-blocked pilot, not a failed agent result.

No custom grader or source-editing pass threshold is committed from this run.
The default semantic grader can assess the requested workflow, but it is not
deterministic evidence that the authored plugin builds or preserves the intended
unitary. Connect and validate the private build grader only after container
staging succeeds.

The source-editing case therefore remains manual or scheduled. Pull request
smoke remains limited to trigger and read-only workflow cases. Re-run the pilot
where the 9.16 GB layer is pre-cached or can be downloaded within the job setup
budget, then measure setup separately from the agent, plugin build, and grader
before considering source editing for pull request smoke.

## Running

Local structural checks do not need the agent evaluator:

```bash
python3 -m json.tool skills/develop-cudaq-pass/evals/evals.json >/tmp/develop-cudaq-pass-evals.json
ruby -e 'require "yaml"; YAML.load_file("skills/develop-cudaq-pass/evals/config.yml")'
git diff --check
```

Install the current evaluator from a reviewed revision, then run its offline
checks without a provider key:

```bash
uv tool install --python 3.13 \
  "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git@84bc44bfb17df63e43d0ba1ffb47b494512c8319"
skillevaluator quality-check ./skills/develop-cudaq-pass
skillevaluator validate ./skills/develop-cudaq-pass --no-dedup
```

Tier 3 needs an evaluator provider credential, agent credentials, and a
sandbox. The current stock command evaluates every dataset entry, so do not run
it against this committed dataset until the source-editing staging prerequisite
under Data-Leakage Controls is satisfied. Once that control is in place, copy
the surrounding CUDA-Q repository rather than using `skill_workspace.include`,
which accepts additional skill directories, not arbitrary source paths:

```bash
skillevaluator tier3 evaluate ./skills/develop-cudaq-pass \
  --agents claude-code,codex --env-mode docker --copy-repo
```

If required credentials or container staging are unavailable, record the exact
blocker. Do not present structural checks or a private infrastructure probe as
a completed agent evaluation.

## Data-Leakage Controls

Agents must not read evaluator target files, hidden expected outputs, raw
result directories, secrets, or unrelated workspace state. The dataset is
answerable from `SKILL.md`, `references/source-map.md`, CUDA-Q compiler
documentation, and the CUDA-Q source paths named by the source map. Repository
staging for a source-editing case must exclude `evals.json`, evaluator-owned
grader inputs, and result directories from the agent-visible workspace.

The reviewed SkillEvaluator revision copies the complete source skill and, with
`--copy-repo`, the tracked repository. Its current ignore rules do not remove
`evals/evals.json`. Do not run `pass-source-edit-external-hxh` through that
stock staging path. First produce and inspect a sanitized task package that
contains the installed skill and required CUDA-Q source but omits the dataset,
grader inputs, and results. Treat that verified staging change as a prerequisite
for the source-editing qualification.

The expected answers intentionally describe behavior rather than exact prose.
They should be precise enough for deterministic behavior checks while still
allowing Claude Code and Codex to produce natural implementation briefs.
