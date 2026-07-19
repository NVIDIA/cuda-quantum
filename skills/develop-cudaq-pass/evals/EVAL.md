# Eval guidance for develop-cudaq-pass

Developer guidance for maintaining and refining `evals.json`.

## Questions

- Plan a built-in CUDA-Q compiler pass from a requested optimization.
- Plan an out-of-tree CUDA-Q pass plugin without changing production pipelines.
- Identify prerequisite IR or interface changes before pass implementation.
- Select focused IR, semantic, and performance validation for a proposed pass.
- Implement and test an out-of-tree `H X H -> Z` pass plugin.
- Avoid activating for existing-pass invocation, runtime optimization, generic
  MLIR questions, or other requests outside CUDA-Q pass development.

## Behaviors

- The agent activates `develop-cudaq-pass` for CUDA-Q pass development and not
  for unrelated or operational compiler requests.
- The agent inspects the current CUDA-Q pass boundary and selects the smallest
  owning change shape before implementation.
- The agent states the supported input and output IR and separates prerequisite
  review units from pass-local work.
- The agent validates quantum semantics separately from textual IR validity and
  optimization metrics.
- The agent activates for invalid optimization requests, rejects the
  non-semantics-preserving premise, and redirects to a valid transformation.
- For `pass-source-edit-external-hxh`, the agent creates an external plugin,
  registers `cudaq-hxh-to-z`, defines a conservative supported boundary, builds
  the external target, and authors focused correctness tests.
- Source-editing qualification must check plugin loading, the three-to-one
  rewrite, exact CircuitCheck equivalence, conservative unsupported behavior,
  and the absence of CUDA-Q production-source or pipeline changes.

## Notes

- Most cases are read-only; `pass-source-edit-external-hxh` edits and tests an
  out-of-tree plugin.
- `expected_script` is `null` because cases are graded from agent behavior.
- Source-editing staging must provide the required CUDA-Q source and tools while
  keeping `evals.json`, evaluator-owned tests and expected results, secrets, and
  result directories outside the agent-visible workspace.
- Expected answers describe behavior rather than exact prose so Claude Code and
  Codex can produce natural implementation plans and reports.
