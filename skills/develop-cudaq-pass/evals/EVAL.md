# Eval guidance for develop-cudaq-pass

This directory evaluates whether agents activate `develop-cudaq-pass` for
CUDA-Q compiler pass work, avoid it for near misses, and follow the documented
workflow when planning or implementing a pass.

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
- The agent examines the relevant CUDA-Q IR and pipeline before choosing the
  appropriate compiler extension point.
- The agent states the supported input and output IR and separates prerequisite
  review units from pass-local work.
- The agent validates quantum semantics separately from textual IR validity and
  optimization metrics.
- The agent activates for invalid optimization requests, rejects the
  non-semantics-preserving premise, and redirects to a valid transformation.
- For `pass-source-edit-external-hxh`, the agent creates an external plugin,
  registers `cudaq-hxh-to-z`, defines a conservative supported boundary, and
  authors the build configuration and focused correctness tests.
- The source-editing case reports plugin loading, the three-to-one rewrite,
  exact CircuitCheck equivalence, conservative unsupported behavior, and the
  absence of CUDA-Q production-source or pipeline changes.

## Grading

SkillEvaluator stages every `evals.json` case with and without the skill and
uses its standard trajectory and answer grader for both conditions. The grader
assesses security, skill execution, tool-call efficiency, answer accuracy, goal
completion, and expected behavior. It reports skill lift by comparing the two
conditions.

The evaluation environment provides a CUDA-Q source checkout and warm build.
For the H-X-H case, the agent builds the plugin and runs focused checks in that
environment. SkillEvaluator grades the resulting tool trajectory and final
solution. It does not independently rebuild the submitted plugin after the
agent finishes.

## Notes

- Keep evaluator-owned inputs, expected results, secrets, and result directories
  outside the agent-visible workspace.
