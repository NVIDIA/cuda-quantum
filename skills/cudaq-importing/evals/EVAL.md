# Eval guidance for cudaq-importing

Developer guidance for generating and refining `evals.json`. This outranks
generated defaults during NV-BASE/NV-ACES generation and refinement.

## Questions

- Port a small Qiskit circuit (x/h/cx + measure) to CUDA-Q and keep the count
  keys comparable across frameworks.
- Migrate a Qiskit Grover implementation that uses `.inverse()` and
  multi-controlled gates to CUDA-Q without a Qiskit runtime dependency.
- Port a parameterized Qiskit circuit (ry/rz with runtime angles) to a CUDA-Q
  kernel that takes the angles as runtime arguments.
- (negative) Pure CUDA-Q questions with no framework porting (e.g. multi-node
  MPI statevector distribution), which should route to `cudaq-guide` or general
  CUDA-Q knowledge rather than this skill.

## Behaviors

- The agent read skills/cudaq-importing/SKILL.md (and, when needed,
  references/porting-reference.md) before producing the port.
- The agent produced a `@cudaq.kernel` decorator-mode port using
  `cudaq.qvector` and the documented gate translations, preferring runtime
  arguments over per-size generated kernels.
- The agent handled the Qiskit-vs-CUDA-Q bit-ordering / count-key convention at
  the port boundary and matched fp32-vs-fp64 precision when comparing results.
- The agent removed any runtime dependency on the source framework (no `qiskit`
  import in the final port).

## Notes

- cudaq-importing is a documentation/porting-guidance skill with **no executable
  script**, so `expected_script` is `null` for every case and the agent should
  never run a script.
- Ground truth is intentionally derived from SKILL.md and the bundled
  references/porting-reference.md, so cases remain answerable in an isolated
  workspace without staging the repo's docs/sphinx sources.
- Keep the CI-gated dataset small (P0 smoke) for the 1-hour NV-CARPS limit.
- Negative cases set `expected_skill: null` and `should_trigger: false` and are
  used to guard against over-routing on pure CUDA-Q (non-porting) requests.
