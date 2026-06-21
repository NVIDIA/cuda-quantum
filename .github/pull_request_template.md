## Summary

<!-- Briefly describe the change and why it is needed. -->

## Change Type

- [ ] `mklq-cpu` backend behavior
- [ ] `mklq-metal` backend behavior
- [ ] Python API or target selection
- [ ] `nvq++` or target configuration
- [ ] Benchmark harness or sanitized benchmark evidence
- [ ] Documentation
- [ ] Repository configuration or public hygiene

## Compatibility Boundary

- [ ] Keeps the public Python namespace as `cudaq`.
- [ ] Keeps the compiler entry point as `nvq++`.
- [ ] Does not make `mklq-metal` the default target.
- [ ] Does not describe local benchmark evidence as release certification.
- [ ] Keeps upstream CUDA-Q compatibility unless the change is explicitly scoped
      to MKL-Q target behavior.
- [ ] If this syncs upstream CUDA-Q, `docs/mklq/upstream-sync.md` was followed.
- [ ] If this changes release behavior, `docs/mklq/release-policy.md` was
      followed.

## Validation

- [ ] `git diff --check`
- [ ] Public hygiene checks equivalent to `.github/workflows/mklq-public-hygiene.yml`
- [ ] `python3 benchmarks/mklq/run_correctness_gate.py --install-prefix "${HOME}/.cudaq-mklq" --build-dir build-python`
- [ ] Focused Python tests:
      `python/tests/backends/test_mklq_python_api.py`,
      `python/tests/builder/test_mklq_targets.py`,
      or `python/tests/backends/test_mklq_nvqpp_smoke.py`
- [ ] Focused `ctest` selection:
      `(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)`
- [ ] Testing matrix impact was considered:
      `docs/mklq/testing-matrix.md`
- [ ] Not run; reason:

## Benchmark Evidence

- [ ] No benchmark evidence changed.
- [ ] Raw local benchmark JSON remains ignored under `benchmarks/mklq/results/`.
- [ ] Sanitized summaries were updated under `benchmarks/mklq/reports/`.
- [ ] `docs/mklq/benchmark-evidence.md` was updated.
- [ ] Benchmark interpretation states machine scope and limitations.

## Public Hygiene

- [ ] No build output, caches, `.DS_Store`, local signing objects, raw benchmark
      JSON, secrets, tokens, or private machine paths are tracked.
- [ ] `.github/workflows/` still contains only intentionally reviewed workflows.
- [ ] Public docs still describe `mklq-cpu` as stable and `mklq-metal` as
      experimental.
- [ ] Public release checklist impact was considered:
      `docs/mklq/public-release-checklist.md`.

## Reviewer Notes

- Affected targets:
- Known limitations:
- Follow-up work:
