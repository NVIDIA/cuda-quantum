# MKL-Q Known Limitations

This page describes the current public support boundary for MKL-Q. It is meant
to help users understand what the source tree can reasonably prove today and
what remains experimental.

Use [`public-release-checklist.md`](public-release-checklist.md) before treating
the current branch as public-release ready. Use
[`release-policy.md`](release-policy.md) before creating any tag, GitHub
Release, wheel, PyPI package, installer, or signed artifact.

## Release Shape

- MKL-Q is currently source-only. It does not publish wheels, PyPI packages,
  binary installers, GitHub Releases, or signed artifacts.
- The public API intentionally keeps the CUDA-Q namespace and compiler
  interface: Python users still import `cudaq`, and C++ users still compile
  with `nvq++`.
- The repository remains an upstream-compatible fork of NVIDIA CUDA-Q. Upstream
  CUDA-Q targets are preserved so future merges remain practical.

## Platform Boundary

- The active development and validation platform is Apple Silicon macOS ARM64.
- The local validation records in this repository were collected on an Apple
  M5 machine with 10 logical cores, 16 GB RAM, and macOS 26.5.1.
- Linux, Intel macOS, CI-hosted macOS, and non-Apple GPU behavior are not
  certified by the current MKL-Q evidence.

## Target Status

| Target | Status | Current boundary |
| --- | --- | --- |
| `mklq-cpu` | Stable local target | Native fp64 state-vector simulator with focused fast paths for common single-qubit, controlled single-qubit, selected two-qubit, measurement, and sampling paths. |
| `mklq-metal` | Experimental target | Mixed Metal/CPU target with resident fp32 Metal paths for supported operations and CPU oracle fallback when needed. It is not the default backend. |
| Upstream CUDA-Q targets | Preserved | Targets such as `qpp-cpu` remain available for compatibility and oracle comparisons. |

## CPU Backend Boundary

`mklq-cpu` is the correctness and performance baseline for MKL-Q. Current tests
cover API smoke behavior, circuit fixtures, builder-level target behavior,
state-vector parity against `qpp-cpu`, `nvq++` smoke compilation, and
TargetConfig registration.

Do not treat `mklq-cpu` as a complete replacement for every CUDA-Q backend. It
is a local simulator target optimized for Apple Silicon development, not a QPU
or distributed simulator.

## Metal Backend Boundary

`mklq-metal` is experimental. The current implementation may keep supported
single-target, two-target, and three-target gate updates in a resident fp32
Metal state buffer. It also has resident probability-fill,
marginal-probability, measurement, and reset paths for supported cases.

Unsupported or not-yet-profitable paths can synchronize back to the MKL-Q CPU
oracle. That fallback is intentional at this stage. A passing `mklq-metal` test
means the mixed-path target preserved CUDA-Q behavior for that fixture; it does
not prove full GPU residency or end-to-end Metal-native execution.

`mklq-metal` should not be made the default target until its supported path
coverage, correctness gates, benchmark gates, and fallback telemetry justify
that change.

## Validation Boundary

The local correctness gate is:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

It aggregates Python target fixtures, `nvq++` smoke tests, and the build-tree
TargetConfig `ctest` selection. This is the strongest local gate currently
maintained by MKL-Q.

GitHub Actions intentionally runs only the lightweight public hygiene workflow.
That workflow checks public metadata, tracked artifact hygiene, benchmark
summary parseability, and helper-script compilation. It does not build CUDA-Q,
run the Apple Silicon simulator tests, or certify Metal runtime behavior.

## Benchmark Boundary

Tracked summaries under `benchmarks/mklq/reports/` are sanitized local evidence.
Raw benchmark payloads under `benchmarks/mklq/results/` are intentionally
ignored.

Benchmark summaries must be read through their `evidence_kind` and
`interpretation` fields. Some older summaries are dirty-worktree tuning
evidence, not clean release evidence. None of the local benchmark summaries is
cross-machine performance certification.

## Not Yet Supported Or Not Yet Claimed

- No PyPI package, wheel, Homebrew formula, installer, or release tarball.
- No claim of full CUDA-Q ecosystem replacement.
- No claim that `mklq-metal` is fully Metal-native or always faster than
  `mklq-cpu`.
- No hosted CI coverage for Apple Silicon simulator correctness gates.
- No public binary compatibility guarantee beyond source-level CUDA-Q API
  compatibility in the tested local build.
- No hardware QPU validation.

## Reporting Issues

When reporting MKL-Q issues, include:

- macOS version, Apple Silicon model, core count, and memory.
- The exact target: `mklq-cpu`, `mklq-metal`, or an upstream CUDA-Q target.
- The command used to build or run the test.
- Whether `python3 benchmarks/mklq/run_correctness_gate.py` passes locally.
- For performance reports, attach or summarize sanitized benchmark JSON rather
  than raw local payloads.
