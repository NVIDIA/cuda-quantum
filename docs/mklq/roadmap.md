# MKL-Q Roadmap

MKL-Q starts as a CUDA-Q-compatible Apple Silicon fork. The first public version
keeps the CUDA-Q API surface and adds MKL-Q targets without renaming the whole
project or removing upstream backends.

The current architecture boundary is documented in
[`architecture.md`](architecture.md). Use that page as the target/backend
contract when planning backend work.

## Current Backend Status

- `mklq-cpu` is the stable MKL-Q target. It uses a native fp64 state-vector
  simulator with focused fast paths for common single-qubit gates, controlled
  single-qubit gates, selected two-qubit gates, measurement, and sampling.
- `mklq-metal` is experimental. It uses resident Metal state for supported gate,
  probability, sampling, measure, and reset paths, and falls back to the MKL-Q
  CPU oracle for unsupported paths.
- Standard non-explicit `cudaq.sample` can use MKL-Q counts-only aggregation.
  Explicit-measurement sampling keeps sequential-shot behavior for compatibility.

The public support boundary, non-goals, and evidence limits are tracked in
[`known-limitations.md`](known-limitations.md).

## Near-term Work

- Keep `mklq-cpu` as the correctness and performance baseline.
- Expand CPU optimization only when circuit-level parity tests pass.
- Keep `mklq-metal` experimental until core supported paths no longer depend on
  CPU fallback and benchmark/correctness gates justify broader claims.
- Add clean CI evidence after the source-only repository is public.
- Decide later whether to publish wheels or GitHub Releases; neither is part of
  the first public source-only version.
- Before any public release-style milestone, rerun
  [`public-release-checklist.md`](public-release-checklist.md).

## Compatibility Rules

- Preserve the `cudaq` Python namespace and `nvq++` compiler interface.
- Preserve upstream CUDA-Q backends, including `qpp-cpu`, so upstream sync
  remains practical.
- Keep performance claims tied to measured benchmark artifacts and machine
  metadata.
- Do not make `mklq-metal` the default target until it passes the same
  correctness and performance gates as the CPU target.
