# MKL-Q Architecture

MKL-Q is an upstream-compatible CUDA-Q fork. It does not rename the CUDA-Q API
surface. Instead, it adds Apple Silicon simulator targets under the existing
CUDA-Q runtime and target configuration contracts.

## Layer Map

| Layer | Files | Responsibility |
| --- | --- | --- |
| Public API | `cudaq`, `nvq++` | Keep CUDA-Q-compatible user entry points. |
| Target configs | `runtime/nvqir/mklq/mklq-cpu.yml`, `runtime/nvqir/mklq/mklq-metal.yml` | Register MKL-Q simulator names and target preprocessor markers. |
| Runtime libraries | `runtime/nvqir/mklq/CMakeLists.txt` | Build and install `nvqir-mklq_cpu` and `nvqir-mklq_metal`. |
| CPU backend | `runtime/nvqir/mklq/MklqCpuCircuitSimulator.cpp` | Stable fp64 state-vector simulator and correctness oracle. |
| Metal runtime | `runtime/nvqir/mklq/MklqMetalRuntime.*` | Experimental resident Metal operations and host synchronization boundary. |
| Target tests | `targettests/TargetConfig/mklq_targets.config`, `targettests/TargetConfig/mklq_runtime_smoke.cpp` | Prove target config installation and C++ `nvq++ --target` smoke behavior. |
| Python tests | `python/tests/backends/test_mklq_*.py`, `python/tests/builder/test_mklq_targets.py` | Prove Python target selection, API smoke behavior, fixtures, and builder integration. |
| Benchmark evidence | `benchmarks/mklq/`, `docs/mklq/benchmark-evidence.md` | Collect local measurements and publish sanitized summaries only. |

## Public Compatibility Boundary

MKL-Q keeps these CUDA-Q-facing contracts stable:

- Python users still import `cudaq`.
- C++ users still compile through `nvq++`.
- Upstream targets such as `qpp-cpu` remain available.
- MKL-Q targets are additive: `mklq-cpu` and `mklq-metal`.
- The repository remains a CUDA-Q fork so upstream merges stay practical.

Do not introduce a hard project-wide rename as part of backend work. If an API
change is useful to upstream CUDA-Q, keep it separable from MKL-Q-specific
Apple Silicon target work.

## Target Registration

`runtime/nvqir/mklq/CMakeLists.txt` creates two simulator libraries:

- `nvqir-mklq_cpu`, registered as backend `mklq_cpu`.
- `nvqir-mklq_metal`, registered as backend `mklq_metal`.

The installed target YAML files expose the public target names:

- `mklq-cpu`
- `mklq-metal`

The target configs also set marker macros used by smoke tests:

- `MKLQ_APPLE_SILICON_CPU_BASELINE`
- `MKLQ_METAL_EXPERIMENTAL_CPU_ORACLE`

Those markers are target identity checks, not user API.

## CPU Backend

`mklq-cpu` is the stable MKL-Q target. Its current role is:

- provide the fp64 state-vector correctness baseline;
- preserve CUDA-Q simulator behavior for supported local workflows;
- host focused Apple Silicon CPU optimizations;
- serve as the oracle for experimental Metal fallback and comparison.

CPU optimization should stay behind this target and should land with circuit
fixtures that compare against `qpp-cpu` or another accepted CUDA-Q-compatible
reference.

## Metal Backend

`mklq-metal` is experimental. It is a mixed-path backend, not a full
Metal-native replacement for `mklq-cpu`.

Current architecture:

- supported single-target, two-target, and three-target updates may stay in a
  resident fp32 Metal state buffer;
- supported dense probability fills, cost-gated marginal fills, measurement
  probability reductions, and measurement collapse paths may run through Metal;
- unsupported or not-yet-profitable paths synchronize back to the MKL-Q fp64 CPU
  oracle;
- host state remains the compatibility boundary for fallback and readback.

A passing Metal fixture proves CUDA-Q-compatible behavior for that fixture. It
does not prove full GPU residency, full Metal-native execution, or default-ready
status.

## Correctness Strategy

Use layered evidence:

1. Target config checks prove `mklq-cpu` and `mklq-metal` are installed and
   select the intended simulator backends.
2. Python API tests prove `cudaq.set_target(...)`, sampling, state access,
   observe paths, and builder integration.
3. CPU fixtures compare `mklq-cpu` behavior against CUDA-Q-compatible reference
   behavior.
4. Metal fixtures compare `mklq-metal` behavior against `qpp-cpu` or the MKL-Q
   CPU oracle with tolerance appropriate for fp32 resident paths.
5. `nvq++` smoke tests prove C++ target selection and runtime execution.

The detailed test-to-capability map is tracked in
[`testing-matrix.md`](testing-matrix.md).

The one-command gate is:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

## Benchmark Strategy

Benchmark data has two forms:

- raw local JSON under `benchmarks/mklq/results/`, which must stay ignored;
- sanitized summaries under `benchmarks/mklq/reports/`, which may be public.

The public index is `docs/mklq/benchmark-evidence.md`. Interpret entries through
their `evidence_kind` and `interpretation` fields. Local measurements are
optimization evidence, not cross-machine performance certification.

## GitHub And Public Hygiene

GitHub Actions intentionally runs only lightweight source hygiene:

- reject generated/local/private tracked artifacts;
- check public metadata and documentation invariants;
- parse sanitized benchmark summaries;
- compile public benchmark helper scripts.

It does not build CUDA-Q, run Apple Silicon backend tests, certify Metal runtime
behavior, or publish release artifacts.

## Adding Backend Work

Use this order for non-trivial backend changes:

1. Add or update the correctness fixture first.
2. Implement the narrow backend behavior.
3. Run the smallest focused test that proves the behavior.
4. Run the one-command correctness gate before treating it as ready.
5. Collect benchmark evidence only after correctness passes.
6. Update public docs when support boundaries or user-visible behavior change.

Keep CPU and Metal changes separable when possible. If a Metal change depends on
CPU oracle behavior, land or verify the CPU behavior first.

## Non-Goals For The Current Public Version

- No PyPI package, wheel, installer, or GitHub Release.
- No hard rename of the `cudaq` namespace.
- No removal of upstream CUDA-Q targets.
- No claim that `mklq-metal` is default-ready or fully Metal-native.
- No claim that local benchmark evidence is release certification.
