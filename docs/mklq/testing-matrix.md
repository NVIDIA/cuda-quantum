# MKL-Q Testing Matrix

This matrix maps the current MKL-Q tests, gates, and benchmark helpers to the
behavior they prove. It is a maintenance guide, not a certification statement.

Use it before changing backend behavior, target configuration, sampling,
measurement/reset, benchmark scripts, or public release metadata.

## Gate Summary

| Gate | Command | Proves | Does Not Prove |
| --- | --- | --- | --- |
| Public hygiene | `.github/workflows/mklq-public-hygiene.yml` or the equivalent local commands in `developer-workflow.md` | Public metadata, ignored-artifact policy, sanitized benchmark JSON parsing, benchmark helper syntax | CUDA-Q build, Apple Silicon runtime behavior, Metal device behavior, release readiness |
| One-command correctness | `python3 benchmarks/mklq/run_correctness_gate.py --install-prefix "${HOME}/.cudaq-mklq" --build-dir build-python` | Python target smoke, CPU fixtures, Metal fixtures, `nvq++` smoke, selected build-tree `ctest` tests | Full upstream CUDA-Q test suite, cross-machine performance, packaging |
| Build-tree Python smoke | `PYTHONPATH="$(pwd)/build-python/python" python3 -m pytest ... -q` | Source-tree Python API behavior against the build products | Installed prefix behavior unless paths are changed |
| Install-prefix Python smoke | `PYTHONPATH="${HOME}/.cudaq-mklq" python3 -m pytest ... -q` | Installed Python package target behavior | C++ `nvq++` behavior unless the nvq++ smoke also runs |
| TargetConfig `ctest` | `ctest --test-dir build-python -R "(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)" --output-on-failure` | MKL-Q backend gtests, target config tests, target setter checks selected by the regex | Every CUDA-Q target test |
| Benchmark dry run | `python3 benchmarks/mklq/bench_mklq_targets.py --dry-run ...` | Benchmark planning, schema, case expansion, row metadata | Runtime performance |
| Clean CPU benchmark | `python3 benchmarks/mklq/run_clean_cpu_benchmark.py --pythonpath "${HOME}/.cudaq-mklq" --stamp YYYY-MM-DD` | Clean-worktree local qpp-cpu vs mklq-cpu benchmark evidence and sanitized report regeneration | Release certification or results on another machine |

## Test Artifacts

| Artifact | Scope | Evidence |
| --- | --- | --- |
| `python/tests/backends/test_mklq_python_api.py` | Python API smoke for `mklq-cpu` and `mklq-metal` | `cudaq.set_target`, `cudaq.sample`, `cudaq.get_state`, `cudaq.observe`, observe lists, shots observe, decorator-kernel `cudaq.run`, mid-circuit feedback/reset |
| `python/tests/backends/test_mklq_cpu_correctness_fixtures.py` | CPU circuit correctness fixtures | Bell, GHZ, sampling support, QFT-like fixtures, deterministic Clifford behavior, parameterized state and observable comparisons against `qpp-cpu` |
| `python/tests/backends/test_mklq_metal_correctness_fixtures.py` | Experimental Metal mixed-path fixtures | Resident single-target, controlled single-target, two-target, deterministic sampling, resident measurement feedback, resident reset after measurement, qpp-cpu tolerance comparisons |
| `python/tests/backends/test_mklq_nvqpp_smoke.py` | Installed or build-tree `nvq++` smoke | `nvq++ --target mklq-cpu`, `nvq++ --target mklq-metal`, target markers, runtime smoke output |
| `python/tests/builder/test_mklq_targets.py` | Builder-level and target behavior | Target presence, precision tolerance, bit order, sampling boundary cases, OpenMP-sized state comparisons, CPU hot paths, state import/export error handling |
| `python/tests/backends/test_mklq_benchmark_harness.py` | Benchmark and report tooling | Dry-run schemas, case expansion, row isolation, summary generation, sanitized evidence fields, probability microbenchmark schemas |
| `python/tests/mklq_test_utils.py` | Test availability helper | Skips MKL-Q tests when targets are unavailable |
| `targettests/TargetConfig/mklq_targets.config` | Target config file checks | Source and installed YAML existence, backend names, marker defines |
| `targettests/TargetConfig/mklq_runtime_smoke.cpp` | C++ target runtime smoke | Target marker exclusivity, sample/observe/mid-circuit behavior through `nvq++` |
| `unittests/nvqpp/backends/MKLQCpuTester.cpp` | CPU backend implementation tests | Backend edge cases, sampling internals, probability fills, built-in fast paths, custom operation fallback behavior |
| `unittests/nvqpp/backends/MKLQMetalTester.cpp` | Metal runtime and mixed-path simulator tests | Metal device detection, resident gate kernels, probability fills, marginal paths, measurement/collapse/reset paths, fallback/error behavior |
| `benchmarks/mklq/run_correctness_gate.py` | Correctness gate wrapper | Aggregates Python smoke, `nvq++` smoke, and selected `ctest` into ignored JSON |
| `benchmarks/mklq/bench_mklq_targets.py` | Target benchmark harness | Local timing rows for gate/state/sampling cases and target notes |
| `benchmarks/mklq/bench_probability_kernels.py` | Probability microbenchmark | Local dense probability kernel experiment data and schema |
| `benchmarks/mklq/make_summary.py` | Summary sanitizer | Converts raw local benchmark JSON to public summary JSON |
| `benchmarks/mklq/summarize_reports.py` | Public evidence index renderer | Builds `docs/mklq/benchmark-evidence.md` from sanitized summaries |
| `benchmarks/mklq/run_clean_cpu_benchmark.py` | Clean benchmark gate | Refuses dirty clean-evidence runs unless explicitly overridden; regenerates sanitized summaries and public index |

## Capability Coverage

| Capability | Primary Evidence | Required When Changing | Current Gap |
| --- | --- | --- | --- |
| Target registration | `mklq_targets.config`, `test_mklq_targets.py`, `test_mklq_nvqpp_smoke.py` | Target YAML, backend name, preprocessor marker, install layout | Does not prove backend math |
| Python target API | `test_mklq_python_api.py` | `cudaq.set_target`, Python runtime integration, observe/sample/run behavior | Limited to selected smoke kernels |
| C++ target API | `mklq_runtime_smoke.cpp`, `test_mklq_nvqpp_smoke.py` | `nvq++ --target`, target marker behavior, C++ runtime smoke | Not a broad C++ API suite |
| CPU state-vector correctness | `test_mklq_cpu_correctness_fixtures.py`, `test_mklq_targets.py`, `MKLQCpuTester.cpp` | CPU gate semantics, sampling, state import/export, error handling, hot paths | Broader noise and full CUDA-Q backend parity are not covered |
| CPU performance-sensitive paths | `MKLQCpuTester.cpp`, `bench_mklq_targets.py`, clean CPU benchmark summaries | Single-qubit, controlled single-qubit, selected two-qubit, probability, sampling fast paths | Benchmark evidence is local machine evidence only |
| Metal resident gate paths | `test_mklq_metal_correctness_fixtures.py`, `MKLQMetalTester.cpp` | Resident single-target/two-target/control gate kernels | Does not prove full GPU residency for every operation |
| Metal probability/sampling paths | `MKLQMetalTester.cpp`, `bench_mklq_targets.py`, sanitized summaries | Full-register probability fill, marginal probability, sampling path labels | Sample draw/count accumulation remains host-side |
| Metal measurement/reset | `test_mklq_metal_correctness_fixtures.py`, `MKLQMetalTester.cpp`, `mklq_runtime_smoke.cpp` | Resident measured-qubit reduction, collapse, reset, mid-circuit behavior | Fixture coverage is finite and tolerance-based |
| Benchmark tooling | `test_mklq_benchmark_harness.py`, benchmark helper `py_compile`, summary JSON parse | Benchmark case/schema/report changes | Does not prove runtime speed unless real benchmark rows run |
| Public release hygiene | `.github/workflows/mklq-public-hygiene.yml`, `public-release-checklist.md` | Public metadata, docs, tracked artifact policy | Does not build or run Apple Silicon backend tests |

## Minimum Test Selection By Change Type

| Change Type | Minimum Local Evidence |
| --- | --- |
| Docs-only public metadata | `git diff --check`, public hygiene metadata checks, banned-token scan |
| Target YAML or target marker | `targettests/TargetConfig/mklq_targets.config`, `test_mklq_targets.py`, `test_mklq_nvqpp_smoke.py` |
| Python target selection | `python/tests/backends/test_mklq_python_api.py`, `python/tests/builder/test_mklq_targets.py` |
| CPU gate semantics | A new or updated CPU fixture plus `test_mklq_cpu_correctness_fixtures.py` and relevant `MKLQCpuTester` case |
| CPU sampling/probability | `MKLQCpuTester.cpp`, `test_mklq_targets.py`, benchmark dry run; clean benchmark evidence only after correctness passes |
| Metal resident gate path | A new or updated `MKLQMetalTester` case plus `test_mklq_metal_correctness_fixtures.py` |
| Metal measurement/reset | `MKLQMetalTester.cpp`, `test_mklq_metal_correctness_fixtures.py`, `mklq_runtime_smoke.cpp` if C++ behavior changes |
| Benchmark harness or summary | `test_mklq_benchmark_harness.py`, helper `py_compile`, summary JSON parse |
| Public release milestone | Full `public-release-checklist.md`, one-command correctness gate, clean benchmark evidence if performance claims changed |

## Known Coverage Gaps

- No GitHub-hosted Apple Silicon backend correctness CI yet.
- No wheel, PyPI, installer, signing, or release artifact gate.
- No claim of complete upstream CUDA-Q backend parity.
- No broad noise-model validation for MKL-Q targets.
- No distributed simulation, multi-GPU, or remote QPU validation.
- No full Metal-native guarantee for `mklq-metal`; unsupported paths may fall
  back to the MKL-Q fp64 CPU oracle.
- No cross-machine performance certification from tracked benchmark summaries.

## Updating This Matrix

Update this file when adding, removing, renaming, or materially changing:

- MKL-Q target configs;
- Python target tests;
- C++ backend gtests;
- TargetConfig lit tests;
- benchmark cases, schemas, or summaries;
- public release or hygiene gates.

If a backend change creates a new supported behavior, add the test first or in
the same commit as the behavior. If a test only proves metadata or planning,
do not describe it as runtime correctness evidence.
