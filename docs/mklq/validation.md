# MKL-Q Validation

This page records the current local validation gate for the MKL-Q public
bootstrap. It is not a release certification and does not replace clean CI.
See [`known-limitations.md`](known-limitations.md) for the current support
boundary and evidence limits.

## Machine

- Host: Apple Silicon local development machine
- CPU: Apple M5, 10 logical cores
- Memory: 16 GB RAM
- OS: macOS 26.5.1
- Install prefix used for the public bootstrap gate: `/Users/a0000/.cudaq-mklq`

## Current Evidence Snapshot

Latest local validation refresh: 2026-06-22.

The install-prefix build, full public healthcheck, one-command correctness
gate, and public example smoke gate were last run against source commit
`997ec1f3c022d854d644257bc7dca990a17bd243` before this readiness-audit helper
update was committed. The clean CPU benchmark summary was refreshed separately
against
`34f4b260d1c657ad626c526eed4e6b9d3a441be4` after adding QFT-like and seeded
Clifford composite rows to the clean evidence gate.

A focused Metal counter-evidence refresh was run in the current worktree on
2026-06-22 after strengthening the Metal runtime counter probe schema and
adding resident built-in Rx/Ry/Rz, controlled-Rx/Ry/Rz, and phase-family
S/T/Sdg/Tdg fixtures, plus multi-control single-qubit resident, unsupported
gate fallback/reupload, and Python custom three-target fallback correctness
fixtures. It did not rerun the full install/build/example gate, but it did rerun
focused Metal/Python fixtures and the tracked bounded counter report.

Raw wrapper output was written to ignored local paths
`benchmarks/mklq/results/public-healthcheck-full-2026-06-22.json`,
`benchmarks/mklq/results/local-correctness-gate-2026-06-22.json`,
`benchmarks/mklq/results/local-metal-runtime-counter-probe-2026-06-22.counter.json`,
and the temporary example-smoke payload embedded in the full healthcheck
output; these raw payloads are not tracked as public evidence.

- Install-prefix build: passed.
- Full public healthcheck: passed, with 15 steps passed and 0 failed.
- One-command correctness gate: passed with 4 steps passed, 0 failed, and 0
  skipped, including the Metal runtime counter probe.
- Public example smoke gate: passed, with 30 steps passed and 0 failed.
- Current `benchmark_harness_tests`: `71 passed`.
- Standalone install-prefix Python subset: `35 passed`.
- `python_target_smoke`: `57 passed`.
- `nvqpp_smoke`: `2 passed`.
- Current `target_config_ctest`: `69/69 passed`.
- Current `metal_runtime_counter_probe`: 18 expected, 18 selected, 0 missing,
  and 18 passed, with each counter ctest run independently.
- Clean CPU benchmark gate: passed, with 18 q20 `qpp-cpu`/`mklq-cpu` rows and
  18 rows reporting `status == "ok"`.

## Install-prefix Gate

```bash
cmake --build build-python --target install -j 6
```

Result: passed in the latest local refresh, installing to
`/Users/a0000/.cudaq-mklq`.

```bash
PYTHONPATH=/Users/a0000/.cudaq-mklq \
python3 -m pytest \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q
```

Result: `35 passed in 3.41s` in the latest local refresh.

```bash
CUDAQ_NVQPP=/Users/a0000/.cudaq-mklq/bin/nvq++ \
PYTHONPATH=/Users/a0000/.cudaq-mklq \
python3 -m pytest python/tests/backends/test_mklq_nvqpp_smoke.py -q
```

Result: `2 passed`.

## Build-tree Gate

```bash
ctest --test-dir build-python \
  -R "(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)" \
  --output-on-failure
```

Result in the latest correctness refresh: `69/69 passed`.

```bash
PYTHONPATH=/Users/a0000/Documents/MKL-Q/build-python/python \
python3 -m pytest \
  python/tests/backends/test_mklq_nvqpp_smoke.py \
  python/tests/backends/test_mklq_benchmark_harness.py \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q
```

Historical bootstrap result: `63 passed`. This build-tree Python bundle is not
part of the latest full public healthcheck; the install-prefix correctness
wrapper is the current public readiness gate.

```bash
PYTHONPATH=/Users/a0000/Documents/MKL-Q/tpls/llvm/llvm/utils/lit \
/opt/anaconda3/bin/python3 /Users/a0000/.local/llvm/bin/llvm-lit \
  -j 1 -sv \
  --filter 'mklq_(targets|runtime_smoke)' \
  --param cudaq_site_config=/Users/a0000/Documents/MKL-Q/build-python/targettests/lit.site.cfg.py \
  /Users/a0000/Documents/MKL-Q/build-python/targettests/TargetConfig
```

Historical bootstrap result: 2 selected MKL-Q TargetConfig tests passed. The
latest correctness refresh uses the broader TargetConfig `ctest` selection
above.

## One-command Correctness Gate

Use the local correctness gate wrapper to run the install-prefix Python smoke
tests, the `nvq++` smoke tests, the build-tree TargetConfig `ctest` gate, and
the Metal runtime counter probe in one command:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

Latest focused local result: passed on 2026-06-22 after the resident
Rx/Ry/Rz, phase-family, multi-control, unsupported fallback/reupload, and
Python custom three-target fallback correctness refresh. It reported 4 wrapper
steps passed, 0 failed, and 0 skipped. The step-level results were:

- `python_target_smoke`: `57 passed`.
- `nvqpp_smoke`: `2 passed`.
- `target_config_ctest`: `69/69 passed`.
- `metal_runtime_counter_probe`: 18 expected, 18 selected, 0 missing, and 18
  independently executed passing counter ctests, including the resident
  built-in Rx/Ry/Rz, controlled-Rx/Ry/Rz, and phase-family S/T/Sdg/Tdg
  fixtures, plus the multi-control single-qubit resident fixture and the
  unsupported gate fallback/reupload boundary fixture.

The Python smoke step includes the MKL-Q API smoke tests, the CPU correctness
fixture suite, the limited experimental Metal correctness fixture suite, and
the builder-level MKL-Q target tests.

The default JSON output path is ignored by Git:
`benchmarks/mklq/results/local-correctness-gate-<date>.json`. The default
Metal runtime counter probe output path is also ignored by Git:
`benchmarks/mklq/results/local-metal-runtime-counter-probe-<date>.counter.json`.
Use `--skip-metal-counter-probe` only when that build-tree counter evidence is
intentionally out of scope. Use `--plan-only` to inspect the exact commands and
environment without running the gate:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python \
  --plan-only
```

For build-tree-only experiments, override the runtime paths explicitly:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --pythonpath /Users/a0000/Documents/MKL-Q/build-python/python \
  --nvqpp /Users/a0000/Documents/MKL-Q/build-python/bin/nvq++ \
  --build-dir build-python
```

## Repository Hygiene Gate

```bash
python3 benchmarks/mklq/run_public_healthcheck.py --full --require-clean
```

Latest 2026-06-22 result: `15/15` steps passed. This includes Git
repository hygiene, tracked-artifact checks, public metadata checks, sanitized
benchmark summary parsing, the clean CPU performance evidence guard, the Metal
evidence boundary guard, bounded Metal runtime counter evidence parsing, helper
`py_compile`, markdown links, benchmark evidence regeneration, benchmark
harness tests, install-prefix build, the one-command correctness gate, and the
public example smoke gate.

## Benchmark Evidence

Sanitized local benchmark summaries are tracked under
`benchmarks/mklq/reports/`. Raw local benchmark JSON under
`benchmarks/mklq/results/` is intentionally ignored.

The compact public index for the tracked summaries is
[`benchmark-evidence.md`](benchmark-evidence.md). Rerun the clean CPU benchmark
gate, regenerate the sanitized summary, and refresh the public index with:

```bash
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp 2026-06-21
```

If the ignored raw JSON already exists, regenerate only the sanitized summary
and public index with:

```bash
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp 2026-06-21 \
  --skip-benchmark
```

Current tracked summaries include:

- `local-clean-cpu-q20-2026-06-21.summary.json`
- `local-current-sampling-fullprob-gated-q20-2026-06-19.summary.json`
- `local-y-cy-fastpath-isolated-q20-2026-06-19.summary.json`
- `local-metal-composite-mixed-path-q20-2026-06-21.summary.json`
- `local-metal-y-cy-resident-isolated-q20-2026-06-19.summary.json`
- `local-counts-only-sampling-shot-scaling-q20-2026-06-19.summary.json`

The clean-worktree local benchmark summary was refreshed against
`34f4b260d1c657ad626c526eed4e6b9d3a441be4` on 2026-06-21. The clean summary now
includes `y-state`, `cy-state`, `cz-state`, `qft-like-state`,
`seeded-clifford-state`, full-register sampling, and partial-register sampling
rows. These files include one clean-worktree local benchmark summary plus older
dirty-worktree tuning summaries. Interpret each file through its
`evidence_kind` and `interpretation` fields. Do not treat any local summary as
cross-machine performance certification.

The Metal composite summary is local tuning evidence only. It records q20
`qft-like-state` and `seeded-clifford-state` rows for `qpp-cpu`, `mklq-cpu`, and
experimental `mklq-metal`; all six rows completed with `status == "ok"`. The
summary keeps the Metal scope as mixed-path state-vector updates followed by
host readback, not full Metal-native execution.
