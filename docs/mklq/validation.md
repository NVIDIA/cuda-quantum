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

Latest local validation refresh: 2026-06-21.

The install-prefix build, full public healthcheck, one-command correctness
gate, and public example smoke gate were last run against source commit
`90b1ebe20b281411880e7704df5b4120692e4686` before this validation note was
committed. The static public-healthcheck composition was then extended with the
experimental Metal evidence boundary guard. The clean CPU benchmark summary was
refreshed separately against
`34f4b260d1c657ad626c526eed4e6b9d3a441be4` after adding QFT-like and seeded
Clifford composite rows to the clean evidence gate.

Raw wrapper output was written to ignored local paths
`benchmarks/mklq/results/public-healthcheck-full-2026-06-21.json` and
`benchmarks/mklq/results/local-correctness-gate-2026-06-21.json`, and
`benchmarks/mklq/results/example-smoke-2026-06-21.json`; these raw payloads are
not tracked as public evidence.

- Install-prefix build: passed.
- Full public healthcheck: passed, with 14 steps passed and 0 failed.
- One-command correctness gate: passed, with 3 steps passed, 0 failed, and 0
  skipped.
- Public example smoke gate: passed, with 30 steps passed and 0 failed.
- `benchmark_harness_tests`: `58 passed`.
- Standalone install-prefix Python subset: `35 passed`.
- `python_target_smoke`: `56 passed`.
- `nvqpp_smoke`: `2 passed`.
- `target_config_ctest`: `63/63 passed`.
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

Result in the latest correctness refresh: `63/63 passed`.

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
tests, the `nvq++` smoke tests, and the build-tree TargetConfig `ctest` gate in
one command:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

Latest local result: passed on 2026-06-21 against
`90b1ebe20b281411880e7704df5b4120692e4686` with 3 wrapper steps passed, 0
failed, and 0 skipped. The step-level results were:

- `python_target_smoke`: `56 passed`.
- `nvqpp_smoke`: `2 passed`.
- `target_config_ctest`: `63/63 passed`.

The Python smoke step includes the MKL-Q API smoke tests, the CPU correctness
fixture suite, the limited experimental Metal correctness fixture suite, and
the builder-level MKL-Q target tests.

The default JSON output path is ignored by Git:
`benchmarks/mklq/results/local-correctness-gate-<date>.json`. Use
`--plan-only` to inspect the exact commands and environment without running the
gate:

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

Result: `14/14` steps passed in the latest local refresh. This includes Git
repository hygiene, tracked-artifact checks, public metadata checks, sanitized
benchmark summary parsing, the clean CPU performance evidence guard, the Metal
evidence boundary guard, helper `py_compile`, markdown links, benchmark
evidence regeneration, benchmark harness tests, install-prefix build, the
one-command correctness gate, and the public example smoke gate.

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
