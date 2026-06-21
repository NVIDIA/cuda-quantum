# MKL-Q Validation

This page records the current local validation gate for the MKL-Q public
bootstrap. It is not a release certification and does not replace clean CI.

## Machine

- Host: Apple Silicon local development machine
- CPU: Apple M5, 10 logical cores
- Memory: 16 GB RAM
- OS: macOS 26.5.1
- Install prefix used for the public bootstrap gate: `/Users/a0000/.cudaq-mklq`

## Install-prefix Gate

```bash
cmake --build build-python --target install -j 6
```

Result: passed, installing to `/Users/a0000/.cudaq-mklq`.

```bash
PYTHONPATH=/Users/a0000/.cudaq-mklq \
python3 -m pytest \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q
```

Result: `35 passed`.

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

Result: `63/63 passed`.

```bash
PYTHONPATH=/Users/a0000/Documents/MKL-Q/build-python/python \
python3 -m pytest \
  python/tests/backends/test_mklq_nvqpp_smoke.py \
  python/tests/backends/test_mklq_benchmark_harness.py \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q
```

Result: `63 passed`.

```bash
PYTHONPATH=/Users/a0000/Documents/MKL-Q/tpls/llvm/llvm/utils/lit \
/opt/anaconda3/bin/python3 /Users/a0000/.local/llvm/bin/llvm-lit \
  -j 1 -sv \
  --filter 'mklq_(targets|runtime_smoke)' \
  --param cudaq_site_config=/Users/a0000/Documents/MKL-Q/build-python/targettests/lit.site.cfg.py \
  /Users/a0000/Documents/MKL-Q/build-python/targettests/TargetConfig
```

Result: 2 selected MKL-Q TargetConfig tests passed.

## Repository Hygiene Gate

```bash
git diff --check
```

Result: no whitespace errors at the time of the bootstrap gate.

## Benchmark Evidence

Sanitized local benchmark summaries are tracked under
`benchmarks/mklq/reports/`. Raw local benchmark JSON under
`benchmarks/mklq/results/` is intentionally ignored.

The compact public index for the tracked summaries is
[`benchmark-evidence.md`](benchmark-evidence.md). Regenerate it with:

```bash
python3 benchmarks/mklq/summarize_reports.py \
  --reports benchmarks/mklq/reports \
  --format markdown \
  --output docs/mklq/benchmark-evidence.md
```

Current tracked summaries include:

- `local-clean-cpu-q20-2026-06-21.summary.json`
- `local-current-sampling-fullprob-gated-q20-2026-06-19.summary.json`
- `local-y-cy-fastpath-isolated-q20-2026-06-19.summary.json`
- `local-metal-y-cy-resident-isolated-q20-2026-06-19.summary.json`
- `local-counts-only-sampling-shot-scaling-q20-2026-06-19.summary.json`

These files now include one clean-worktree local benchmark summary plus older
dirty-worktree tuning summaries. Interpret each file through its
`evidence_kind` and `interpretation` fields. Do not treat any local summary as
cross-machine performance certification.
