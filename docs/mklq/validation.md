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
[`benchmark-evidence.md`](benchmark-evidence.md). Regenerate the clean CPU
summary from ignored local raw JSON with:

```bash
python3 benchmarks/mklq/make_summary.py \
  --raw benchmarks/mklq/results/local-clean-cpu-gate-y-cy-q20-2026-06-21.json \
  --raw benchmarks/mklq/results/local-clean-cpu-sampling-q20-2026-06-21.json \
  --summary-id local-clean-cpu-q20-2026-06-21 \
  --evidence-kind clean_local_benchmark_evidence \
  --ratio-group clean_worktree_cross_target_ratio \
  --performance-scope 'local Apple M5 q20 CPU target comparison only; not a cross-machine release benchmark' \
  --summary-text 'Clean-worktree local run comparing qpp-cpu and mklq-cpu for q20 Y/CY state updates plus full/partial-register sampling at 1024 and 65536 shots.' \
  --runtime-note 'The CUDA-Q Python runtime came from /Users/a0000/.cudaq-mklq and reports f98433b6; source HEAD 4b112725 adds docs/benchmark evidence tooling on top of the same backend code.' \
  --output benchmarks/mklq/reports/local-clean-cpu-q20-2026-06-21.summary.json
```

Then regenerate the public index with:

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
