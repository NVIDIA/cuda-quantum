# MKL-Q Benchmark Harness

This directory contains local benchmark tooling for comparing CUDA-Q `qpp-cpu`
against MKL-Q Apple Silicon targets. It records measurements and machine
metadata; it does not encode performance claims.

The script's default target list includes `mklq-cpu` and `mklq-metal` only on
Apple Silicon (`Darwin arm64/aarch64`). On other platforms, pass MKL-Q targets
explicitly only after building them intentionally.

`mklq-metal` is included as an experimental target name while the Metal backend
is being built. It currently loads the MKL-Q `mklq_metal` mixed-path simulator:
supported single-target and two-target updates, including controlled forms, can
stay in a resident fp32 Metal state buffer across supported gate sequences, and
dense full-register probability fills, cost-gated resident marginal probability
fills, and measure/reset collapse paths can read or update that resident buffer
directly.
Measurement probability uses a
dedicated measured-qubit Metal reduction kernel with a small host partial-sum
finish; branch collapse uses a Metal kernel. Unsupported paths fall back to the
MKL-Q fp64 CPU oracle after synchronizing host state, and sample draw/count
accumulation remains host-side.
Treat `mklq-metal` benchmark rows as mixed-path evidence, not full Metal GPU
backend performance.

## Dry Run

```bash
python3 benchmarks/mklq/bench_mklq_targets.py \
  --dry-run \
  --targets qpp-cpu,mklq-cpu,mklq-metal \
  --cases gate-state,sample-basis,sample-ghz,sample-full-register,sample-partial-register,single-qubit-state,h-state,y-state,rx-state,ry-state,rz-state,controlled-state,ch-state,cy-state,crx-state,cry-state,crz-state,cz-state,two-qubit-state \
  --qubits 4,8,12 \
  --shot-counts 256,1024,8192 \
  --output /tmp/mklq-benchmark-plan.json
```

## Smoke Benchmark

Use the built Python tree when running from the repository:

```bash
PYTHONPATH="$(pwd)/build-python/python" \
python3 benchmarks/mklq/bench_mklq_targets.py \
  --targets qpp-cpu,mklq-cpu,mklq-metal \
  --cases gate-state,sample-basis,sample-ghz,sample-full-register,sample-partial-register,single-qubit-state,h-state,y-state,rx-state,ry-state,rz-state,controlled-state,ch-state,cy-state,crx-state,cry-state,crz-state,cz-state,two-qubit-state \
  --qubits 4 \
  --shots 32 \
  --repeats 1 \
  --warmups 1 \
  --layers 2 \
  --output /tmp/mklq-benchmark-smoke.json
```

The `--qubits 4` smoke command is only a quick wiring check. Use larger qubit
counts, such as q15-q20 isolated rows, when collecting dense sampling evidence.
For CPU-backed `sample-full-register`, q4 can still fit the sparse
full-register sampling fast path; use q7 or larger with more than 64 nonzero
outcomes to exercise the dense probability-fill path. For `mklq-metal`, a dirty
resident Metal state skips the sparse host probe and can exercise resident
dense probability-fill even at small smoke sizes, but q7+ remains the clearer
path-level check.

For performance comparisons where row ordering, allocator history, or
`ru_maxrss` inheritance matter, run each row in a fresh Python process:

```bash
OMP_NUM_THREADS=10 \
PYTHONPATH="$(pwd)/build-python/python" \
python3 benchmarks/mklq/bench_mklq_targets.py \
  --isolate-rows \
  --targets mklq-cpu \
  --cases gate-state,sample-basis,sample-ghz,sample-full-register,sample-partial-register,single-qubit-state,h-state,y-state,rx-state,ry-state,rz-state,controlled-state,ch-state,cy-state,crx-state,cry-state,crz-state,cz-state,two-qubit-state \
  --qubits 15,16,17,18,19,20 \
  --shots 1024 \
  --repeats 2 \
  --warmups 1 \
  --layers 8 \
  --output /tmp/mklq-benchmark-isolated.json
```

## Output

The JSON report includes:

- machine metadata: platform, macOS version, CPU brand, core count, memory
- provenance metadata: cwd, git branch/commit/dirty status, and selected
  OpenMP/vector-library environment variables
- runtime metadata: CUDA-Q module path/version and Python path context for
  non-isolated runs, or per-row child runtime metadata with `--isolate-rows`
- command/config metadata: targets, cases, qubits, shots, shot counts, repeats,
  warmups
- per-row measurements: elapsed time, throughput/latency, estimated state bytes,
  and cumulative process max RSS

`single-qubit-state`, `h-state`, `y-state`, `rx-state`, `ry-state`,
`rz-state`,
`controlled-state`, `ch-state`, `cy-state`, `crx-state`, `cry-state`, `crz-state`,
`cz-state`, and
`two-qubit-state` are focused state-vector update microbenchmarks. The dedicated
H/Y/Rx/Ry/Rz cases initialize a non-uniform state, then apply layers of one
built-in single-qubit gate; their elapsed times include the state-preparation
gates, while the gate-specific throughput fields use only the repeated
target-gate count. Use those rows to compare built-in uncontrolled
single-qubit hot paths, not custom or controlled gate behavior. The dedicated
CH/CY/CRX/CRY/CRZ cases initialize a non-uniform state, then apply layers of one
built-in controlled single-qubit gate; their elapsed times include the
state-preparation gates, while the controlled-gate throughput fields use only
the repeated target-gate count. Use those rows as evidence for built-in
controlled-gate hot paths, not custom controlled operations. The `cz-state` case
initializes a
non-uniform state, then applies CZ-only layers; use it as evidence for the CZ
phase fast path, not as a general claim about every controlled single-qubit
gate. The `two-qubit-state` case initializes a non-uniform state, then applies
SWAP layers; use it as evidence for this hot path, not as a general claim about
every custom 4x4 gate.
`sample-basis` targets deterministic sparse full-register sampling from the
allocated `|0...0>` basis state. `sample-ghz` targets sparse full-register
sampling with two nonzero outcomes, while
`sample-full-register` targets dense full-register sampling after non-uniform
single-qubit rotations. `sample-partial-register` applies non-uniform
single-qubit rotations across the whole state but measures only every other
qubit; use it to exercise cost-gated partial-register sampling paths. Standard
non-explicit `cudaq.sample` rows use counts-only aggregation rather than
retaining per-shot sequential data, so dense sampling rows exercise the
backend's aggregate draw-count path. The benchmark does not call the public
sequential-data accessor, which may expand counts on demand for API
compatibility. For
`mklq-metal`, small marginal buffers use the resident marginal probability
kernel, while q15-q20 every-other-qubit rows currently route to resident
full-register probability fill plus host marginal folding.

Measured timings are post-warmup execution calls; target setup and kernel
construction are outside the timed region. `process_max_rss_bytes` is the
maximum RSS for the benchmark Python process. The JSON field is named
`process_max_rss_bytes_cumulative` because later rows inherit earlier rows'
memory history. Use one fresh process per row if you need strict per-target
peak memory isolation; `--isolate-rows` automates this for benchmark rows.

By default, the script exits nonzero if any benchmark row has `status != "ok"`.
Use `--allow-errors` only when collecting partial data from experimental
targets.

Use larger qubit counts and repeats only after correctness gates are green. To
run the local aggregate correctness gate before collecting benchmark evidence:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

The gate writes ignored local JSON under `benchmarks/mklq/results/`. When
preserving rejected tuning runs, label them clearly and keep them separate from
the local baseline so they are not read as performance evidence.

## Public Health Check

Use the public health check as the default local pre-push maintenance command:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py
```

The default mode checks Git remotes and shallow state, tracked artifact hygiene,
public metadata and banned tokens, sanitized benchmark summary JSON, helper
syntax, local markdown links, regenerated benchmark-evidence consistency, and
the benchmark harness tests. It writes an ignored JSON report under
`benchmarks/mklq/results/`.

Before describing a commit as public-ready, run the heavier local gate:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py --full --require-clean
```

`--full` adds the install-prefix build and one-command correctness gate. It does
not refresh benchmark evidence. To intentionally refresh clean CPU benchmark
evidence, run from a clean worktree:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py \
  --full \
  --require-clean \
  --refresh-clean-cpu-benchmark
```

## Tracked Accepted Local Benchmark Evidence

To rerun the clean CPU benchmark gate, regenerate the sanitized summary, and
refresh the public evidence index, run:

```bash
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp 2026-06-21
```

The gate writes ignored raw JSON under `benchmarks/mklq/results/`, writes the
sanitized summary under `benchmarks/mklq/reports/`, and refreshes
`docs/mklq/benchmark-evidence.md`. It refuses to collect clean evidence from a
dirty worktree unless `--allow-dirty` is passed explicitly.

If the raw JSON already exists and you only need to regenerate the sanitized
summary and public evidence index, run:

```bash
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp 2026-06-21 \
  --skip-benchmark
```

For a compact table across all tracked sanitized summaries, run:

```bash
python3 benchmarks/mklq/summarize_reports.py \
  --reports benchmarks/mklq/reports \
  --format markdown \
  --output docs/mklq/benchmark-evidence.md
```

The generated public index is tracked at
`docs/mklq/benchmark-evidence.md`.

- `reports/local-clean-cpu-q20-2026-06-21.summary.json`: tracked sanitized
  summary for ignored raw results
  `results/local-clean-cpu-gate-y-cy-q20-2026-06-21.json`
  (`sha256: fe2c7b1f755924fc8ba8034e8c1cf0743dd11f6a0d965487578f76e0b5d9ce75`)
  and `results/local-clean-cpu-sampling-q20-2026-06-21.json`
  (`sha256: 98e06439cb50fdc1dac435ce2bccb50ca570c1086603533964f0ea5b0342123c`).
  This run was collected from a clean worktree at
  `8946ad33679f60d7c22dc55415fc60b048ef614c` with `qpp-cpu` and `mklq-cpu`
  rows for `y-state`, `cy-state`, `sample-full-register`, and
  `sample-partial-register` at q20 with `OMP_NUM_THREADS=10`,
  `OMP_PROC_BIND=close`, `OMP_DYNAMIC=false`, `VECLIB_MAXIMUM_THREADS=1`,
  `repeats=2`, `warmups=1`, and `layers=8` on Apple M5, 10 logical cores,
  16 GB RAM, macOS 26.5.1. All 12 rows completed with `status == "ok"`.
  In this local run, q20 median elapsed ratios for `qpp-cpu` over `mklq-cpu`
  were 119.16x for `y-state`, 96.35x for `cy-state`, 93.42x for
  `sample-full-register` at 1024 shots, 65.16x for `sample-full-register` at
  65536 shots, 106.72x for `sample-partial-register` at 1024 shots, and 95.37x
  for `sample-partial-register` at 65536 shots. Treat this as local
  clean-worktree CPU evidence, not as cross-machine performance certification.
- `reports/local-current-sampling-fullprob-gated-q20-2026-06-19.summary.json`:
  tracked sanitized summary for the ignored raw result
  `results/local-current-sampling-fullprob-gated-q20-2026-06-19.json`
  (`sha256: 8ca6a4f7a7aea1670aa572ea6897a125ea4ff0a9e0d1d93502c1158e81ba33b3`).
  isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for
  `sample-full-register` and `sample-partial-register` at q20 with
  `OMP_NUM_THREADS=10`, `OMP_PROC_BIND=close`, `OMP_DYNAMIC=false`,
  `shots=1024`, `repeats=2`, `warmups=1`, and `layers=4` on Apple M5,
  10 logical cores, 16 GB RAM, macOS 26.5.1. All six rows completed with
  `status == "ok"`. Treat this as local tuning evidence for the Metal
  partial-register sampling cost gate, not as clean-release provenance. In this
  run, q20 `mklq-metal` median elapsed time was 0.0370576665 s for
  `sample-full-register` and 0.022011521 s for `sample-partial-register`.
  The same-day pre-gate probe
  `results/local-current-sampling-shot-scaling-q20-2026-06-19.json` measured
  0.255696875 s for the q20 `mklq-metal` `sample-partial-register` row at
  1024 shots with `repeats=1`, so use the comparison as a tuning signal rather
  than a formal release benchmark.
- `reports/local-counts-only-sampling-shot-scaling-q20-2026-06-19.summary.json`:
  tracked sanitized summary for the ignored raw result
  `results/local-counts-only-sampling-shot-scaling-q20-2026-06-19.json`
  (`sha256: ef9846673b461e3abc6d359933408be58e1f745d8b68738b757a76339f9b5092`).
  Isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for
  `sample-full-register` and `sample-partial-register` at q20 with
  `OMP_NUM_THREADS=10`, `OMP_PROC_BIND=close`, `OMP_DYNAMIC=false`, shot counts
  `256,1024,8192,65536`, `repeats=2`, `warmups=1`, and `layers=8` on Apple M5,
  10 logical cores, 16 GB RAM, macOS 26.5.1. All 24 rows completed with
  `status == "ok"`. Treat this as local dirty-worktree tuning evidence for the
  standard non-explicit `cudaq.sample` counts-only backend path, not as
  clean-release provenance. The benchmark does not call
  `sample_result::sequential_data()`, so it measures backend sample/count
  aggregation rather than the public accessor's lazy counts expansion. In this
  run at q20 and 65536 shots, `mklq-cpu` median elapsed time was
  0.01916737499414012 s for `sample-full-register` and 0.016119854502903763 s
  for `sample-partial-register`; `mklq-metal` median elapsed time was
  0.04015256251295796 s and 0.03547552099917084 s for the same cases.
- `reports/local-y-cy-fastpath-isolated-q20-2026-06-19.summary.json`:
  tracked sanitized summary for the ignored raw result
  `results/local-y-cy-fastpath-isolated-q20-2026-06-19.json`
  (`sha256: 93bce3b77fccce0ce48611fbccc2a88d81e31b8a34f4885ff9235750178701fa`).
  Isolated `qpp-cpu` and `mklq-cpu` rows for `y-state` and `cy-state` at q20
  with `OMP_NUM_THREADS=10`, `OMP_PROC_BIND=close`, `OMP_DYNAMIC=false`,
  `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8` on Apple M5,
  10 logical cores, 16 GB RAM, macOS 26.5.1. All four rows completed with
  `status == "ok"`. Treat this as local dirty-worktree tuning evidence for the
  CPU built-in Y/CY structured fast path, not as clean-release provenance. In
  this run, q20 `mklq-cpu` median elapsed time was 0.04815118750411784 s for
  `y-state` and 0.08607120799570112 s for `cy-state`.
- `reports/local-metal-y-cy-resident-isolated-q20-2026-06-19.summary.json`:
  tracked sanitized summary for the ignored raw result
  `results/local-metal-y-cy-resident-isolated-q20-2026-06-19.json`
  (`sha256: 84891e8f907c38295a4975b1d0b0c493c2658b9b36b29975c539b93fcdfff9bb`).
  Isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for `y-state` and
  `cy-state` at q20 with `OMP_NUM_THREADS=10`, `OMP_PROC_BIND=close`,
  `OMP_DYNAMIC=false`, `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8`
  on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All six rows
  completed with `status == "ok"`. Treat this as local dirty-worktree tuning
  evidence for resident fp32 Metal Y/CY gate updates followed by host readback
  for `cudaq.get_state`, not as clean-release provenance. In this run, q20
  `mklq-metal` median elapsed time was 0.09025897899846314 s for `y-state` and
  0.09229137500369688 s for `cy-state`. The summary's path labels are curated
  labels inferred from runtime tests and code inspection; the raw benchmark
  JSON does not currently emit resident-path counters.

## Untracked Local Benchmark Notes

The following ignored `results/*.json` files are local development notes only.
They are useful for understanding why a path was tuned, but they are not
accepted commit evidence unless a tracked sanitized summary under `reports/`
records their hashes and bounded metrics.

- `results/local-controlled-h-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `ch-state` at q15-q20 with
  `OMP_NUM_THREADS=10`, `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8`
  on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All 12 rows completed
  with `status == "ok"`. The benchmark JSON records a dirty worktree, so treat
  this as local development evidence for the CPU built-in controlled H
  structured fast path, not as clean-release provenance. At q20, median elapsed
  time was 9.98382543751 s for `qpp-cpu` and 0.103235583993 s for `mklq-cpu`, a
  96.71x local cross-target ratio. This run was collected after the controlled H
  fast path was already implemented, so it is not a before/after speedup claim.
- `results/local-controlled-rotation-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `crx-state`, `cry-state`, and
  `crz-state` at q15-q20 with `OMP_NUM_THREADS=10`, `shots=1024`,
  `repeats=2`, `warmups=1`, and `layers=8` on Apple M5, 10 logical cores,
  16 GB RAM, macOS 26.5.1. All 36 rows completed with `status == "ok"`. The
  benchmark JSON records a dirty worktree, so treat this as local
  development evidence for the CPU built-in controlled Rx/Ry/Rz structured fast
  paths, not as clean-release provenance. Compared with
  `results/local-controlled-rotation-breakdown-isolated-q15-q20-2026-06-18.json`,
  q20 `mklq-cpu` median elapsed time changed from 0.128347500002 s to
  0.0893733750054 s for `crx-state`, from 0.127916166493 s to
  0.0786046455032 s for `cry-state`, and from 0.123464874996 s to
  0.115874896001 s for `crz-state`. That is 1.44x, 1.63x, and 1.07x local q20
  speedup respectively. Across q15-q20 this run improved all three dedicated
  controlled-rotation cases, but CRZ's q20 improvement was small, so rerun with
  higher repeats before making fine-grained CRZ tuning claims.
- `results/local-controlled-rotation-breakdown-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `crx-state`, `cry-state`, and
  `crz-state` at q15-q20 with the same command shape as the fast-path run
  above. All 36 rows completed with `status == "ok"`, and this JSON also records
  a dirty worktree. Treat this as the local pre-fast-path breakdown that
  identified built-in controlled rotations as a remaining `controlled-state`
  hot path, not as current performance evidence or clean-release provenance.
- `results/local-single-gate-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `h-state`, `rx-state`,
  `ry-state`, and `rz-state` at q15-q20 with `OMP_NUM_THREADS=10`,
  `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8` on Apple M5,
  10 logical cores, 16 GB RAM, macOS 26.5.1. All 48 rows completed with
  `status == "ok"`. The benchmark JSON records a dirty worktree, so treat this
  as local development evidence for the CPU built-in uncontrolled
  H/Rx/Ry/Rz structured fast paths, not as clean-release provenance. Compared with
  `results/local-single-gate-breakdown-isolated-q15-q20-2026-06-18.json`,
  q20 `mklq-cpu` median elapsed time changed from 0.110392645998 s to
  0.0517783540017 s for `h-state`, from 0.128242312501 s to
  0.0645899794981 s for `rx-state`, from 0.104353729501 s to
  0.0679914585016 s for `ry-state`, and from 0.107659312496 s to
  0.0797631040004 s for `rz-state`. That is 2.13x, 1.99x, 1.53x, and 1.35x
  local q20 speedup respectively. Across q15-q20, H/Rx/Ry improved in this
  run; Rz improved from q16-q20 but the q15 row was slower, 0.02130858349846676 s
  after versus 0.018260312001075363 s before, so do not cite this as an
  all-size Rz win without a higher-repeat follow-up.
- `results/local-single-gate-breakdown-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `h-state`, `rx-state`,
  `ry-state`, and `rz-state` at q15-q20 with the same command shape as the
  fast-path run above. All 48 rows completed with `status == "ok"`, and this
  JSON also records a dirty worktree. Treat this as the local pre-fast-path
  breakdown that identified `rx-state` as the slowest q20 `mklq-cpu` dedicated
  single-gate case before the structured H/Rx/Ry/Rz CPU fast paths landed, not
  as current performance evidence or clean-release provenance.
- `results/local-cz-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu` and `mklq-cpu` rows for `cz-state` at q15-q20 with
  `OMP_NUM_THREADS=10`, `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8`
  on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All 12 rows completed
  with `status == "ok"`. Treat this as the local cross-target evidence
  for the CPU built-in CZ phase fast path. At q20, median elapsed time was
  7.334453104002023 s for `qpp-cpu` and 0.10550895799678983 s for `mklq-cpu`, a
  69.51x local cross-target ratio. This benchmark is not a before/after
  comparison against an older MKL-Q implementation, because `cz-state` was added
  with this change.
- `results/local-bitflip-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `mklq-cpu` rows for `gate-state` and `controlled-state` at q15-q20
  with `OMP_NUM_THREADS=10`, `shots=1024`, `repeats=2`, `warmups=1`, and
  `layers=8` on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All 12
  rows completed with `status == "ok"`. Treat this as the local
  evidence for the CPU built-in CNOT/controlled-X bit-flip permutation fast
  path. Compared with
  `results/local-focused-allcases-isolated-q15-q20-2026-06-18.json`, median q20
  `gate-state` elapsed time changed from 0.683557729000313 s to
  0.25968133350033895 s, a 2.63x local speedup, and q20 `controlled-state`
  elapsed time changed from 0.4340257920011936 s to 0.2211792920024891 s, a
  1.96x local speedup. Built-in X correctness is covered by unit tests, but this
  benchmark does not claim standalone X performance improvements.
- `results/local-swap-fastpath-isolated-q15-q20-2026-06-18.json`:
  isolated `mklq-cpu` rows for `two-qubit-state` at q15-q20 with
  `OMP_NUM_THREADS=10`, `shots=1024`, `repeats=2`, `warmups=1`, and `layers=8`
  on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All six rows
  completed with `status == "ok"`. Treat this as the local evidence
  for the CPU SWAP permutation fast path. Compared with
  `results/local-focused-allcases-isolated-q15-q20-2026-06-18.json`, median
  `two-qubit-state` elapsed time at q20 changed from 0.3724326045012276 s to
  0.04745125000044936 s, a 7.85x local speedup. q17-q20 improved by
  3.53x-7.85x, and all q15-q20 rows remained `status == "ok"` in this rerun.
- `results/local-focused-allcases-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for
  `gate-state`, `sample-basis`, `sample-ghz`, `sample-full-register`,
  `sample-partial-register`, `single-qubit-state`, `controlled-state`, and
  `two-qubit-state` at q15-q20 with `OMP_NUM_THREADS=10`, `shots=1024`,
  `repeats=2`, `warmups=1`, and `layers=8` on Apple M5, 10 logical cores,
  16 GB RAM, macOS 26.5.1. All 144 rows completed with `status == "ok"`.
  Treat this as the current local focused benchmark baseline for choosing the
  next hot path, not as portable performance evidence. In its q20 rows,
  `mklq-cpu` median elapsed time was 0.6906239789932442 s for
  `single-qubit-state`, 0.683557729000313 s for `gate-state`,
  0.4340257920011936 s for `controlled-state`, and
  0.3724326045012276 s for `two-qubit-state`; these state-update rows are the
  largest remaining `mklq-cpu` local latencies in this benchmark. In the same
  q20 rows, `mklq-metal` was faster than `mklq-cpu` for dense state-update and
  dense sampling cases, but slower for `sample-basis` and `sample-ghz`, so do
  not move sparse sampling work to Metal without a separate benchmark gate.
- `results/local-metal-sampling-shot-scaling-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for
  `sample-full-register` and `sample-partial-register` at q15-q20 with
  `OMP_NUM_THREADS=10`, `shot_counts=256,1024,8192,65536`, `repeats=2`, and
  `warmups=1` on Apple M5, 10 logical cores, 16 GB RAM, macOS 26.5.1. All
  144 rows completed with `status == "ok"`. Treat this as the local
  shot-scaling gate for sample draw/count decisions, not as portable
  performance evidence. In its q20 `mklq-metal` rows, median elapsed time
  changed from 0.02171218749936088 s at 256 shots to 0.027931499997066567 s at
  65536 shots for `sample-full-register`, and from 0.024053000001003966 s to
  0.02549454149630037 s for `sample-partial-register`; this measured range
  does not justify moving sample count accumulation onto the GPU yet. The next
  accepted low-risk step is host-side counts-only aggregation for
  `includeSequentialData=false`, using bounded dense counters for small outcome
  spaces and sparse maps for larger outcome spaces; this is not Metal RNG or
  GPU count accumulation.
- `results/local-sample-basis-isolated-q15-q20-2026-06-18.json`:
  isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for deterministic
  `sample-basis` at q15-q20 with `OMP_NUM_THREADS=10`, `shots=1024`,
  `repeats=2`, and `warmups=1`. This run covers sparse full-register sampling
  from the allocated `|0...0>` state with one deterministic outcome. Treat it
  as local Apple M5 evidence for the deterministic sparse sampling path only.
- `results/local-sampling-full-partial-fullprob-isolated-q15-q20-2026-06-18.json`:
  current isolated `qpp-cpu`, `mklq-cpu`, and `mklq-metal` rows for
  `sample-full-register` and `sample-partial-register` at q15-q20 with
  `OMP_NUM_THREADS=10`, `shots=1024`, `repeats=2`, and `warmups=1`.
  This run covers the current `mklq-metal` partial-register path, which fills
  resident full-register probabilities once and folds them to marginal
  probabilities on the host. Treat this as local Apple M5 evidence for sampling
  latency and memory rows, not as a portable performance claim.
- `results/local-sampling-full-partial-isolated-q15-q20-2026-06-18.json`:
  historical before/after comparison point for the earlier isolated
  `sample-full-register` and `sample-partial-register` run. Its Metal
  partial-register rows used the earlier marginal scan path, so do not treat it
  as evidence for the current implementation.

## Probability Kernel Microbenchmark

Use this standalone C++ microbenchmark before changing the dense
full-register probability helper in `runtime/nvqir/mklq`. It compares the
probability-vector fill kernels without Python or CUDA-Q target overhead:

```bash
OMP_NUM_THREADS=4 \
python3 benchmarks/mklq/bench_probability_kernels.py \
  --variants scalar-norm,scalar-split,accelerate-interleaved,accelerate-vdsp,openmp-split \
  --qubits 15,16,17,18,19,20 \
  --repeats 5 \
  --warmups 2 \
  --output benchmarks/mklq/results/local-probability-kernels-interleaved-vdsp-omp4-q15-q20-2026-06-19.json
```

The local Apple M5 run in
`results/local-probability-kernels-interleaved-vdsp-omp4-q15-q20-2026-06-19.json`
with 10 logical cores, 16 GB RAM, macOS 26.5.1, and `OMP_NUM_THREADS=4`
produced `ok` rows for all variants. In that run, `scalar-split` was fastest at
q15 and `openmp-split` was fastest at q16-q20. The runtime-shaped
`accelerate-interleaved` variant was slower than `openmp-split` across q16-q20;
at q20 its median elapsed time was 0.000581792 s versus 0.00016575 s for
`openmp-split`. The older `accelerate-vdsp` split-complex variant was also
slower than `openmp-split` at q16-q20. Treat these rows as evidence for keeping
the default `mklq-cpu` dense full-register probability fill on the OpenMP/scalar
path for now, not as a whole-backend performance claim.
