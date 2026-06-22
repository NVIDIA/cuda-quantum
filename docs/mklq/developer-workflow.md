# MKL-Q Developer Workflow

This workflow is for source changes in the public MKL-Q fork. It keeps the
project easy to sync with upstream CUDA-Q while allowing focused Apple Silicon
backend work.

## Baseline Rules

- Keep CUDA-Q API compatibility unless the change is explicitly limited to an
  MKL-Q target.
- Keep the Python package namespace as `cudaq` and the compiler entry point as
  `nvq++`.
- Keep `mklq-cpu` as the stable correctness target.
- Keep `mklq-metal` experimental until its supported paths pass the same
  correctness and benchmark gates as CPU.
- Do not commit generated files, build directories, raw benchmark JSON, local
  signing objects, caches, secrets, tokens, or machine-specific credentials.

## Branch Setup

Start each change from public `main`:

```bash
git status --short --branch
git fetch origin main
git switch main
git pull --ff-only origin main
git switch -c codex/<short-topic>
```

Keep `upstream` configured for CUDA-Q:

```bash
git remote -v
git remote get-url upstream
```

Expected upstream URL:

```text
https://github.com/NVIDIA/cuda-quantum.git
```

## Upstream Sync

Do not hard-rewrite MKL-Q onto a renamed tree. Follow the dedicated upstream
sync procedure:

```text
docs/mklq/upstream-sync.md
```

After an upstream sync, run the public hygiene gate and the smallest relevant
MKL-Q correctness gate before adding new backend work. Resolve conflicts by
preserving upstream compatibility first, then reapplying MKL-Q-specific target
registration, docs, and tests.

## Change Classes

Keep changes reviewable:

- Backend code: one target, gate family, or runtime behavior at a time.
- Tests: add or update fixtures in the same batch as the behavior they prove.
- Benchmarks: keep raw local JSON ignored; publish only sanitized summaries.
- Docs: update user-facing docs when install steps, supported targets, gates,
  benchmark interpretation, or public limitations change.
- Repository config: keep GitHub workflows lightweight unless a heavier CI job
  has a clear maintenance owner and cost.

Use [`architecture.md`](architecture.md) as the source of truth for the current
target layering, CPU oracle role, Metal fallback boundary, and benchmark
evidence boundary.
Use [`testing-matrix.md`](testing-matrix.md) to choose the minimum gate that
actually proves a backend, target, benchmark, or docs change.
Use [`maintainer-runbook.md`](maintainer-runbook.md) for issue triage, PR intake,
PR-first maintainer flow, release stop conditions, and recovery steps.

## Local Build

Use the source build path documented in the README:

```bash
cmake -S . -B build-python -D CUDAQ_ENABLE_MKLQ_BACKEND=ON \
  -D CMAKE_INSTALL_PREFIX="${HOME}/.cudaq-mklq"
cmake --build build-python --target install -j 6
```

If the build directory already exists, prefer incremental rebuilds over a clean
build unless CMake configuration changed.

## Correctness Gates

For MKL-Q backend changes, run the one-command gate:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

The default gate also runs `benchmarks/mklq/run_metal_runtime_counter_probe.py`
against the build tree and writes an ignored `.counter.json` report under
`benchmarks/mklq/results/`. Use `--skip-metal-counter-probe` only for focused
debugging when the Metal counter probe is intentionally out of scope.

For focused local debugging, use the smaller checks:

```bash
PYTHONPATH="$(pwd)/build-python/python" \
python3 -m pytest \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q

CUDAQ_NVQPP="$(pwd)/build-python/bin/nvq++" \
PYTHONPATH="$(pwd)/build-python/python" \
python3 -m pytest python/tests/backends/test_mklq_nvqpp_smoke.py -q

ctest --test-dir build-python \
  -R "(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)" \
  --output-on-failure
```

Document any skipped gate in the pull request or change summary. A passing
metadata workflow is not a substitute for backend correctness evidence.

## Benchmark Evidence

Only collect benchmark evidence from a clean worktree after correctness passes:

```bash
git status --short --branch
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp YYYY-MM-DD
```

Publish the sanitized files under `benchmarks/mklq/reports/` and update
`docs/mklq/benchmark-evidence.md`. Do not commit raw
`benchmarks/mklq/results/*.json` files. Treat local results as machine-specific
evidence, not release certification.

## Public Hygiene

Before pushing a public branch, run:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py
git diff --check
git ls-files .github/workflows | sort
python3 benchmarks/mklq/check_performance_evidence.py
python3 benchmarks/mklq/check_metal_evidence.py
python3 -m py_compile \
  benchmarks/mklq/bench_mklq_targets.py \
  benchmarks/mklq/bench_probability_kernels.py \
  benchmarks/mklq/check_metal_evidence.py \
  benchmarks/mklq/check_performance_evidence.py \
  benchmarks/mklq/make_summary.py \
  benchmarks/mklq/run_clean_cpu_benchmark.py \
  benchmarks/mklq/run_correctness_gate.py \
  benchmarks/mklq/run_metal_runtime_counter_probe.py \
  benchmarks/mklq/run_public_readiness_audit.py \
  benchmarks/mklq/run_public_healthcheck.py \
  benchmarks/mklq/summarize_metal_runtime_counters.py \
  benchmarks/mklq/summarize_reports.py \
  examples/mklq/python/bell.py \
  examples/mklq/python/clifford_chain.py \
  examples/mklq/python/ghz.py \
  examples/mklq/python/parametric.py \
  examples/mklq/python/phase_kickback.py \
  examples/mklq/verify_examples.py
```

Expected workflow state:

```text
.github/workflows/mklq-public-hygiene.yml
```

Run the full public release checklist before public release-style milestones:

```text
docs/mklq/public-release-checklist.md
```

## Commit And Push

Use concise English commit messages:

```bash
git status --short
git add <files>
git commit -s -m "docs: update mklq workflow"
git push origin HEAD
```

Do not force push `main`. Do not create tags, GitHub Releases, wheels, or PyPI
packages unless that publishing work has its own reviewed release plan and
passes [`release-policy.md`](release-policy.md).

## Pull Request Review

Each pull request should state:

- What changed.
- Which targets are affected.
- Which correctness gates ran.
- Whether benchmark evidence changed.
- Whether upstream CUDA-Q compatibility is affected.
- Any known limitations or skipped tests.

For backend work, compare `mklq-cpu` and `mklq-metal` behavior against existing
CUDA-Q-compatible target fixtures where practical. Keep `mklq-cpu` as the oracle
for experimental Metal paths until the Metal implementation has stronger
coverage.

## Stop Conditions

Stop and fix before publishing if any of these are true:

- The branch tracks generated files, build output, raw local benchmark JSON, or
  private artifacts.
- Public docs describe `mklq-metal` as fully Metal-native or default-ready.
- A benchmark result is presented as cross-machine certification.
- A workflow file was copied from upstream without reviewing bots, secrets,
  release jobs, permissions, and cost.
- The latest GitHub Actions result for the pushed commit is failing or unknown.
