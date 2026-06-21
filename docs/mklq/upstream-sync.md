# MKL-Q Upstream Sync

This procedure keeps MKL-Q synchronized with `NVIDIA/cuda-quantum` without
turning the fork into a hard rename or losing Apple Silicon target work.

Use it when pulling upstream CUDA-Q changes into MKL-Q `main`.

## Preflight

Start from a clean local branch:

```bash
git status --short --branch
git remote -v
git rev-parse --is-shallow-repository
git sparse-checkout list
gh repo view wuls968/MKL-Q --json nameWithOwner,isFork,parent,defaultBranchRef,url
```

Expected:

- `main` is clean.
- `origin` points to `https://github.com/wuls968/MKL-Q.git`.
- `upstream` points to `https://github.com/NVIDIA/cuda-quantum.git`.
- The repository is not shallow.
- Sparse checkout includes `.github`, `docs`, `runtime`, `python`,
  `targettests`, `unittests`, `benchmarks`, and top-level metadata files.
- `wuls968/MKL-Q` remains a fork of `NVIDIA/cuda-quantum`.

Do not push to `upstream`. All MKL-Q publication goes through `origin`.

## Inspect Upstream Delta

Fetch both remotes first:

```bash
git fetch origin main
git fetch upstream main
git switch main
git pull --ff-only origin main
git log --oneline --decorate --left-right main...upstream/main | sed -n '1,120p'
git diff --name-status main...upstream/main | sed -n '1,220p'
```

Classify the delta before merging:

- `.github/`: review manually; do not restore upstream release, bot, or heavy
  workflow automation by default.
- top-level metadata: protect MKL-Q README, citation, contributing, security,
  license, notice, and source-only public messaging.
- `runtime/nvqir/`, `targettests/`, `python/tests/`, `unittests/`: check for
  runtime contract changes that affect MKL-Q target registration or simulator
  APIs.
- `cmake/`, `CMakeLists.txt`, `scripts/`: check for build-system changes that
  affect `CUDAQ_ENABLE_MKLQ_BACKEND` or target config installation.
- `benchmarks/mklq/` and `docs/mklq/`: preserve MKL-Q evidence boundaries and
  public docs unless intentionally updated.

## Sync Branch

Create a dedicated branch:

```bash
git switch -c codex/upstream-sync-YYYYMMDD
git merge --no-ff upstream/main
```

Use a merge commit for upstream sync work. It preserves the fork relationship
and keeps upstream changes auditable. Do not squash upstream history into a
large synthetic rewrite.

If the merge is clearly wrong before conflicts are resolved:

```bash
git merge --abort
```

## Conflict Rules

Resolve conflicts by preserving upstream CUDA-Q behavior first, then reapplying
MKL-Q additive target work.

Keep these MKL-Q-owned surfaces unless the sync intentionally changes them:

- `runtime/nvqir/mklq/`
- `runtime/nvqir/mklq/mklq-cpu.yml`
- `runtime/nvqir/mklq/mklq-metal.yml`
- `targettests/TargetConfig/mklq_targets.config`
- `targettests/TargetConfig/mklq_runtime_smoke.cpp`
- `python/tests/backends/test_mklq_*.py`
- `python/tests/builder/test_mklq_targets.py`
- `python/tests/mklq_test_utils.py`
- `unittests/nvqpp/backends/MKLQCpuTester.cpp`
- `unittests/nvqpp/backends/MKLQMetalTester.cpp`
- `benchmarks/mklq/`
- `docs/mklq/`
- `.github/workflows/mklq-public-hygiene.yml`
- `.github/ISSUE_TEMPLATE/`
- `.github/pull_request_template.md`

Protect these invariants:

- Python package namespace remains `cudaq`.
- C++ compiler entry point remains `nvq++`.
- `mklq-cpu` remains the stable fp64 CPU target.
- `mklq-metal` remains experimental and mixed-path.
- Upstream targets such as `qpp-cpu` remain available.
- Raw local benchmark JSON under `benchmarks/mklq/results/` stays untracked.
- Public docs do not claim wheel, PyPI, release, or full Metal-native support.
- Public GitHub Actions remains lightweight unless a heavier workflow is
  intentionally reviewed.

## Post-merge Gates

Run source hygiene first:

```bash
git diff --check
if git ls-files | grep -E '(^|/)(__pycache__|\.pytest_cache)(/|$)|\.pyc$|\.DS_Store$|^build(-python)?/|^benchmarks/mklq/results/|^docs/superpowers/|^(dist|wheelhouse)/|\.(whl|dmg|pkg|zip)$|\.tar\.gz$'; then
  echo "Generated, local, release, or agent-internal files are tracked."
  exit 1
fi
```

Run public hygiene locally using the same classes of checks as
`.github/workflows/mklq-public-hygiene.yml`:

```bash
python3 -m py_compile \
  benchmarks/mklq/bench_mklq_targets.py \
  benchmarks/mklq/bench_probability_kernels.py \
  benchmarks/mklq/make_summary.py \
  benchmarks/mklq/run_clean_cpu_benchmark.py \
  benchmarks/mklq/run_correctness_gate.py \
  benchmarks/mklq/summarize_reports.py
```

If upstream changed build, runtime, target config, Python bindings, or tests,
rebuild before claiming the sync is ready:

```bash
cmake --build build-python --target install -j 6
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

Use `docs/mklq/testing-matrix.md` to choose any additional focused gates. For
example, backend implementation conflicts should include the relevant
`MKLQCpuTester` or `MKLQMetalTester` coverage, and target config conflicts
should include TargetConfig `ctest` plus `nvq++` smoke evidence.

## Benchmark Evidence

Do not collect new benchmark evidence until correctness gates pass.

If performance-sensitive MKL-Q paths changed, collect clean local evidence:

```bash
git status --short --branch
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp YYYY-MM-DD
```

Commit only sanitized summaries under `benchmarks/mklq/reports/` and the
regenerated `docs/mklq/benchmark-evidence.md`. Do not commit raw local JSON.

## Publish

After local gates pass:

```bash
git status --short --branch
git push origin HEAD
gh run list --repo wuls968/MKL-Q --branch "$(git branch --show-current)" --limit 5
```

For a branch-based review, open a pull request and use the PR template. For a
maintainer direct-to-main sync, fast-forward or merge only after the same gates
pass locally, then confirm the remote workflow result:

```bash
git switch main
git pull --ff-only origin main
git merge --ff-only codex/upstream-sync-YYYYMMDD
git push origin main
gh run watch <run-id> --repo wuls968/MKL-Q --exit-status
```

## Stop Conditions

Do not publish the sync if any of these are true:

- The merge restores upstream heavy `.github` automation without review.
- The branch rewrites MKL-Q as a hard project rename.
- `mklq-metal` becomes the default target or is described as full
  Metal-native.
- Upstream target compatibility is broken without an explicit migration plan.
- Raw benchmark JSON, build output, caches, local signing objects, or private
  paths become tracked.
- The one-command correctness gate fails after runtime or target changes.
- The latest GitHub Actions result for the pushed commit is failing or unknown.

## After Sync

Update these documents if the sync changes the public support boundary:

- `docs/mklq/architecture.md`
- `docs/mklq/testing-matrix.md`
- `docs/mklq/known-limitations.md`
- `docs/mklq/validation.md`
- `docs/mklq/benchmark-evidence.md`
- `docs/mklq/public-release-checklist.md`

Keep the final report tied to evidence: upstream commit range, conflicts
resolved, gates run, benchmark evidence changed or unchanged, and remaining
risks.
