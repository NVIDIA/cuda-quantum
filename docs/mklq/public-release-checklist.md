# MKL-Q Public Release Checklist

This checklist is for preparing the public `main` branch of MKL-Q. It describes
the current source-only release-readiness gate; it is not a wheel, PyPI, binary,
or GitHub Release procedure.

## Scope

- [ ] Keep the repository as an upstream-compatible fork of NVIDIA CUDA-Q.
- [ ] Keep the first public version source-only.
- [ ] Do not create tags, GitHub Releases, wheels, PyPI packages, installers,
  or signed artifacts as part of this checklist.
- [ ] `docs/mklq/release-policy.md` confirms the current branch is source-only
  and has no release artifacts.
- [ ] Keep the public API surface compatible with CUDA-Q: Python users import
  `cudaq`, and C++ users compile with `nvq++`.
- [ ] Keep `mklq-cpu` as the stable local target and `mklq-metal` as an
  experimental mixed-path target.

## Git And Remotes

Run:

```bash
git status --short --branch
git remote -v
git rev-parse --is-shallow-repository
git log --oneline -5
```

Expected:

- [ ] `main` is clean before collecting clean evidence.
- [ ] `origin` points to the public MKL-Q fork.
- [ ] `upstream` points to `https://github.com/NVIDIA/cuda-quantum.git`.
- [ ] The repository is not shallow before public publication.
- [ ] The latest commits are MKL-Q commits on top of upstream CUDA-Q history.
- [ ] If upstream CUDA-Q was synced, `docs/mklq/upstream-sync.md` was followed
  and the sync gates are recorded in the change summary.

## Public Metadata

Check:

- [ ] `README.md` identifies MKL-Q as a CUDA-Q-compatible Apple Silicon fork.
- [ ] `README.md` says the first public version is source-only.
- [ ] `CITATION.cff`, `Contributing.md`, `SECURITY.md`, `LICENSE`, and
  `NOTICE` do not misdirect users to NVIDIA-only project contacts or workflows.
- [ ] `docs/mklq/known-limitations.md` is linked from the README and explains
  the current support boundary.
- [ ] `docs/mklq/testing-matrix.md` is linked from the README and explains
  which local gates prove which target/backend behavior.
- [ ] `docs/mklq/release-policy.md` is linked from the README and explains
  current source-only release boundaries.
- [ ] `docs/mklq/maintainer-runbook.md` is linked from the README and explains
  maintainer triage, validation, and recovery boundaries.
- [ ] `docs/mklq/issue-labels.md` is linked from the README and matches
  `.github/labels.yml`.
- [ ] `docs/mklq/branch-protection.md` is linked from the README and matches
  `.github/branch-protection-main.json`.
- [ ] `docs/mklq/public-readiness.md` is linked from the README and records the
  current public repository readiness snapshot.
- [ ] GitHub About metadata describes MKL-Q, uses Apache-2.0, and avoids stale
  NVIDIA workflow or badge links.

## Tree Hygiene

Run:

```bash
git status --ignored --short
git ls-files .github | sort
git diff --check
```

Expected:

- [ ] No tracked `build/`, `build-python/`, `__pycache__/`, `.pytest_cache/`,
  `.DS_Store`, local signing objects, or raw `benchmarks/mklq/results/*.json`.
- [ ] No tracked `dist/`, `wheelhouse/`, wheel, installer, signed artifact, or
  release archive files.
- [ ] No `docs/superpowers/` or agent-internal paths are tracked.
- [ ] `.github/workflows/` contains only the lightweight MKL-Q public hygiene
  workflow unless a new workflow has been intentionally reviewed.
- [ ] `git diff --check` has no whitespace errors.

## Local Build Gate

Run on Apple Silicon macOS:

```bash
cmake --build build-python --target install -j 6
```

Expected:

- [ ] The install prefix is the intended local prefix, usually
  `${HOME}/.cudaq-mklq`.
- [ ] Build and install complete without errors.
- [ ] The installed Python path and `nvq++` path match the next gate.

## Correctness Gate

Run:

```bash
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

Expected:

- [ ] Python target fixtures pass for `mklq-cpu` and the limited experimental
  `mklq-metal` fixture suite.
- [ ] `nvq++` smoke tests pass for `mklq-cpu` and `mklq-metal`.
- [ ] TargetConfig `ctest` selection passes.
- [ ] The generated JSON is stored under ignored `benchmarks/mklq/results/` or
  another local path, not committed as public evidence.

## Benchmark Evidence

Run only after the correctness gate is green:

```bash
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp YYYY-MM-DD
```

Expected:

- [ ] Clean benchmark evidence is collected only from a clean worktree.
- [ ] Raw local JSON remains ignored under `benchmarks/mklq/results/`.
- [ ] Sanitized summaries are written under `benchmarks/mklq/reports/`.
- [ ] `docs/mklq/benchmark-evidence.md` is regenerated.
- [ ] Each summary is interpreted through its `evidence_kind` and
  `interpretation` fields; no local result is treated as cross-machine
  performance certification.

## Public Hygiene Gate

Run the same classes of checks as `.github/workflows/mklq-public-hygiene.yml`:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py
```

For pre-publication local evidence that also rebuilds and reruns backend
correctness:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py --full --require-clean
```

The underlying lightweight checks include:

```bash
git diff --check
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
  benchmarks/mklq/run_public_healthcheck.py \
  benchmarks/mklq/summarize_reports.py \
  examples/mklq/python/bell.py \
  examples/mklq/python/clifford_chain.py \
  examples/mklq/python/ghz.py \
  examples/mklq/python/parametric.py \
  examples/mklq/python/phase_kickback.py \
  examples/mklq/verify_examples.py
```

Expected:

- [ ] Public metadata keywords are present.
- [ ] Public example files exist under `examples/mklq/`.
- [ ] Banned upstream workflow/contact tokens are absent from public metadata
  and `.github`.
- [ ] Sanitized benchmark summaries parse as JSON.
- [ ] `check_performance_evidence.py` passes for tracked clean CPU summaries.
- [ ] `check_metal_evidence.py` passes for tracked experimental Metal
      summaries.
- [ ] Public benchmark helper and Python example scripts compile.
- [ ] `run_public_healthcheck.py` passes in default mode.
- [ ] `run_public_healthcheck.py --full --require-clean` passes before
  describing the commit as public-ready.

## Push And GitHub Verification

After local gates pass:

```bash
git push origin main
git ls-remote origin refs/heads/main
gh repo view wuls968/MKL-Q --json nameWithOwner,isFork,parent,defaultBranchRef,url
gh run list --repo wuls968/MKL-Q --branch main --limit 5
```

Expected:

- [ ] Remote `main` equals local `HEAD`.
- [ ] `wuls968/MKL-Q` remains a fork of `NVIDIA/cuda-quantum`.
- [ ] The default branch is `main`.
- [ ] Only intended MKL-Q lightweight workflows run.
- [ ] The latest MKL-Q public hygiene workflow completes with `success`.
- [ ] `main` branch protection is enabled and requires
  `Source-only repository checks`.
- [ ] `docs/mklq/public-readiness.md` is current for the pushed commit.

## Stop Conditions

Do not publish or describe the branch as ready if any of these are true:

- [ ] The worktree is dirty and the change was not intentionally reviewed.
- [ ] Raw local benchmark payloads, generated files, build products, or
  private artifacts are tracked.
- [ ] `mklq-metal` is described as full Metal-native or default-ready.
- [ ] Local benchmark evidence is described as release certification.
- [ ] The GitHub Actions run for the pushed commit is failing or still unknown.
