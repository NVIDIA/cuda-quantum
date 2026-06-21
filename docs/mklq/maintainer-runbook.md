# MKL-Q Maintainer Runbook

This runbook is for maintainers of the public MKL-Q fork. It describes routine
repository operations, issue and pull request triage, validation choices, and
stop conditions for the current source-only project.

Use this page with:

- [`developer-workflow.md`](developer-workflow.md) for contributor-level change
  flow.
- [`testing-matrix.md`](testing-matrix.md) for choosing proof gates.
- [`upstream-sync.md`](upstream-sync.md) for pulling CUDA-Q upstream changes.
- [`release-policy.md`](release-policy.md) before any tag, GitHub Release,
  wheel, PyPI package, installer, or signed artifact.
- [`issue-labels.md`](issue-labels.md) for public GitHub triage labels.
- [`branch-protection.md`](branch-protection.md) for the public `main` branch
  protection policy.
- [`public-readiness.md`](public-readiness.md) for the current public repository
  readiness snapshot.

## Maintainer Principles

- Keep MKL-Q as an upstream-compatible fork of NVIDIA CUDA-Q.
- Keep the public Python namespace as `cudaq` and the compiler entry point as
  `nvq++`.
- Keep `mklq-cpu` as the stable local correctness target.
- Keep `mklq-metal` experimental unless a reviewed readiness plan changes that
  status.
- Keep the first public version source-only.
- Prefer small, auditable changes over broad rewrites.
- Do not commit build output, raw local benchmark JSON, caches, signing objects,
  tokens, secrets, or machine-specific credentials.

## Routine Health Check

Use this quick check before maintenance work:

```bash
git status --short --branch
git remote -v
git rev-parse --is-shallow-repository
git log --oneline --decorate -5
git ls-files .github | sort
gh repo view wuls968/MKL-Q --json nameWithOwner,isFork,parent,defaultBranchRef,url
gh run list --repo wuls968/MKL-Q --branch main --limit 5
python3 benchmarks/mklq/run_public_healthcheck.py
```

Expected state:

- `main` is clean before collecting clean validation or benchmark evidence.
- `origin` points to the public MKL-Q fork.
- `upstream` points to `https://github.com/NVIDIA/cuda-quantum.git`.
- The repository is not shallow.
- `.github/workflows/` contains only intentionally reviewed MKL-Q workflows.
- The latest pushed commit has a completed `MKL-Q public hygiene` run.
- `main` branch protection matches [`branch-protection.md`](branch-protection.md).
- [`public-readiness.md`](public-readiness.md) matches the current public GitHub
  repository state before describing the repository as ready.
- `run_public_healthcheck.py` passes in default mode before routine public
  metadata or benchmark-tooling pushes.

For a heavier local pre-publication gate, run:

```bash
python3 benchmarks/mklq/run_public_healthcheck.py --full --require-clean
```

Use `--refresh-clean-cpu-benchmark` only when intentionally replacing tracked
clean benchmark evidence from a clean worktree.

## Issue Triage

For each issue, first classify it:

- build or install;
- `mklq-cpu` correctness;
- `mklq-cpu` performance;
- `mklq-metal` experimental behavior;
- upstream CUDA-Q compatibility;
- Python API or target selection;
- `nvq++` or target configuration;
- benchmark tooling or evidence;
- documentation, public metadata, or release policy.

Ask for the minimum useful reproduction data:

- macOS version, Apple Silicon model, core count, and memory;
- source commit or branch;
- build command and install prefix;
- exact target name;
- Python or C++ reproducer;
- whether the one-command correctness gate passes;
- for performance issues, sanitized benchmark summary or enough command detail
  to reproduce locally.

Do not treat a local performance report as cross-machine certification. Route
performance claims through sanitized benchmark summaries under
`benchmarks/mklq/reports/`.

Use [`issue-labels.md`](issue-labels.md) to apply area labels. Remove
`needs-repro` only after the issue has enough information to reproduce or route.

## Pull Request Intake

Before reviewing a pull request, check:

- the PR template is filled out;
- the change keeps the CUDA-Q API boundary unless explicitly scoped to MKL-Q;
- `mklq-metal` is not made default-ready by wording or behavior;
- raw local JSON and generated files are not tracked;
- `testing-matrix.md` points to a gate that actually proves the changed
  behavior;
- `release-policy.md` was followed if the change affects tags, releases,
  wheels, PyPI, installers, or signed artifacts.

For untrusted external contributions, inspect the diff before running scripts or
build commands. Be especially careful with shell scripts, CMake changes,
workflow permissions, install hooks, generated files, and any token or signing
surface.

## Gate Selection

Use the smallest gate that proves the change, then add broader gates when the
change touches shared behavior.

| Change class | Minimum maintainer gate |
| --- | --- |
| Docs-only metadata | `git diff --check`, public hygiene metadata checks, banned-token scan |
| Public workflow or repo hygiene | Local public hygiene checks plus GitHub Actions success on the pushed commit |
| Python target behavior | Focused Python tests plus `testing-matrix.md` review |
| `nvq++` or target config | `test_mklq_nvqpp_smoke.py`, TargetConfig `ctest`, target YAML checks |
| Backend runtime behavior | `cmake --build build-python --target install -j 6` and one-command correctness gate |
| Benchmark tooling | Benchmark helper `py_compile`, harness tests, summary JSON parse |
| Benchmark evidence | Correctness gate first, clean-worktree benchmark run, sanitized summaries only |
| Public health check tooling | `test_mklq_benchmark_harness.py`, `python3 benchmarks/mklq/run_public_healthcheck.py` |
| Upstream sync | `upstream-sync.md` post-merge gates |
| Release artifact proposal | `release-policy.md`, public release checklist, correctness gate, packaging-specific fresh-environment tests |

## Direct-to-main Maintainer Flow

Use direct `main` pushes only for maintainer-owned batches that have already
passed their local gates. The current branch protection policy keeps
administrator enforcement disabled so this recovery path remains available.

```bash
git status --short --branch
git fetch origin main
git switch main
git pull --ff-only origin main
git switch -c codex/<short-topic>

# edit, validate, and review the diff
git diff --check
git status --short --branch
git add <files>
git diff --cached --check
git commit -m "docs: update mklq maintainer docs"

git switch main
git pull --ff-only origin main
git merge --ff-only codex/<short-topic>

# rerun the relevant gates on main
git push origin main
gh run list --repo wuls968/MKL-Q --branch main --workflow "MKL-Q public hygiene" --limit 1
```

Do not force push `main`. If remote `main` moved, inspect the delta before
merging.

## Backend Runtime Changes

For changes under `runtime/nvqir/mklq/`, target config files, C++ backend tests,
or Python backend fixtures:

```bash
cmake --build build-python --target install -j 6
python3 benchmarks/mklq/run_correctness_gate.py \
  --install-prefix "${HOME}/.cudaq-mklq" \
  --build-dir build-python
```

Update [`validation.md`](validation.md) when a validation refresh is meant to be
public evidence. Keep raw wrapper JSON under ignored
`benchmarks/mklq/results/`.

## Benchmark Evidence

Only collect public benchmark evidence after correctness passes and the worktree
is clean:

```bash
git status --short --branch
python3 benchmarks/mklq/run_clean_cpu_benchmark.py \
  --pythonpath "${HOME}/.cudaq-mklq" \
  --stamp YYYY-MM-DD
```

Commit only sanitized summaries under `benchmarks/mklq/reports/` and the
regenerated `docs/mklq/benchmark-evidence.md`. Do not commit raw local payloads.

## Upstream Sync

Use [`upstream-sync.md`](upstream-sync.md) for CUDA-Q upstream merges. Treat
`.github/`, top-level metadata, build scripts, target config installation, and
runtime contracts as high-risk areas.

After the sync, rerun public hygiene and the one-command correctness gate if
runtime, target config, Python bindings, or tests changed.

## Release And Artifact Requests

The default answer is no for release artifacts in the current source-only phase.
Before creating any tag, GitHub Release, wheel, PyPI package, installer,
Homebrew formula, checksum file, or signed artifact, follow
[`release-policy.md`](release-policy.md).

Stop immediately if a release plan would:

- publish raw local benchmark payloads;
- publish generated build products or local signing artifacts;
- imply official NVIDIA CUDA-Q packaging;
- describe `mklq-metal` as default-ready without a passed readiness plan;
- claim binary compatibility or cross-machine performance certification without
  evidence.

## GitHub Actions And Public Metadata

The default public workflow is lightweight. It checks public metadata, tracked
artifact hygiene, sanitized benchmark summary parseability, and benchmark helper
syntax. It does not build CUDA-Q or run Apple Silicon backend correctness tests.

Before adding a heavier workflow, define:

- why local validation is insufficient;
- runner platform and expected cost;
- secrets and permissions;
- failure ownership;
- whether upstream CUDA-Q workflows or bots are being copied.

Do not copy upstream release, bot, Slack, or required-check automation into
MKL-Q without explicit review.

## Security And Sensitive Data

Use [`SECURITY.md`](../../SECURITY.md) for the public security reporting
boundary.

Do not commit:

- tokens, private keys, credentials, or `.env` files;
- local signing or notarization objects;
- private machine paths in public evidence;
- raw benchmark payloads when sanitized summaries are sufficient;
- generated build output or package artifacts.

If sensitive data is found in a commit that reached GitHub, stop normal
maintenance work and rotate the credential or artifact first. Then remove the
exposure through a reviewed remediation plan.

## Recovery

If a pushed commit breaks public hygiene:

```bash
git status --short --branch
gh run list --repo wuls968/MKL-Q --branch main --limit 1
gh run view <run-id> --repo wuls968/MKL-Q --log-failed
git revert <bad-commit>
git diff --check
git push origin main
```

Prefer a revert commit over rewriting public `main`.

If a backend change breaks correctness, keep the failure evidence, revert or fix
on a focused branch, and rerun the one-command correctness gate before pushing.

## Maintenance Cadence

- After every push to `main`, confirm the latest GitHub hygiene run completed
  successfully for that exact commit.
- After backend/runtime batches, refresh `validation.md` only when the gate is
  intended as public evidence.
- After benchmark batches, refresh sanitized summaries and
  `benchmark-evidence.md`.
- After upstream syncs, check this runbook, `upstream-sync.md`,
  `testing-matrix.md`, and `known-limitations.md` for stale assumptions.
- Before any release-style milestone, run `public-release-checklist.md` and
  `release-policy.md` from the top.
