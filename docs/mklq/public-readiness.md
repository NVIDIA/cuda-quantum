# MKL-Q Public Readiness

This page records the public repository readiness snapshot for MKL-Q. It is a
source-only repository audit, not a release certification, package
certification, Apple Silicon CI replacement, or performance certification.

Snapshot date: 2026-06-22.

## Scope

This readiness snapshot covers:

- public GitHub repository identity;
- upstream fork relationship;
- source-only artifact boundary;
- GitHub workflow and branch protection configuration;
- public documentation coverage;
- current local validation evidence;
- current caveats before any tag, package, or release artifact is created.

It does not certify:

- wheel, PyPI, Homebrew, installer, or binary artifact readiness;
- full CUDA-Q target parity;
- full Metal-native execution;
- cross-machine performance claims;
- hosted Apple Silicon correctness CI.

## Repository Identity

The intended public repository is `wuls968/MKL-Q`:

- URL: <https://github.com/wuls968/MKL-Q>
- Default branch: `main`
- Parent fork: `NVIDIA/cuda-quantum`
- License: Apache-2.0
- Description: `CUDA-Q-compatible Apple Silicon simulator fork with MKL-Q targets`
- Topics: `accelerate`, `apple-silicon`, `cuda-quantum`, `metal`, `mklq`,
  `quantum-computing`

MKL-Q keeps CUDA-Q public API compatibility for the first public source phase:
Python users still import `cudaq`, and C++ users still compile with `nvq++`.

## Git And Remotes

The current public branch model is:

- `origin` points to `https://github.com/wuls968/MKL-Q.git`.
- `upstream` points to `https://github.com/NVIDIA/cuda-quantum.git`.
- The repository is not shallow.
- The public branch keeps CUDA-Q upstream history plus MKL-Q commits on top.
- Sparse checkout may be used locally, but `.github` must be visible before
  public hygiene work.

Use:

```bash
git status --short --branch
git remote -v
git rev-parse --is-shallow-repository
git sparse-checkout list
git log --oneline --decorate -5
```

Expected result: `main` is clean before collecting clean evidence, the repo is
not shallow, and `origin/main` matches the intended public commit.

## Public Documentation

The public MKL-Q support boundary is documented in:

- [`architecture.md`](architecture.md)
- [`validation.md`](validation.md)
- [`testing-matrix.md`](testing-matrix.md)
- [`benchmark-evidence.md`](benchmark-evidence.md)
- [`known-limitations.md`](known-limitations.md)
- [`roadmap.md`](roadmap.md)
- [`upstream-sync.md`](upstream-sync.md)
- [`release-policy.md`](release-policy.md)
- [`public-release-checklist.md`](public-release-checklist.md)
- [`developer-workflow.md`](developer-workflow.md)
- [`maintainer-runbook.md`](maintainer-runbook.md)
- [`issue-labels.md`](issue-labels.md)
- [`branch-protection.md`](branch-protection.md)

These documents intentionally describe the current source-only state. They must
not imply that MKL-Q publishes wheels, PyPI packages, installers, release tags,
or GitHub Releases.

## GitHub Configuration

The public GitHub configuration is intentionally lightweight:

- `.github/workflows/mklq-public-hygiene.yml` is the only tracked workflow.
- `.github/ISSUE_TEMPLATE/bug_report.yaml` and
  `.github/ISSUE_TEMPLATE/feature_request.yaml` are the only issue templates.
- `.github/pull_request_template.md` records compatibility, validation,
  benchmark evidence, and public hygiene checks.
- `.github/labels.yml` records the public triage label taxonomy.
- `.github/branch-protection-main.json` records the intended `main` protection
  API payload.

The lightweight workflow checks source-only repository hygiene, public metadata,
tracked benchmark summary parseability, bounded Metal runtime counter probe
parseability, and benchmark helper syntax. It does not build CUDA-Q or run
Apple Silicon backend correctness tests.

## Branch Protection

The public `main` branch is intended to be protected with:

- required status check: `Source-only repository checks`;
- strict status checks enabled;
- force pushes disabled;
- branch deletion disabled;
- administrator enforcement disabled for documented maintainer recovery;
- no required pull-request review policy yet;
- no branch push restrictions yet.

The branch protection policy is documented in
[`branch-protection.md`](branch-protection.md), and the machine-readable API
payload is `.github/branch-protection-main.json`.

## Validation Snapshot

The latest public local validation evidence is recorded in
[`validation.md`](validation.md):

- latest validation refresh date: 2026-06-22;
- source commit used for the latest clean-worktree runtime validation gate:
  `997ec1f3c022d854d644257bc7dca990a17bd243`;
- install-prefix build: passed;
- full public healthcheck: passed with 15/15 steps passed;
- one-command correctness gate: passed with 4/4 steps passed, including
  `metal_runtime_counter_probe`;
- public example smoke gate: passed with 30/30 steps passed;
- benchmark harness tests: `67 passed`;
- standalone install-prefix Python subset: `35 passed`;
- `python_target_smoke`: `56 passed`;
- `nvqpp_smoke`: `2 passed`;
- `target_config_ctest`: `63/63 passed`.
- clean CPU benchmark gate: passed with 18 q20 `qpp-cpu`/`mklq-cpu` rows,
  including `cz-state`, `qft-like-state`, and `seeded-clifford-state`, with
  18 rows reporting `status == "ok"` against
  `34f4b260d1c657ad626c526eed4e6b9d3a441be4`.

This evidence is local Apple Silicon evidence. It is useful for source bootstrap
confidence, but it is not hosted CI, release certification, or cross-machine
performance certification.

The current public healthcheck also includes the static
`check_metal_evidence.py` guard for tracked `mklq-metal` summaries. That guard
checks local tuning provenance, ignored raw payload paths, successful Metal
rows, and wording that keeps the experimental mixed-path/host boundary clear.

## No Tags Or Releases

The current public state is source-only:

- no public version tags;
- no GitHub Releases;
- no PyPI packages;
- no wheels;
- no installers;
- no Homebrew formula;
- no signed artifacts;
- no raw local benchmark payloads tracked.

Do not change this boundary without updating [`release-policy.md`](release-policy.md)
and running the release gates described there.

## Benchmark Evidence Boundary

Sanitized benchmark summaries may be tracked under `benchmarks/mklq/reports/`.
Raw local benchmark JSON under `benchmarks/mklq/results/` is intentionally
ignored.

Benchmark summaries must be interpreted through their `evidence_kind`,
`machine`, and `interpretation` fields. They are not cross-machine performance
certification.

## Current Caveats

- `mklq-cpu` is the stable local Apple Silicon target.
- `mklq-metal` is experimental and must not be described as default-ready.
- Public GitHub Actions currently run source hygiene only.
- Backend correctness still depends on local Apple Silicon validation.
- No package manager or binary artifact support is published.
- Upstream CUDA-Q syncs must follow [`upstream-sync.md`](upstream-sync.md).

## Readiness Commands

Use these commands for the public repository readiness audit:

```bash
python3 benchmarks/mklq/run_public_readiness_audit.py
git status --short --branch
git rev-parse --is-shallow-repository
git sparse-checkout list
git ls-files .github | sort
git ls-files docs/mklq | sort
git ls-remote --tags origin 'refs/tags/*'
gh repo view wuls968/MKL-Q \
  --json nameWithOwner,isFork,parent,defaultBranchRef,url,description,repositoryTopics,licenseInfo
gh api repos/wuls968/MKL-Q/branches/main --jq '{name,protected,commit:.commit.sha}'
gh run list --repo wuls968/MKL-Q --branch main --workflow 'MKL-Q public hygiene' --limit 1
gh release list --repo wuls968/MKL-Q --limit 20
```

Expected result:

- the repository remains a fork of `NVIDIA/cuda-quantum`;
- `main` is the default branch;
- `main` is protected;
- the latest pushed commit has a successful `MKL-Q public hygiene` run;
- no release tags or GitHub Releases exist in the current source-only phase;
- only intentional MKL-Q public docs, issue templates, branch protection config,
  and the lightweight workflow are tracked.

## Stop Conditions

Do not describe the public repository as ready if any of these are true:

- the worktree is dirty before collecting clean evidence;
- `origin/main` does not match the intended public commit;
- the latest GitHub Actions result is failing, missing, or still pending;
- branch protection is missing or no longer requires `Source-only repository
  checks`;
- raw benchmark JSON, build output, caches, `.DS_Store`, signing artifacts,
  tokens, secrets, or `docs/superpowers/` are tracked;
- release tags, GitHub Releases, wheels, installers, or package artifacts were
  created without a reviewed release plan;
- `mklq-metal` is presented as default-ready or full Metal-native without a
  separate readiness plan.
