# MKL-Q Issue Labels

This page defines the MKL-Q-specific GitHub labels used for issue and pull
request triage. The source-of-truth label list is tracked in
[`../../.github/labels.yml`](../../.github/labels.yml).

Maintainers may add these labels to the public GitHub repository with GitHub CLI
or API. Do not delete GitHub's default labels unless that cleanup is reviewed
separately.

## Label Taxonomy

| Label | Use For | Remove When |
| --- | --- | --- |
| `backend:cpu` | `mklq-cpu` correctness, runtime behavior, sampling, state import/export, or CPU performance-sensitive paths. | The issue is no longer specific to `mklq-cpu`. |
| `backend:metal` | Experimental `mklq-metal`, resident Metal paths, mixed Metal/CPU fallback, Metal measurement/reset, or Metal precision behavior. | The issue is no longer specific to `mklq-metal`. |
| `build` | CMake, source build, install prefix, `nvq++` discovery, target YAML installation, or toolchain problems. | The root cause is reclassified as backend/runtime/docs. |
| `performance` | Benchmarks, regressions, timing variance, benchmark harness changes, or sanitized performance evidence. | The issue is purely correctness or docs. |
| `docs` | README, `docs/mklq`, issue templates, public metadata, runbooks, or examples. | The issue is reclassified as code, build, or release policy. |
| `upstream-sync` | Pulling, merging, or reviewing `NVIDIA/cuda-quantum` upstream changes. | The sync-specific work is done or reclassified. |
| `release-policy` | Source-only policy, tags, GitHub Releases, wheels, PyPI, installers, signing, checksums, or packaging boundaries. | The issue is not about public release or artifact policy. |
| `needs-repro` | Reports that need a minimal reproducer, local environment details, exact target, command output, or correctness-gate result. | The report has enough information for a maintainer to reproduce or route it. |

## Default Template Labels

- Bug reports start with `bug` and `needs-repro`.
- Feature requests start with `enhancement`.

After reading the issue, add one or more area labels such as `backend:cpu`,
`backend:metal`, `build`, `performance`, `docs`, `upstream-sync`, or
`release-policy`.

## Triage Rules

- Use `needs-repro` when a report lacks the source commit, platform, target,
  command, or output needed to reproduce locally.
- Prefer one primary area label, then add secondary labels only when the issue
  genuinely spans multiple surfaces.
- Do not use `performance` unless the report includes a timing claim, benchmark
  command, regression claim, or request for optimization.
- Use `release-policy` for any request to publish artifacts, create tags,
  change source-only wording, or prepare packaging.
- Use `upstream-sync` for CUDA-Q upstream merges even when conflicts touch
  backend files.

## Label Maintenance

Before changing the label set:

```bash
git status --short --branch
gh label list --repo wuls968/MKL-Q --limit 100
```

Update both:

- `.github/labels.yml`
- this page

Then apply the change on GitHub with `gh label create` or `gh label edit`.
Do not delete default GitHub labels as part of ordinary MKL-Q maintenance.
