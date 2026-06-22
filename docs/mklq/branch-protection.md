# MKL-Q Branch Protection

This page documents the intended public `main` branch protection for MKL-Q. The
machine-readable reference is
[`../../.github/branch-protection-main.json`](../../.github/branch-protection-main.json).

The goal is to protect public history and require the lightweight MKL-Q hygiene
check before `main` advances, including maintainer-owned changes.

## Current Policy

The public `main` branch should be protected with:

- required status check: `Source-only repository checks`;
- strict status checks, so pull requests are tested against current `main`;
- administrator enforcement enabled, so maintainers use the same required
  status check path as external contributors;
- force pushes disabled;
- branch deletion disabled;
- pull request reviews not required by policy yet;
- branch restrictions not required by policy yet.

This is a source-only repository hygiene gate. It does not build CUDA-Q, run the
Apple Silicon correctness gate, certify Metal behavior, or validate release
artifacts.

## Applying The Policy

Use the GitHub API so the configuration stays explicit:

```bash
gh api -X PUT repos/wuls968/MKL-Q/branches/main/protection \
  --input .github/branch-protection-main.json
```

Then verify:

```bash
gh api repos/wuls968/MKL-Q/branches/main --jq '{name, protected}'
gh api repos/wuls968/MKL-Q/branches/main/protection --jq '{
  required_status_checks,
  enforce_admins,
  allow_force_pushes,
  allow_deletions
}'
```

## Maintainer Recovery Boundary

Administrator enforcement is intentionally enabled. Routine work should land
through a branch or pull request after the `Source-only repository checks` job
passes for the proposed commit.

If a pushed commit or branch protection change blocks recovery, use the GitHub
settings/API to make a documented temporary protection change, land the revert
or recovery commit, verify the hygiene workflow, and restore this policy in the
same maintenance window. Prefer a revert commit over rewriting public `main`.

## Stop Conditions

Do not tighten branch protection if any of these are true:

- the required check name does not match the GitHub Actions job name;
- the latest `MKL-Q public hygiene` run is failing or unknown;
- the repository owner cannot perform documented recovery through GitHub
  settings/API;
- release or packaging work would need a different protected branch model;
- the policy would block upstream sync recovery without a reviewed alternative.

## Related Docs

- [`maintainer-runbook.md`](maintainer-runbook.md)
- [`public-readiness.md`](public-readiness.md)
- [`public-release-checklist.md`](public-release-checklist.md)
- [`release-policy.md`](release-policy.md)
