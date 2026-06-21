# MKL-Q Branch Protection

This page documents the intended public `main` branch protection for MKL-Q. The
machine-readable reference is
[`../../.github/branch-protection-main.json`](../../.github/branch-protection-main.json).

The goal is to protect public history and require the lightweight MKL-Q hygiene
check for pull request merges without blocking maintainer emergency recovery.

## Current Policy

The public `main` branch should be protected with:

- required status check: `Source-only repository checks`;
- strict status checks, so pull requests are tested against current `main`;
- administrator enforcement disabled, so the repository owner can still perform
  documented maintainer recovery;
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

## Maintainer Bypass Boundary

Administrator enforcement is intentionally disabled for now. This keeps
maintainer direct-to-main and emergency revert workflows usable while still
documenting the expected public gate.

If MKL-Q gains regular external contributors, revisit this policy before
requiring reviews, requiring pull requests, or enforcing admins.

## Stop Conditions

Do not tighten branch protection if any of these are true:

- the required check name does not match the GitHub Actions job name;
- the latest `MKL-Q public hygiene` run is failing or unknown;
- the repository owner cannot perform documented recovery;
- release or packaging work would need a different protected branch model;
- the policy would block upstream sync recovery without a reviewed alternative.

## Related Docs

- [`maintainer-runbook.md`](maintainer-runbook.md)
- [`public-release-checklist.md`](public-release-checklist.md)
- [`release-policy.md`](release-policy.md)
