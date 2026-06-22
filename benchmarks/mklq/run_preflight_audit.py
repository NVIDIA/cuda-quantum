#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Run a local MKL-Q preflight audit before publishing or opening PRs."""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mklq-preflight-audit-v1"
DEFAULT_REPO = "wuls968/MKL-Q"
EXPECTED_ORIGIN = "https://github.com/wuls968/MKL-Q.git"
EXPECTED_UPSTREAM = "https://github.com/NVIDIA/cuda-quantum.git"
REQUIRED_STATUS_CHECK = "Source-only repository checks"
LOCK_NAMES = ("index.lock", "HEAD.lock", "config.lock", "packed-refs.lock",
              "shallow.lock")
LOCK_RECHECK_DELAY_SECONDS = 0.25
TRACKED_ARTIFACT_PATTERN = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache)(/|$)|"
    r"\.pyc$|\.DS_Store$|^build(-python)?/|"
    r"^benchmarks/mklq/results/|^docs/superpowers/|"
    r"^(dist|wheelhouse)/|\.(whl|dmg|pkg|zip)$|\.tar\.gz$")


@dataclass(frozen=True)
class PreflightConfig:
    repo_root: Path
    repo: str
    output: Path
    require_clean: bool
    check_github: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_default(stamp: str) -> Path:
    return Path("benchmarks/mklq/results") / f"preflight-audit-{stamp}.json"


def command_output(cwd: Path, command: list[str]) -> str:
    return subprocess.check_output(command,
                                   cwd=cwd,
                                   text=True,
                                   stderr=subprocess.STDOUT).rstrip("\n")


def passed(name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed",
        "details": details or {},
    }


def failed(name: str, message: str,
           details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": "failed",
        "message": message,
    }
    if details:
        payload["details"] = details
    return payload


def load_json(text: str, fallback: Any) -> Any:
    if not text.strip():
        return fallback
    return json.loads(text)


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed_count = sum(1 for check in checks if check["status"] == "failed")
    passed_count = len(checks) - failed_count
    return {
        "status": "failed" if failed_count else "passed",
        "passed": passed_count,
        "failed": failed_count,
    }


def git_dir(config: PreflightConfig) -> Path:
    raw = command_output(config.repo_root, ["git", "rev-parse", "--git-dir"])
    path = Path(raw)
    return path if path.is_absolute() else config.repo_root / path


def has_fetch_remote(remotes: list[str], name: str, url: str) -> bool:
    prefix = f"{name}\t{url} (fetch)"
    return any(line.startswith(prefix) for line in remotes)


def git_lock_files(directory: Path, repo_root: Path) -> list[dict[str, Any]]:
    locks = []
    for name in LOCK_NAMES:
        path = directory / name
        if path.exists():
            locks.append({
                "path": path.relative_to(repo_root).as_posix(),
                "size_bytes": path.stat().st_size,
            })
    return locks


def check_git_worktree(config: PreflightConfig) -> dict[str, Any]:
    status = command_output(config.repo_root,
                            ["git", "status", "--short", "--branch"])
    shallow = command_output(config.repo_root,
                             ["git", "rev-parse", "--is-shallow-repository"])
    remotes = command_output(config.repo_root, ["git", "remote", "-v"])

    failures: list[str] = []
    dirty = [line for line in status.splitlines() if not line.startswith("##")]
    if config.require_clean and dirty:
        failures.append("working tree is dirty")
    if shallow.strip() != "false":
        failures.append("repository is shallow")
    remote_lines = remotes.splitlines()
    if not has_fetch_remote(remote_lines, "origin", EXPECTED_ORIGIN):
        failures.append("origin remote is missing or unexpected")
    if not has_fetch_remote(remote_lines, "upstream", EXPECTED_UPSTREAM):
        failures.append("upstream remote is missing")

    details = {
        "status_short_branch": status.splitlines(),
        "is_shallow": shallow.strip(),
        "remotes": remote_lines,
        "expected_origin": EXPECTED_ORIGIN,
        "expected_upstream": EXPECTED_UPSTREAM,
        "require_clean": config.require_clean,
    }
    return failed("git_worktree", "; ".join(failures),
                  details) if failures else passed("git_worktree", details)


def check_git_locks(config: PreflightConfig) -> dict[str, Any]:
    directory = git_dir(config)
    locks = git_lock_files(directory, config.repo_root)
    rechecks = 0
    if locks:
        time.sleep(LOCK_RECHECK_DELAY_SECONDS)
        rechecks += 1
        locks = git_lock_files(directory, config.repo_root)

    details = {
        "git_dir": directory.as_posix(),
        "lock_files": locks,
        "rechecks": rechecks,
        "recheck_delay_seconds": LOCK_RECHECK_DELAY_SECONDS,
    }
    return failed("git_locks", "git lock files are present",
                  details) if locks else passed("git_locks", details)


def check_tracked_artifacts(config: PreflightConfig) -> dict[str, Any]:
    tracked = command_output(config.repo_root, ["git", "ls-files"]).splitlines()
    bad = [path for path in tracked if TRACKED_ARTIFACT_PATTERN.search(path)]
    details = {"tracked_file_count": len(tracked), "bad_paths": bad}
    return failed("tracked_artifacts",
                  "generated or local artifacts are tracked",
                  details) if bad else passed("tracked_artifacts", details)


def check_ignored_local_artifacts(config: PreflightConfig) -> dict[str, Any]:
    status = command_output(config.repo_root,
                            ["git", "status", "--ignored", "--short"])
    ignored = []
    for line in status.splitlines():
        if line.startswith("!! "):
            ignored.append(line[3:])
    return passed("ignored_local_artifacts", {
        "ignored_count": len(ignored),
        "ignored_paths": ignored[:80],
        "truncated": len(ignored) > 80,
    })


def check_branch_protection(config: PreflightConfig) -> dict[str, Any]:
    if not config.check_github:
        return passed("branch_protection", {"skipped": True})

    try:
        branch = load_json(
            command_output(config.repo_root, [
                "gh",
                "api",
                f"repos/{config.repo}/branches/main",
            ]), {})
        protection = load_json(
            command_output(config.repo_root, [
                "gh",
                "api",
                f"repos/{config.repo}/branches/main/protection",
            ]), {})
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return failed("branch_protection", "GitHub branch protection check failed",
                      {"error": str(exc), "repo": config.repo})

    required = protection.get("required_status_checks") or {}
    contexts = set(required.get("contexts") or [])
    checks = {check.get("context") for check in required.get("checks") or []}
    failures: list[str] = []
    if branch.get("protected") is not True:
        failures.append("branch is not protected")
    if REQUIRED_STATUS_CHECK not in contexts and REQUIRED_STATUS_CHECK not in checks:
        failures.append("required status check is missing")
    if required.get("strict") is not True:
        failures.append("required status checks are not strict")
    if (protection.get("allow_force_pushes") or {}).get("enabled") is not False:
        failures.append("force pushes are not disabled")
    if (protection.get("allow_deletions") or {}).get("enabled") is not False:
        failures.append("branch deletion is not disabled")
    if (protection.get("enforce_admins") or {}).get("enabled") is not True:
        failures.append("administrator enforcement is not enabled")

    details = {
        "repo": config.repo,
        "branch": branch,
        "protection": protection,
        "required_status_check": REQUIRED_STATUS_CHECK,
    }
    return failed("branch_protection", "; ".join(failures),
                  details) if failures else passed("branch_protection", details)


def build_report(config: PreflightConfig) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    checks = [
        check_git_worktree(config),
        check_git_locks(config),
        check_tracked_artifacts(config),
        check_ignored_local_artifacts(config),
        check_branch_protection(config),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "repo": config.repo,
        "repo_root": config.repo_root.as_posix(),
        "summary": summarize(checks),
        "checks": checks,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        default=DEFAULT_REPO,
                        help=f"GitHub repository to audit. Default: {DEFAULT_REPO}")
    parser.add_argument("--output",
                        type=Path,
                        default=None,
                        help="JSON report path. Defaults under benchmarks/mklq/results.")
    parser.add_argument("--require-clean",
                        action="store_true",
                        help="Fail when the working tree has uncommitted changes.")
    parser.add_argument("--skip-github",
                        action="store_true",
                        help="Skip live GitHub branch protection checks.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = repo_root()
    output = args.output or output_default(stamp)
    output = output if output.is_absolute() else root / output
    config = PreflightConfig(repo_root=root,
                             repo=args.repo,
                             output=output,
                             require_clean=args.require_clean,
                             check_github=not args.skip_github)
    report = build_report(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    summary = report["summary"]
    print(f"MKL-Q preflight audit: {summary['status']}")
    print(f"Report: {output}")
    for check in report["checks"]:
        if check["status"] == "failed":
            print(f"- {check['name']}: {check['message']}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
