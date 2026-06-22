#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Audit MKL-Q public GitHub readiness and write bounded JSON evidence."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mklq-public-readiness-audit-v1"
DEFAULT_REPO = "wuls968/MKL-Q"
DEFAULT_WORKFLOW = "MKL-Q public hygiene"
REQUIRED_STATUS_CHECK = "Source-only repository checks"
EXPECTED_DESCRIPTION = (
    "CUDA-Q-compatible Apple Silicon simulator fork with MKL-Q targets")
EXPECTED_TOPICS = {
    "accelerate",
    "apple-silicon",
    "cuda-quantum",
    "metal",
    "mklq",
    "quantum-computing",
}
TRACKED_ARTIFACT_PATTERN = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache)(/|$)|"
    r"\.pyc$|\.DS_Store$|^build(-python)?/|"
    r"^benchmarks/mklq/results/|^docs/superpowers/|"
    r"^(dist|wheelhouse)/|\.(whl|dmg|pkg|zip)$|\.tar\.gz$")


@dataclass(frozen=True)
class AuditConfig:
    repo_root: Path
    repo: str
    workflow: str
    output: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_default(stamp: str) -> Path:
    return Path("benchmarks/mklq/results") / (
        f"public-readiness-audit-{stamp}.json")


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


def remote_head_sha(output: str) -> str:
    if not output.strip():
        return ""
    return output.split()[0]


def check_local_git(config: AuditConfig) -> dict[str, Any]:
    status = command_output(config.repo_root,
                            ["git", "status", "--short", "--branch"])
    shallow = command_output(config.repo_root,
                             ["git", "rev-parse", "--is-shallow-repository"])
    head = command_output(config.repo_root, ["git", "rev-parse", "HEAD"])
    remote = command_output(config.repo_root,
                            ["git", "ls-remote", "origin", "refs/heads/main"])
    remote_sha = remote_head_sha(remote)
    failures: list[str] = []
    dirty = [line for line in status.splitlines() if not line.startswith("##")]
    if dirty:
        failures.append("working tree is dirty")
    if shallow.strip() != "false":
        failures.append("repository is shallow")
    if head != remote_sha:
        failures.append("local HEAD does not match origin/main")
    details = {
        "status_short_branch": status.splitlines(),
        "is_shallow": shallow.strip(),
        "head": head,
        "origin_main": remote_sha,
    }
    return failed("local_git", "; ".join(failures),
                  details) if failures else passed("local_git", details)


def check_tracked_artifacts(config: AuditConfig) -> dict[str, Any]:
    tracked = command_output(config.repo_root, ["git", "ls-files"]).splitlines()
    bad = [path for path in tracked if TRACKED_ARTIFACT_PATTERN.search(path)]
    details = {"tracked_file_count": len(tracked), "bad_paths": bad}
    return failed("tracked_artifacts", "generated or local artifacts are tracked",
                  details) if bad else passed("tracked_artifacts", details)


def check_workflows(config: AuditConfig) -> dict[str, Any]:
    workflows = command_output(config.repo_root,
                               ["git", "ls-files",
                                ".github/workflows"]).splitlines()
    expected = [".github/workflows/mklq-public-hygiene.yml"]
    details = {"workflows": workflows}
    return failed("github_workflows", "unexpected workflow set",
                  details) if workflows != expected else passed(
                      "github_workflows", details)


def check_repository(config: AuditConfig) -> dict[str, Any]:
    payload = load_json(
        command_output(config.repo_root, [
            "gh",
            "repo",
            "view",
            config.repo,
            "--json",
            "nameWithOwner,isFork,parent,defaultBranchRef,url,description,"
            "repositoryTopics,licenseInfo,visibility",
        ]), {})
    failures: list[str] = []
    parent = payload.get("parent") or {}
    parent_owner = parent.get("owner") or {}
    topics = {item.get("name") for item in payload.get("repositoryTopics", [])}
    if payload.get("nameWithOwner") != config.repo:
        failures.append("unexpected repository")
    if payload.get("isFork") is not True:
        failures.append("repository is not a fork")
    if parent.get("name") != "cuda-quantum" or parent_owner.get("login") != "NVIDIA":
        failures.append("parent is not NVIDIA/cuda-quantum")
    if (payload.get("defaultBranchRef") or {}).get("name") != "main":
        failures.append("default branch is not main")
    if payload.get("visibility") != "PUBLIC":
        failures.append("repository is not public")
    if (payload.get("licenseInfo") or {}).get("key") != "apache-2.0":
        failures.append("license is not Apache-2.0")
    if payload.get("description") != EXPECTED_DESCRIPTION:
        failures.append("description does not match MKL-Q public metadata")
    missing_topics = sorted(EXPECTED_TOPICS - topics)
    if missing_topics:
        failures.append("expected topics are missing")
    details = dict(payload)
    details["missing_topics"] = missing_topics
    return failed("github_repository", "; ".join(failures),
                  details) if failures else passed("github_repository", details)


def check_branch_protection(config: AuditConfig) -> dict[str, Any]:
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
    details = {"branch": branch, "protection": protection}
    return failed("branch_protection", "; ".join(failures),
                  details) if failures else passed("branch_protection", details)


def check_latest_public_hygiene(config: AuditConfig) -> dict[str, Any]:
    runs = load_json(
        command_output(config.repo_root, [
            "gh",
            "run",
            "list",
            "--repo",
            config.repo,
            "--branch",
            "main",
            "--workflow",
            config.workflow,
            "--limit",
            "1",
            "--json",
            "status,conclusion,headSha,url,name,event,createdAt",
        ]), [])
    head = command_output(config.repo_root, ["git", "rev-parse", "HEAD"])
    run = runs[0] if runs else {}
    failures: list[str] = []
    if not run:
        failures.append("no public hygiene run found")
    if run.get("status") != "completed":
        failures.append("latest public hygiene run is not completed")
    if run.get("conclusion") != "success":
        failures.append("latest public hygiene run did not succeed")
    if run.get("headSha") != head:
        failures.append("latest public hygiene run is not for local HEAD")
    return failed("latest_public_hygiene", "; ".join(failures),
                  run) if failures else passed("latest_public_hygiene", run)


def check_no_releases(config: AuditConfig) -> dict[str, Any]:
    tags = command_output(config.repo_root,
                          ["git", "ls-remote", "--tags", "origin", "refs/tags/*"])
    releases = command_output(config.repo_root, [
        "gh",
        "release",
        "list",
        "--repo",
        config.repo,
        "--limit",
        "20",
    ])
    failures: list[str] = []
    if tags.strip():
        failures.append("release tags exist")
    if releases.strip():
        failures.append("GitHub releases exist")
    details = {
        "remote_tags": tags.splitlines(),
        "releases": releases.splitlines(),
    }
    return failed("no_tags_or_releases", "; ".join(failures),
                  details) if failures else passed("no_tags_or_releases", details)


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed_count = sum(1 for check in checks if check["status"] == "passed")
    failed_count = sum(1 for check in checks if check["status"] == "failed")
    return {
        "status": "passed" if failed_count == 0 else "failed",
        "passed": passed_count,
        "failed": failed_count,
    }


def build_report(config: AuditConfig) -> dict[str, Any]:
    checks = [
        check_local_git(config),
        check_tracked_artifacts(config),
        check_workflows(config),
        check_repository(config),
        check_branch_protection(config),
        check_latest_public_hygiene(config),
        check_no_releases(config),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "repo_root": config.repo_root.as_posix(),
            "repo": config.repo,
            "workflow": config.workflow,
            "output": config.output.as_posix(),
        },
        "checks": checks,
        "summary": summarize(checks),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MKL-Q public GitHub readiness.")
    parser.add_argument("--repo",
                        default=DEFAULT_REPO,
                        help="GitHub repository, owner/name.")
    parser.add_argument("--workflow",
                        default=DEFAULT_WORKFLOW,
                        help="GitHub Actions workflow name to check.")
    parser.add_argument("--output",
                        type=Path,
                        help="JSON output path. Defaults under ignored results/.")
    parser.add_argument("--stamp",
                        default=date.today().isoformat(),
                        help="Date or label for the default output filename.")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> AuditConfig:
    root = repo_root()
    output = args.output or output_default(args.stamp)
    if not output.is_absolute():
        output = root / output
    return AuditConfig(repo_root=root,
                       repo=args.repo,
                       workflow=args.workflow,
                       output=output)


def main(argv: list[str]) -> int:
    config = make_config(parse_args(argv))
    report = build_report(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, indent=2, sort_keys=True) +
                             "\n",
                             encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
