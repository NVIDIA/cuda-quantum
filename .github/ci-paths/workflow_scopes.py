#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Generate not-linux-ci.txt and not-macos-ci.txt from the workflow graph.

Each file lists the workflows and actions that ci.yml or ci_macos.yml never
reaches through `uses: ./.github/...`.

  python3 .github/ci-paths/workflow_scopes.py --write   # regenerate
  python3 .github/ci-paths/workflow_scopes.py --check   # fail if out of date
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Output file -> (entry workflow, platform name, extra entries).
SCOPES = {
    "not-linux-ci.txt": ("ci.yml", "Linux", []),
    "not-macos-ci.txt": ("ci_macos.yml", "macOS", [
        "# Dockerfiles and the Linux release image assets.", "docker/build/",
        "docker/release/"
    ]),
}


def uses(path):
    """Local workflows and actions that `path` calls."""
    out = set()
    with open(os.path.join(ROOT, path)) as f:
        text = f.read()
    for m in re.finditer(
            r"uses:\s*\./(\.github/(?:workflows|actions)/[A-Za-z0-9_.\-]+)",
            text):
        target = m.group(1)
        out.add(
            target +
            "/action.yml" if target.startswith(".github/actions/") else target)
    return out


def reachable(entry):
    seen, stack = set(), [entry]
    while stack:
        path = stack.pop()
        if path in seen or not os.path.exists(os.path.join(ROOT, path)):
            continue
        seen.add(path)
        stack.extend(uses(path))
    return seen


def render(entry, platform, extra):
    reached = reachable(f".github/workflows/{entry}")
    candidates = sorted(
        os.path.relpath(p, ROOT)
        for p in glob.glob(os.path.join(ROOT, ".github/workflows/*.yml")) +
        glob.glob(os.path.join(ROOT, ".github/actions/*/action.yml")))
    lines = [
        f"# Workflows and actions the {platform} CI never calls (from {entry}).",
        "# Generated: python3 .github/ci-paths/workflow_scopes.py --write",
    ]
    for path in candidates:
        if path in reached:
            continue
        lines.append(
            os.path.dirname(path) +
            "/" if path.endswith("/action.yml") else path)
    return "\n".join(lines + extra) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stale = []
    for name, (entry, platform, extra) in SCOPES.items():
        target = os.path.join(ROOT, ".github/ci-paths", name)
        content = render(entry, platform, extra)
        if args.write:
            with open(target, "w") as f:
                f.write(content)
        elif not os.path.exists(target) or open(target).read() != content:
            stale.append(name)
    if stale:
        print("Out of date, run with --write: " + ", ".join(stale),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
