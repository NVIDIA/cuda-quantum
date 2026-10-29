# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Discovery of community projects under `docs/sphinx/community/`.

A project is any folder `<root>/<name>/` containing an `index.rst`. Its
metadata is the field list at the top of that page, followed by the page title.
This module has no Sphinx dependency so that it can be shared by the docs build
(`community_projects.py`) and CI, which uses the command line interface:

    python3 community_metadata.py <community_dir> [--json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = ("repository", "maintainers", "license", "example", "summary")
# The optional `install` field holds the `pip install` arguments (package
# specifiers) needed before the example can run, e.g. `cudaq-mock-project==1.2`.
_FIELD = re.compile(r"^:([\w-]+):\s*(.*)$")
_TITLE_UNDERLINE = re.compile(r"^([=\-~^\"#*+])\1{2,}\s*$")


def parse_project(page: Path):
    """Return (title, metadata) from the leading field list and first heading."""
    meta, title = {}, None
    lines = page.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = _FIELD.match(line)
        if m:
            meta[m.group(1)] = m.group(2).strip()
        elif title is None and i + 1 < len(lines) and line.strip() and \
                _TITLE_UNDERLINE.match(lines[i + 1]):
            title = line.strip()
            break
    return title, meta


def problems(title, meta):
    """Return a description of what a project page is missing, or None."""
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    parts = (["a title"] if title is None else
             []) + (["fields: " + ", ".join(missing)] if missing else [])
    return " and ".join(parts) or None


def discover(root: Path):
    """Return (projects, errors) for every `<root>/*/index.rst`."""
    projects, errors = [], []
    for page in sorted(root.glob("*/index.rst")):
        title, meta = parse_project(page)
        issue = problems(title, meta)
        if issue:
            errors.append(f"Community project '{page.parent.name}' is missing "
                          f"{issue}")
            continue
        projects.append({
            **meta,
            "name": page.parent.name,
            "title": title,
            "page": page,
            "example": page.parent / meta["example"],
        })
    return projects, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json",
                        action="store_true",
                        help="print the projects to test as a JSON list")
    args = parser.parse_args()
    projects, errors = discover(args.root)
    for e in errors:
        print(f"::error::{e}", file=sys.stderr)
    for p in projects:
        if not p["example"].is_file():
            print(f"::error::Example '{p['example']}' not found",
                  file=sys.stderr)
            errors.append(p["name"])
    if args.json:
        print(
            json.dumps([{
                "name": p["name"],
                "example": p["example"].as_posix(),
                "install": p.get("install", "")
            } for p in projects]))
    else:
        for p in projects:
            print(f"{p['name']}: {p['example']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
