#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Reads apt package lists (one package name per line), optionally fetches
# Debian source metadata for each package, and writes a NOTICE-style attribution
# file with references to license and copyright (in-image path and upstream).

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import OrderedDict

DEBIAN_SRC_API = "https://sources.debian.org/api/src/{name}/"
DEBIAN_LICENSE_PAGE = "https://sources.debian.org/copyright/license/{name}/{version}/"
# First 4-digit year or year range in copyright context (e.g. 1994, 1996-2022)
COPYRIGHT_YEAR_RE = re.compile(r"\b(19|20)\d{2}(?:-(19|20)\d{2})?\b")

HEADER = """Apt packages - third-party attribution
This file lists license and copyright information for apt packages included
in the CUDA-Q package sources image. Full text is in /usr/share/doc/<pkg>/copyright
in the image; links below point to Debian or Ubuntu package pages.

"""


def fetch_copyright_year(name: str, version: str) -> str | None:
    """Fetch Debian license page and return first copyright year or year range, or None."""
    url = DEBIAN_LICENSE_PAGE.format(
        name=urllib.parse.quote(name, safe=""),
        version=urllib.parse.quote(version, safe=""),
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    # Restrict to content before "Licenses" section to avoid matching license text
    if "## Licenses" in body:
        body = body.split("## Licenses")[0]
    m = COPYRIGHT_YEAR_RE.search(body)
    if m:
        return m.group(0)
    return None


def fetch_debian_versions(name: str) -> str | None:
    """Fetch Debian source package versions. Returns one version string or None."""
    url = DEBIAN_SRC_API.format(name=urllib.parse.quote(name, safe=""))
    try:
        req = urllib.request.Request(url,
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except Exception:
        return None
    versions = data.get("versions") or []
    # Prefer a stable suite (bookworm, trixie) then take first
    for v in versions:
        if v.get("suites") and any(s in ("bookworm", "trixie", "sid")
                                   for s in v.get("suites", [])):
            return v.get("version")
    return versions[0].get("version") if versions else None


def format_entry(
    name: str,
    version: str | None,
    copyright_year: str | None = None,
) -> str:
    """Format a single NOTICE-style entry for one package."""
    doc_path = f"/usr/share/doc/{name}/copyright"
    lines = [
        f"{name} - See {doc_path} in the image.",
        "",
    ]
    if version:
        license_url = (
            f"https://sources.debian.org/copyright/license/{name}/{version}/")
        lines.append(f"<{license_url}>")
    else:
        lines.append("<https://packages.ubuntu.com/>")
    lines.append("")
    if copyright_year:
        lines.append(f"Copyright (c) {copyright_year}.")
        lines.append("")
    lines.append(
        "----------------------------------------------------------------")
    lines.append("")
    return "\n".join(lines)


def load_package_names(paths: list[str]) -> OrderedDict[str, None]:
    """Load unique package names from one or more apt list files (one name per line)."""
    seen: OrderedDict[str, None] = OrderedDict()
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    name = line.strip()
                    if not name or name.startswith("#"):
                        continue
                    seen[name] = None
        except FileNotFoundError:
            continue
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="Generate NOTICE-style attribution file for apt packages.")
    parser.add_argument(
        "list_files",
        nargs="+",
        metavar="apt_list.txt",
        help="One or more files with one package name per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="NOTICE_APT",
        help="Output file path (default: NOTICE_APT)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the header comment",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch Debian API; use Ubuntu link only",
    )
    parser.add_argument(
        "--no-copyright-year",
        action="store_true",
        help="Do not fetch copyright year from Debian license page",
    )
    args = parser.parse_args()

    packages = load_package_names(args.list_files)
    if not packages:
        sys.stderr.write("No packages found in list files.\n")
        sys.exit(1)

    out_lines = [] if args.no_header else [HEADER]

    for name in packages:
        version = None
        copyright_year = None
        if not args.no_fetch:
            version = fetch_debian_versions(name)
            if version and not args.no_copyright_year:
                copyright_year = fetch_copyright_year(name, version)
        out_lines.append(format_entry(name, version, copyright_year))

    with open(args.output, "w") as f:
        f.write("".join(out_lines))

    print(
        f"Wrote {args.output} with {len(packages)} packages.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
