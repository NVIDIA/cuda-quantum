#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Reads pip package lists (name==version per line), fetches PyPI metadata for
# each package, and writes a NOTICE-style attribution file with license and
# copyright year(s).

import argparse
import json
import re
import sys
import urllib.request
from collections import OrderedDict

PYPI_URL = "https://pypi.org/pypi/{name}/{version}/json"
HEADER = """Pip packages - third-party attribution
This file lists license and copyright information for pip packages included
in the CUDA-Q package sources image. Data is derived from PyPI metadata.

"""


def fetch_pypi_metadata(name: str, version: str) -> dict | None:
    """Fetch PyPI JSON for the given package and version. Returns None on failure."""
    url = PYPI_URL.format(name=name, version=version)
    try:
        req = urllib.request.Request(url,
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception:
        return None


def get_license_from_classifiers(classifiers: list[str] | None) -> str | None:
    """Extract license from Trove classifiers (e.g. 'License :: OSI Approved :: MIT')."""
    if not classifiers:
        return None
    for c in classifiers:
        if c.startswith("License ::"):
            # Return the last part, e.g. "MIT License" from "License :: OSI Approved :: MIT License"
            parts = c.split(" :: ")
            return parts[-1] if len(parts) >= 2 else c
    return None


def get_copyright_year(data: dict) -> str | None:
    """Get copyright year from release upload_time (this version)."""
    urls = data.get("urls") or []
    for u in urls:
        t = u.get("upload_time") or u.get("upload_time_iso_8601")
        if t:
            # "2025-03-05" or "2025-03-05T20:05:00"
            year = t[:4]
            if year.isdigit():
                return year
    return None


def get_author(data: dict) -> str | None:
    """Get author or maintainer for copyright line."""
    info = data.get("info") or {}
    author = (info.get("author") or "").strip()
    if author:
        return author
    maintainer = (info.get("maintainer") or "").strip()
    if maintainer:
        return maintainer
    return None


def get_project_url(data: dict) -> str | None:
    """Get project URL (home_page or package_url)."""
    info = data.get("info") or {}
    return info.get("home_page") or info.get("package_url") or info.get(
        "project_url")


def format_entry(name: str, version: str, data: dict) -> str:
    """Format a single NOTICE-style entry for one package."""
    info = data.get("info") or {}
    license_ = (info.get("license") or
                "").strip() or get_license_from_classifiers(
                    info.get("classifiers"))
    if not license_:
        license_ = "See project for license."
    year = get_copyright_year(data)
    author = get_author(data)
    url = get_project_url(data)

    lines = [
        f"{name} {version} - {license_}",
        "",
    ]
    if url:
        lines.append(f"<{url}>")
        lines.append("")
    if year and author:
        lines.append(f"Copyright (c) {year} {author}.")
    elif year:
        lines.append(f"Copyright (c) {year}.")
    elif author:
        lines.append(f"Copyright (c) {author}.")
    if lines[-1]:
        lines.append("")
    lines.append(
        "----------------------------------------------------------------")
    lines.append("")
    return "\n".join(lines)


def load_package_versions(paths: list[str]) -> OrderedDict[str, str]:
    """Load name -> version from one or more pip list files (name==version). Keeps first seen version."""
    seen = OrderedDict()
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "==" in line:
                        name, _, version = line.partition("==")
                        name = name.strip()
                        version = version.strip()
                        if name and version:
                            seen.setdefault(name, version)
        except FileNotFoundError:
            continue
    return seen


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate NOTICE-style attribution file for pip packages from PyPI metadata."
    )
    parser.add_argument(
        "list_files",
        nargs="+",
        metavar="pip_list.txt",
        help="One or more files with name==version per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="NOTICE_PIP",
        help="Output file path (default: NOTICE_PIP)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the header comment",
    )
    args = parser.parse_args()

    packages = load_package_versions(args.list_files)
    if not packages:
        sys.stderr.write("No packages found in list files.\n")
        sys.exit(1)

    out_lines = [] if args.no_header else [HEADER]

    for name, version in packages.items():
        data = fetch_pypi_metadata(name, version)
        if data:
            out_lines.append(format_entry(name, version, data))
        else:
            out_lines.append(
                f"{name} {version} - (license/copyright not retrieved from PyPI)\n"
                f"<https://pypi.org/project/{name}/>\n\n"
                "----------------------------------------------------------------\n\n"
            )

    with open(args.output, "w") as f:
        f.write("".join(out_lines))

    print(f"Wrote {args.output} with {len(packages)} packages.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
