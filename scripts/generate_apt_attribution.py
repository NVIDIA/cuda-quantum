#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Writes a NOTICE-style attribution file for the apt packages that an image
# ships on top of its base image. The input is apt_package_details.tsv from
# diff_image_packages.sh and the license files collected from the image itself
# (apt_licenses/<binary>/*), so the notice describes the versions that are
# installed and needs no network access. Binary packages are grouped by the
# source package they are built from.

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

HEADER = """Apt packages - third-party attribution
This file lists license and copyright information for the apt packages that the
image ships on top of its base image, at the installed versions. The license
and copyright files of every package are kept next to the sources, under
{licenses_dir}/<binary package>/; the line "Files:" names the ones of each entry.

"""

SEPARATOR = "----------------------------------------------------------------\n"
KEYWORDS = [("LGPL", "LGPL"), ("GPL", "GPL"), ("Apache", "Apache-2.0"),
            ("BSD", "BSD"), ("MIT", "MIT"), ("Expat", "Expat"), ("MPL", "MPL"),
            ("zlib", "zlib"), ("ISC", "ISC"),
            ("Public domain", "public domain"),
            ("End User License", "proprietary EULA"),
            ("NVIDIA", "NVIDIA license")]
MAX_COPYRIGHTS = 4
MAX_LICENSES = 12


def license_files(licenses: Path, binary: str) -> list[Path]:
    # Multiarch packages are listed as name:arch; the directory is name_arch.
    d = licenses / binary.replace(":", "_")
    return sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []


def parse_copyright(text: str) -> tuple[list[str], list[str]]:
    """(licenses, copyright lines) of a Debian copyright file.

    Machine-readable (DEP-5) files give License: and Copyright: fields; older
    free-form files are scanned for copyright lines and known license names.
    """
    licenses: list[str] = []
    copyrights: list[str] = []

    def add(items: list[str], value: str):
        value = " ".join(value.split())
        if value and value not in items:
            items.append(value)

    if re.match(r"\s*Format:", text):
        field = None
        for line in text.splitlines():
            if line.startswith((" ", "\t")):
                if field == "copyright" and line.strip():
                    add(copyrights, line)
                continue
            m = re.match(r"([A-Za-z-]+):\s*(.*)", line)
            field = m.group(1).lower() if m else None
            if field == "license" and m.group(2):
                for name in re.split(r"\s+(?:or|and|with)\s+|,\s*", m.group(2)):
                    add(licenses, name)
            elif field == "copyright" and m.group(2):
                add(copyrights, m.group(2))
    else:
        for line in text.splitlines():
            if re.match(r"\s*(?:Copyright\b|\(c\)|©)", line, re.I):
                add(copyrights, line)
        for word, name in KEYWORDS:
            if re.search(word, text):
                add(licenses, name)
    return licenses, copyrights


def format_entry(source: str, version: str, binaries: list[str],
                 licenses_dir: Path, rel_dir: str) -> str:
    licenses: list[str] = []
    copyrights: list[str] = []
    files: list[str] = []
    for binary in binaries:
        for path in license_files(licenses_dir, binary):
            files.append(f"{rel_dir}/{binary.replace(':', '_')}/{path.name}")
            if not re.search(r"copyright", path.name, re.I):
                continue
            lic, cop = parse_copyright(path.read_text(errors="replace"))
            for item in lic:
                if item not in licenses:
                    licenses.append(item)
            for item in cop:
                if item not in copyrights:
                    copyrights.append(item)
    shown = ", ".join(licenses[:MAX_LICENSES]) or "see the license files"
    if len(licenses) > MAX_LICENSES:
        shown += ", ..."
    lines = [
        f"{source} {version} - {shown}", "",
        f"Binary packages: {', '.join(binaries)}", ""
    ]
    for item in copyrights[:MAX_COPYRIGHTS]:
        lines.append(item if item.lower().startswith((
            "copyright", "(c)", "©")) else f"Copyright: {item}")
    if len(copyrights) > MAX_COPYRIGHTS:
        lines.append(
            f"... and {len(copyrights) - MAX_COPYRIGHTS} more in the copyright file."
        )
    if copyrights:
        lines.append("")
    lines.append("Files: " +
                 (", ".join(files) if files else "none found in the image"))
    lines.append("")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NOTICE-style attribution for apt packages.")
    parser.add_argument(
        "details",
        help="apt_package_details.tsv (binary, version, source, source version)"
    )
    parser.add_argument("--licenses-dir",
                        required=True,
                        help="directory with <binary>/ license files")
    parser.add_argument(
        "--licenses-rel",
        help=
        "where the license files are in the source image (default: the directory name)"
    )
    parser.add_argument("-o", "--output", default="NOTICE_APT")
    args = parser.parse_args()

    groups: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    try:
        rows = [
            l.rstrip("\n").split("\t") for l in open(args.details) if l.strip()
        ]
    except FileNotFoundError:
        rows = []
    for binary, _version, source, source_version in sorted(rows,
                                                           key=lambda r:
                                                           (r[2], r[0])):
        groups.setdefault((source, source_version), []).append(binary)

    licenses_dir = Path(args.licenses_dir)
    rel = args.licenses_rel or str(licenses_dir)
    out = [HEADER.format(licenses_dir=rel)]
    for (source, version), binaries in groups.items():
        out.append(format_entry(source, version, binaries, licenses_dir, rel))
    Path(args.output).write_text("".join(out))
    print(f"Wrote {args.output} with {len(groups)} source packages.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
