#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Writes a NOTICE-style attribution file for the pip packages that an image
# ships. The license and the copyright lines come from the package itself:
# the metadata (PKG-INFO) and the license files of the unpacked sdist, or, for
# packages without an sdist, the METADATA and license files collected from the
# installed distribution. There is no network access, and no copyright year is
# guessed: a package without a copyright line in its files has none listed.

import argparse
import re
import sys
from email.parser import Parser
from pathlib import Path

HEADER = """Pip packages - third-party attribution
This file lists license and copyright information for the pip packages that the
image ships, at the installed versions. The license and copyright lines are
taken from each package's own metadata and license files; the line "Source:"
says where its source or its license files are in the source image.

"""

SEPARATOR = "----------------------------------------------------------------\n"
LICENSE_FILE = re.compile(r"(licen[cs]e|copying|copyright|notice|authors)",
                          re.I)
# "Copyright (c) 2018 Name", "Copyright 2018 Name", "(c) 2018 Name"; not the
# disclaimer "COPYRIGHT HOLDERS BE LIABLE" or "copyright notice, this list".
COPYRIGHT_LINE = re.compile(
    r"^\s*(?:Copyright\s*(?:\(c\)|©)|Copyright\s+\d|\(c\)\s*\d|©\s*\d)", re.I)
MAX_COPYRIGHTS = 3


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_aliases(path: str | None) -> dict[str, str]:
    aliases = {}
    if path and Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                aliases[normalize(parts[0])] = normalize(parts[1])
    return aliases


def find_source_dir(pip_dir: Path, name: str, version: str,
                    aliases: dict[str, str]) -> Path | None:
    wanted = {
        normalize(f"{n}-{version}")
        for n in (name, aliases.get(normalize(name), name))
    }
    for d in pip_dir.iterdir() if pip_dir.is_dir() else []:
        if d.is_dir() and normalize(d.name) in wanted:
            return d
    return None


def license_from_metadata(meta) -> str | None:
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    classifiers = [
        c.split(" :: ")[-1]
        for c in meta.get_all("Classifier", [])
        if c.startswith("License ::")
    ]
    classifiers = [c for c in classifiers if c not in ("OSI Approved",)]
    if classifiers:
        return " / ".join(classifiers)
    text = (meta.get("License") or "").strip()
    # A License field that is the license text itself is too long to be a name.
    if text and "\n" not in text and len(text) <= 100:
        return text
    return None


def metadata_of(source_dir: Path | None, files: list[Path]):
    candidates = []
    if source_dir:
        candidates.append(source_dir / "PKG-INFO")
    candidates += [f for f in files if f.name.endswith("METADATA")]
    for c in candidates:
        if c.is_file():
            return Parser().parsestr(c.read_text(errors="replace"),
                                     headersonly=True)
    return None


def license_file_paths(source_dir: Path | None,
                       files: list[Path]) -> list[Path]:
    found = list(files)
    if source_dir:
        for p in sorted(source_dir.iterdir()):
            if p.is_file() and LICENSE_FILE.search(p.name):
                found.append(p)
            elif p.is_dir() and p.name.lower() in ("licenses", "license",
                                                   "licences"):
                found += [q for q in sorted(p.iterdir()) if q.is_file()]
    return [f for f in found if not f.name.endswith("METADATA")]


def copyright_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if COPYRIGHT_LINE.match(line):
                line = " ".join(line.split())
                if line not in lines:
                    lines.append(line)
    return lines


def format_entry(name: str, version: str, source_dir: Path | None,
                 lic_files: list[Path], rel_source: str, rel_licenses: str,
                 license_name: str | None, url: str | None,
                 copyrights: list[str]) -> str:
    lines = [
        f"{name} {version} - {license_name or 'see the license files'}", ""
    ]
    if url:
        lines += [f"<{url}>", ""]
    for line in copyrights[:MAX_COPYRIGHTS]:
        lines.append(line)
    if len(copyrights) > MAX_COPYRIGHTS:
        lines.append(
            f"... and {len(copyrights) - MAX_COPYRIGHTS} more in the license files."
        )
    if copyrights:
        lines.append("")
    if source_dir:
        lines.append(f"Source: {rel_source}/{source_dir.name}")
    else:
        lines.append(
            f"Source: none (no sdist); license files in {rel_licenses}/{normalize(name)}"
        )
    lines += ["", SEPARATOR]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NOTICE-style attribution for pip packages.")
    parser.add_argument("list_file", help="file with name==version per line")
    parser.add_argument("--pip-dir",
                        required=True,
                        help="directory with the unpacked sdists")
    parser.add_argument(
        "--licenses-dir",
        required=True,
        help=
        "directory with <name>/ license files from the installed distributions")
    parser.add_argument(
        "--pip-rel",
        help="where the sdists are in the source image (default: --pip-dir)")
    parser.add_argument(
        "--licenses-rel",
        help=
        "where the license files are in the source image (default: --licenses-dir)"
    )
    parser.add_argument("--aliases", help="pip_source_aliases.txt")
    parser.add_argument("-o", "--output", default="NOTICE_PIP")
    args = parser.parse_args()

    pip_dir, licenses_dir = Path(args.pip_dir), Path(args.licenses_dir)
    rel_source = args.pip_rel or args.pip_dir
    rel_licenses = args.licenses_rel or args.licenses_dir
    aliases = read_aliases(args.aliases)

    packages = []
    try:
        for line in Path(args.list_file).read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                name, _, version = line.strip().partition("==")
                packages.append((name, version))
    except FileNotFoundError:
        pass

    # An empty list is a valid result (e.g. an image that ships no Python);
    # still write the header so downstream copies find a file.
    out = [HEADER]
    for name, version in sorted(packages, key=lambda p: normalize(p[0])):
        source_dir = find_source_dir(pip_dir, name, version, aliases)
        installed = licenses_dir / normalize(name)
        if not installed.is_dir(
        ):  # importlib.metadata keeps the project's own spelling
            installed = next((d for d in licenses_dir.iterdir() if normalize(d.name) == normalize(name)), installed) \
                if licenses_dir.is_dir() else installed
        files = sorted(p for p in installed.iterdir()
                       if p.is_file()) if installed.is_dir() else []
        meta = metadata_of(source_dir, files)
        lic_files = license_file_paths(source_dir, files)
        url = None
        if meta:
            url = meta.get("Home-page") or next(
                (v.split(",", 1)[-1].strip()
                 for v in meta.get_all("Project-URL", [])
                 if v.lower().startswith(("homepage", "home", "source",
                                          "repository"))), None)
        out.append(
            format_entry(name, version, source_dir, lic_files, rel_source,
                         rel_licenses,
                         license_from_metadata(meta) if meta else None, url,
                         copyright_lines(lic_files)))
    Path(args.output).write_text("".join(out))
    print(f"Wrote {args.output} with {len(packages)} packages.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
