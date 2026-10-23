# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Build the cudaq-edem source distribution and wheel from isolated sources."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


def source_files(root):
    root = root.resolve()
    files = []
    seen = set()
    for line in (root / "edem/package-sources.txt").read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or
                str(relative) != name or "\\" in name or name in seen):
            raise ValueError(f"Invalid package source path: {name}")
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or
                not path.is_file()):
            raise ValueError(f"Missing or invalid package source: {name}")
        seen.add(name)
        files.append(relative)
    for required in ("edem/package-sources.txt", "pyproject.toml.edem"):
        if required not in seen:
            raise ValueError(f"Required package source is absent: {required}")
    return files


def stage_sources(root, destination):
    for relative in source_files(root):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    shutil.copy2(destination / "pyproject.toml.edem",
                 destination / "pyproject.toml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist/edem"))
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="cudaq-edem-build-") as temporary:
        source = Path(temporary)
        stage_sources(root, source)
        # The default build creates a source distribution, then builds its wheel
        # from that archive.
        subprocess.run([
            sys.executable, "-m", "build", "--outdir",
            str(outdir),
            str(source)
        ],
                       check=True)


if __name__ == "__main__":
    main()
