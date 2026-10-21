# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Build and verify an EDEM source distribution with extra local files present."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


def check_archive(archive, root, files):
    expected = {str(path): (root / path).read_bytes() for path in files}
    expected["pyproject.toml"] = expected["pyproject.toml.edem"]
    allowed = set(expected) | {"PKG-INFO"}
    directories = {
        str(parent)
        for name in allowed
        for parent in PurePosixPath(name).parents
    }
    roots = set()
    actual = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or
                    str(path) != member.name or "\\" in member.name):
                raise RuntimeError(
                    f"Invalid source archive path: {member.name}")
            roots.add(path.parts[0])
            relative = str(PurePosixPath(*path.parts[1:]))
            if member.isdir() and relative in directories:
                continue
            if (not member.isfile() or relative not in allowed or
                    relative in actual):
                raise RuntimeError(
                    f"Unexpected source archive entry: {member.name}")
            actual[relative] = source.extractfile(member).read()
    if len(roots) != 1 or set(actual) != allowed:
        raise RuntimeError("Source archive does not match the source manifest")
    for name, content in expected.items():
        if actual[name] != content:
            raise RuntimeError(f"Source archive changed file contents: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist/edem"))
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "edem/scripts"))
    import build_wheel

    files = build_wheel.source_files(root)
    with tempfile.TemporaryDirectory(
            prefix="cudaq-edem-sdist-test-") as temporary:
        work = Path(temporary)
        checkout = work / "source"
        build_wheel.stage_sources(root, checkout)
        for name in (
                "edem/unlisted.txt",
                "edem/lib/unlisted.cpp",
                "math/gf2/unlisted.txt",
                "math/gf2/lib/unlisted.cpp",
        ):
            (checkout /
             name).write_text("Unlisted source packaging test file.\n")
        artifacts = work / "dist"
        subprocess.run([
            sys.executable,
            str(checkout / "edem/scripts/build_wheel.py"), "--outdir",
            str(artifacts)
        ],
                       check=True)
        archives = list(artifacts.glob("*.tar.gz"))
        wheels = list(artifacts.glob("*.whl"))
        if len(archives) != 1 or len(wheels) != 1:
            raise RuntimeError("Expected one source distribution and one wheel")
        check_archive(archives[0], root, files)
        outdir.mkdir(parents=True, exist_ok=True)
        for artifact in archives + wheels:
            shutil.copy2(artifact, outdir / artifact.name)
    print(
        "EDEM source manifest, file contents, and unlisted-file exclusion passed"
    )


if __name__ == "__main__":
    main()
