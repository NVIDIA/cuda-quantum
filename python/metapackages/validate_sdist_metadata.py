# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Regression checks for cudaq metapackage sdist metadata (issue #3433)."""

from __future__ import annotations

import argparse
import email.parser
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _read_pkg_info_from_sdist(sdist_path: Path) -> str:
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.name.endswith("/PKG-INFO"):
                payload = archive.extractfile(member)
                if payload is None:
                    raise ValueError(f"Could not read {member.name} from {sdist_path}")
                return payload.read().decode()

    raise ValueError(f"No PKG-INFO found in {sdist_path}")


def _metadata_name(metadata_text: str) -> str:
    message = email.parser.Parser().parsestr(metadata_text)
    name = message.get("Name")
    if not name:
        raise ValueError("Metadata is missing the Name field")
    return name


def _assert_no_legacy_setuptools_packaging(sdist_path: Path) -> None:
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        names = {Path(member.name).name for member in archive.getmembers()}
    legacy = {"setup.py", "setup.cfg"} & names
    if legacy:
        raise ValueError(
            f"{sdist_path} still ships legacy packaging files: {sorted(legacy)}"
        )


def _assert_prepare_metadata_name(sdist_path: Path, expected_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_dir = Path(tmp_dir) / "sdist"
        with tarfile.open(sdist_path, mode="r:gz") as archive:
            archive.extractall(extract_dir, filter="data")

        source_dir = next(extract_dir.iterdir())
        metadata_dir = Path(tmp_dir) / "metadata"
        metadata_dir.mkdir()

        env = os.environ.copy()
        env.pop("CUDAQ_META_SDIST_BUILD", None)
        env.pop("CUDAQ_META_WHEEL_BUILD", None)

        script = f"""
import glob
import os
import subprocess
import sys

def runner(cmd, cwd=None, extra_environ=None):
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    completed = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        completed.check_returncode()
    return completed

from pyproject_hooks import BuildBackendHookCaller

hook_caller = BuildBackendHookCaller(
    source_dir={str(source_dir)!r},
    build_backend="hatchling.build",
    backend_path=None,
    runner=runner,
)
hook_caller.prepare_metadata_for_build_wheel({str(metadata_dir)!r})

for metadata_file in glob.glob({str(metadata_dir)!r} + "/**/*.dist-info/METADATA",
                               recursive=True):
    with open(metadata_file, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Name:"):
                print(line.strip().split(":", 1)[1].strip())
                raise SystemExit(0)

raise SystemExit("METADATA file with Name field was not generated")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "prepare_metadata_for_build_wheel failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )

        generated_name = completed.stdout.strip()
        if generated_name.lower() == "unknown":
            raise ValueError(
                "prepare_metadata_for_build_wheel reported project name 'unknown'"
            )
        if generated_name != expected_name:
            raise ValueError(
                f"prepare_metadata_for_build_wheel reported name {generated_name!r}, "
                f"expected {expected_name!r}"
            )


def validate_sdist_metadata(sdist_path: Path, expected_name: str) -> None:
    pkg_info = _read_pkg_info_from_sdist(sdist_path)
    pkg_info_name = _metadata_name(pkg_info)
    if pkg_info_name.lower() == "unknown":
        raise ValueError(f"PKG-INFO Name is 'unknown' in {sdist_path}")
    if pkg_info_name != expected_name:
        raise ValueError(
            f"PKG-INFO Name is {pkg_info_name!r}, expected {expected_name!r}"
        )

    _assert_no_legacy_setuptools_packaging(sdist_path)
    _assert_prepare_metadata_name(sdist_path, expected_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate cudaq metapackage sdist metadata."
    )
    parser.add_argument("sdist", type=Path, help="Path to the cudaq sdist tarball")
    parser.add_argument(
        "--name",
        default="cudaq",
        help="Expected project name in sdist metadata (default: cudaq)",
    )
    args = parser.parse_args()

    validate_sdist_metadata(args.sdist.resolve(), args.name)
    print(f"Validated sdist metadata for {args.name!r} in {args.sdist}")


if __name__ == "__main__":
    main()
