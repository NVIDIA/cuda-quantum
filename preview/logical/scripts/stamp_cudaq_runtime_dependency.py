#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Stamp the ABI-paired CUDA-Q runtime version into CUDA-Q Logical release metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

try:
    from packaging.version import Version
except ModuleNotFoundError:
    from pip._vendor.packaging.version import Version

# The CUDA-Q runtime arrives through these two extras; the base package is
# runtime-agnostic. Release CI stamps both bare entries with ==CUDAQ_VERSION.
RUNTIME_DISTRIBUTIONS = ("cuda-quantum-cu12", "cuda-quantum-cu13")


def stamp_runtime_dependency(
    pyproject: Path,
    *,
    distribution: str,
    version: str,
) -> str:
    """Replace one bare runtime dependency with an exact validated pin."""

    normalized_version = str(Version(version))
    text = pyproject.read_text()
    # Match the bare quoted dependency wherever it sits in the TOML array
    # (its own line, or collapsed onto one line by a formatter) rather than
    # requiring it to be the sole content of its line.
    pattern = re.compile(rf'"{re.escape(distribution)}"')
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{pyproject} must contain exactly one bare \"{distribution}\" "
            f"dependency entry; found {len(matches)}")
    dependency = f"{distribution}=={normalized_version}"
    replacement = f'"{dependency}"'
    pyproject.write_text(pattern.sub(replacement, text, count=1))
    return dependency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    parser.add_argument(
        "--distribution",
        dest="distributions",
        action="append",
        default=None,
        help="Bare dependency entry to pin; repeatable. Defaults to both "
        "CUDA-Q runtime extras: " + ", ".join(RUNTIME_DISTRIBUTIONS),
    )
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    for distribution in args.distributions or RUNTIME_DISTRIBUTIONS:
        dependency = stamp_runtime_dependency(
            args.pyproject,
            distribution=distribution,
            version=args.version,
        )
        print(f"stamped CUDA-Q Logical runtime dependency: {dependency}")


if __name__ == "__main__":
    main()
