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


def stamp_runtime_dependency(
    pyproject: Path,
    *,
    distribution: str,
    version: str,
) -> str:
    """Replace one bare runtime dependency with an exact validated pin."""

    normalized_version = str(Version(version))
    text = pyproject.read_text()
    pattern = re.compile(
        rf'^(?P<indent>\s*)"{re.escape(distribution)}",\s*$',
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{pyproject} must contain exactly one bare \"{distribution}\", "
            f"dependency; found {len(matches)}")
    dependency = f"{distribution}=={normalized_version}"
    replacement = f'{matches[0].group("indent")}"{dependency}",'
    pyproject.write_text(pattern.sub(replacement, text, count=1))
    return dependency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("--distribution", default="cudaq")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    dependency = stamp_runtime_dependency(
        args.pyproject,
        distribution=args.distribution,
        version=args.version,
    )
    print(f"stamped CUDA-Q Logical runtime dependency: {dependency}")


if __name__ == "__main__":
    main()
