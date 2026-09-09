# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Install a `cudaq.logical` wheel into the release image's existing CUDA-Q prefix."""

import argparse
from importlib.metadata import Distribution
from pathlib import Path
import subprocess
import sys
import tempfile


def pip_install(*args):
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", *args
    ],
                   check=True)


def install(wheel_dir: Path, prefix: Path):
    wheels = sorted(wheel_dir.glob("cudaq_logical-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one Logical wheel in {wheel_dir}, "
                         f"found {len(wheels)}")
    if not (prefix / "cudaq/__init__.py").is_file():
        raise ValueError(f"Missing CUDA-Q Python installation in {prefix}")

    with tempfile.TemporaryDirectory(prefix="cudaq-logical-") as temporary:
        staging = Path(temporary) / "package"
        # pip checks wheel compatibility before copying into the existing package.
        pip_install("--no-index", "--no-deps", "--no-compile", "--target",
                    str(staging), str(wheels[0].resolve()))
        metadata_dir, = staging.glob("cudaq_logical-*.dist-info")
        distribution = Distribution.at(metadata_dir)
        directories = ("cudaq/logical", "cudaq/mlir/dialects",
                       "cudaq/mlir/_mlir_libs", "bin")
        for relative in directories:
            if not (staging / relative).is_dir():
                raise ValueError(f"`cudaq.logical` wheel is missing {relative}")

        # Install declared dependencies (including Python Stim), without extras.
        if distribution.requires:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("\n".join(distribution.requires) + "\n",
                                    encoding="utf-8")
            pip_install("--break-system-packages", "-r", str(requirements))

        # Copy the package and tool launchers, preserving existing files and modes.
        for relative in directories:
            destination = prefix / relative
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "cp", "--archive", "--no-clobber",
                str(staging / relative) + "/.",
                str(destination)
            ],
                           check=True)
        print(f"Copied cudaq-logical {distribution.version} into {prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    arguments = parser.parse_args()
    install(arguments.wheel_dir, arguments.prefix)
