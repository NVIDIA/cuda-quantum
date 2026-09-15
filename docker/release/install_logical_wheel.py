# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Install a `cudaq.logical` wheel into the release image's existing CUDA-Q prefix."""

import argparse
import base64
import csv
import hashlib
from importlib.metadata import Distribution
from pathlib import Path
import shutil
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
    if list(prefix.glob("cudaq_logical-*.dist-info")):
        raise ValueError(f"Logical package metadata already exists in {prefix}")

    with tempfile.TemporaryDirectory(prefix="cudaq-logical-") as temporary:
        staging = Path(temporary) / "package"
        # pip checks wheel compatibility before copying into the existing package.
        pip_install("--no-index", "--no-deps", "--no-compile", "--target",
                    str(staging), str(wheels[0].resolve()))
        metadata_dir, = staging.glob("cudaq_logical-*.dist-info")
        distribution = Distribution.at(metadata_dir)
        directories = ("cudaq/logical", "cudaq/mlir/dialects",
                       "cudaq/mlir/_mlir_libs", "bin", metadata_dir.name)
        for relative in directories:
            if not (staging / relative).is_dir():
                raise ValueError(f"`cudaq.logical` wheel is missing {relative}")

        # Install declared dependencies (including Python Stim), without extras.
        if distribution.requires:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("\n".join(distribution.requires) + "\n",
                                    encoding="utf-8")
            pip_install("--break-system-packages", "-r", str(requirements))

        # Copy package files, tools, and version metadata without replacing
        # existing CUDA-Q files or claiming them in Logical's installed-file list.
        records = []
        for directory in directories:
            for source in sorted((staging / directory).rglob("*")):
                if not source.is_file() or source in (metadata_dir / "RECORD",
                                                      metadata_dir /
                                                      "direct_url.json"):
                    continue
                relative = source.relative_to(staging)
                destination = prefix / relative
                if destination.exists() or destination.is_symlink():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                with destination.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").digest()
                encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                records.append((relative.as_posix(), f"sha256={encoded}",
                                destination.stat().st_size))

        # Record final paths, including bin/qlx-opt and bin/qlx-translate.
        record = Path(metadata_dir.name) / "RECORD"
        records.append((record.as_posix(), "", ""))
        with (prefix / record).open("w", encoding="utf-8",
                                    newline="") as handle:
            csv.writer(handle).writerows(records)
        print(f"Installed cudaq-logical {distribution.version} into {prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    arguments = parser.parse_args()
    install(arguments.wheel_dir, arguments.prefix)
