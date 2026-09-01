#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Locate the build SDK exported by an installed CUDA-Q development wheel.

The development distribution is intentionally not imported.  Discovery uses
wheel metadata so the SDK may remain a data-only package and so CI can verify
that every build input comes from that wheel rather than a pre-existing
``/opt/llvm`` installation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
from typing import Iterable, Protocol


class DistributionLike(Protocol):
    files: Iterable[Path] | None
    metadata: object
    version: str

    def locate_file(self, path: Path) -> Path:
        ...


@dataclass(frozen=True)
class CudaqSDK:
    distribution: str
    version: str
    llvm_dir: Path
    mlir_dir: Path
    cudaq_dir: Path
    lib_dir: Path

    def value(self, field: str) -> str:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "llvm-dir": str(self.llvm_dir),
            "mlir-dir": str(self.mlir_dir),
            "cudaq-dir": str(self.cudaq_dir),
            "lib-dir": str(self.lib_dir),
        }[field]

    def as_json(self) -> dict[str, str]:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "llvm_dir": str(self.llvm_dir),
            "mlir_dir": str(self.mlir_dir),
            "cudaq_dir": str(self.cudaq_dir),
            "lib_dir": str(self.lib_dir),
        }


def normalized_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _metadata_name(distribution: DistributionLike, fallback: str) -> str:
    try:
        value = distribution.metadata["Name"]  # type: ignore[index]
    except (KeyError, TypeError):
        value = fallback
    return str(value)


def _one_sdk_file(
    distribution: DistributionLike,
    *,
    label: str,
    suffix: str,
) -> Path:
    files = tuple(distribution.files or ())
    matches = sorted({
        Path(distribution.locate_file(file)).resolve()
        for file in files
        if Path(str(file)).as_posix().endswith(suffix)
    })
    if not matches:
        raise RuntimeError(
            f"CUDA-Q development wheel does not provide {label} ({suffix})")
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"CUDA-Q development wheel provides ambiguous {label}: {rendered}")
    path = matches[0]
    if not path.is_file():
        raise RuntimeError(f"CUDA-Q development wheel entry is missing: {path}")
    return path


def locate_sdk(
    distribution: DistributionLike,
    *,
    requested_name: str,
    expected_version: str | None = None,
) -> CudaqSDK:
    """Validate and return the required CUDA-Q Logical build surface from *distribution*."""

    actual_name = _metadata_name(distribution, requested_name)
    if normalized_name(actual_name) != normalized_name(requested_name):
        raise RuntimeError(
            "CUDA-Q development distribution name mismatch: "
            f"expected {requested_name!r}, found {actual_name!r}")
    if normalized_name(actual_name) == "cudaq":
        raise RuntimeError(
            "the standard cudaq runtime distribution is not a development SDK")
    if expected_version and distribution.version != expected_version:
        raise RuntimeError(
            "CUDA-Q development SDK version mismatch: "
            f"expected {expected_version}, found {distribution.version}")

    llvm_config = _one_sdk_file(
        distribution,
        label="LLVM CMake package",
        suffix="lib/cmake/llvm/LLVMConfig.cmake",
    )
    mlir_config = _one_sdk_file(
        distribution,
        label="MLIR CMake package",
        suffix="lib/cmake/mlir/MLIRConfig.cmake",
    )
    cudaq_config = _one_sdk_file(
        distribution,
        label="CUDA-Q CMake package",
        suffix="lib/cmake/cudaq/CUDAQConfig.cmake",
    )
    llvm_lib_dir = llvm_config.parents[2]
    mlir_lib_dir = mlir_config.parents[2]
    cudaq_lib_dir = cudaq_config.parents[2]
    if not llvm_lib_dir == mlir_lib_dir == cudaq_lib_dir:
        raise RuntimeError(
            "CUDA-Q, LLVM, and MLIR CMake packages do not share one SDK lib "
            "directory")

    return CudaqSDK(
        distribution=actual_name,
        version=distribution.version,
        llvm_dir=llvm_config.parent,
        mlir_dir=mlir_config.parent,
        cudaq_dir=cudaq_config.parent,
        lib_dir=llvm_lib_dir,
    )


def installed_sdk(
    distribution_name: str,
    *,
    expected_version: str | None = None,
) -> CudaqSDK:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"CUDA-Q development distribution {distribution_name!r} "
            "is not installed") from error
    return locate_sdk(
        distribution,
        requested_name=distribution_name,
        expected_version=expected_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", default="cudaq-devel")
    parser.add_argument("--expect-version")
    parser.add_argument(
        "--field",
        choices=(
            "distribution",
            "version",
            "llvm-dir",
            "mlir-dir",
            "cudaq-dir",
            "lib-dir",
        ),
    )
    args = parser.parse_args()

    sdk = installed_sdk(
        args.distribution,
        expected_version=args.expect_version,
    )
    if args.field:
        print(sdk.value(args.field))
    else:
        print(json.dumps(sdk.as_json(), sort_keys=True))


if __name__ == "__main__":
    main()
