#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Assert cudaq-logical is co-located with cudaq and shares its MLIR bindings."""

from __future__ import annotations

from pathlib import Path

import cudaq
import cudaq.logical
import cudaq.mlir.ir as mlir_ir


def main() -> None:
    cudaq_root = Path(cudaq.__file__).resolve().parent
    logical_root = Path(cudaq.logical.__file__).resolve().parent
    if logical_root.parent != cudaq_root:
        raise SystemExit("cudaq.logical is not co-located with cudaq:\n"
                         f"  cudaq:    {cudaq_root}\n"
                         f"  logical:  {logical_root}")

    import cudaq.mlir._mlir_libs._qlx_ext  # noqa: F401

    from cudaq.mlir import ir as mlir_ir_again

    if mlir_ir is not mlir_ir_again or mlir_ir is not cudaq.mlir.ir:
        raise SystemExit(
            "cudaq.logical and cudaq do not agree on mlir.ir module identity:\n"
            f"  cudaq.mlir.ir: {cudaq.mlir.ir}\n"
            f"  import ir:     {mlir_ir}\n"
            f"  from cudaq.mlir import ir: {mlir_ir_again}")

    print("cudaq.logical namespace install ok")
    print(f"  cudaq:     {cudaq_root}")
    print(f"  logical:   {logical_root}")
    print(f"  mlir.ir:   {mlir_ir.__file__}")


if __name__ == "__main__":
    main()
