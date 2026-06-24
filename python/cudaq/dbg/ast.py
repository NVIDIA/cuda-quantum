# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def print_i64(value: int) -> None:
    raise RuntimeError(
        'cudaq.dbg.ast.print_i64 can only be called from a CUDA-Q kernel')


def print_f64(value: float) -> None:
    raise RuntimeError(
        'cudaq.dbg.ast.print_f64 can only be called from a CUDA-Q kernel')
