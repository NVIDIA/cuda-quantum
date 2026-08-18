# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
from typing import Tuple

import cudaq


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_typed_tuple_as_kernel_argument():
    @cudaq.kernel
    def kernel(a: tuple[int, int]):
        q = cudaq.qvector(2)

    cudaq.sample(kernel, (1, 2))


def test_list_of_typed_tuples_as_kernel_argument():
    """Matches the original issue report once tuple element types are specified."""

    @cudaq.kernel
    def kernel(a: list[tuple[int, int]]):
        q = cudaq.qvector(2)

    cudaq.sample(kernel, [(1, 2)])


def test_bare_list_tuple_annotation_reports_helpful_error():
    with pytest.raises(
            cudaq.kernel.ast_bridge.CompilerError,
            match="tuple argument annotation must provide element types"):

        @cudaq.kernel
        def kernel(a: list[tuple]):
            q = cudaq.qvector(2)


def test_bare_tuple_annotation_reports_helpful_error():
    with pytest.raises(
            cudaq.kernel.ast_bridge.CompilerError,
            match="tuple argument annotation must provide element types"):

        @cudaq.kernel
        def kernel(a: tuple):
            q = cudaq.qvector(2)


def test_bare_tuple_capitalized_annotation_reports_helpful_error():
    with pytest.raises(
            cudaq.kernel.ast_bridge.CompilerError,
            match="tuple argument annotation must provide element types"):

        @cudaq.kernel
        def kernel(a: Tuple):
            q = cudaq.qvector(2)
