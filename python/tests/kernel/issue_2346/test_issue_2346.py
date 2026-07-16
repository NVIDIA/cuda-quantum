# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Regression test for issue #2346:
# A kernel called via an intermediate import path (gates.qft_ops.qft_kernel)
# must resolve even though it is registered under the fully-qualified path
# (algo_lib.gates.qft_ops.qft_kernel).
#
# Package layout (relative to this file):
#   algo_lib/__init__.py              <- from .gates import qft_ops
#   algo_lib/gates/__init__.py
#   algo_lib/gates/qft_ops.py         <- defines @cudaq.kernel def qft_kernel()

import importlib

import cudaq
import pytest

from algo_lib import gates

ast_bridge = importlib.import_module("cudaq.kernel.ast_bridge")


@pytest.fixture(autouse=True)
def clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_kernel_call_via_partial_module_path():

    @cudaq.kernel
    def test0():
        gates.qft_ops.qft_kernel()

    counts = cudaq.sample(test0)
    assert '0' in counts


def test_repeated_kernel_call_via_partial_module_path(monkeypatch):

    class CountingModuleAlias:

        def __init__(self, module):
            self.module = module
            self.accesses = 0

        @property
        def qft_ops(self):
            self.accesses += 1
            return self.module.qft_ops

    qft_alias = CountingModuleAlias(gates)

    @cudaq.kernel
    def test0():
        qft_alias.qft_ops.qft_kernel()
        qft_alias.qft_ops.qft_kernel()

    @cudaq.kernel
    def test1():
        qft_alias.qft_ops.qft_kernel()
        qft_alias.qft_ops.qft_kernel()

    original_recover = ast_bridge.recover_value_of_or_none
    qualified_recovery_calls = 0

    def counting_recover(name, *args, **kwargs):
        nonlocal qualified_recovery_calls
        if name == "algo_lib.gates.qft_ops.qft_kernel":
            qualified_recovery_calls += 1
        return original_recover(name, *args, **kwargs)

    monkeypatch.setattr(ast_bridge, "recover_value_of_or_none",
                        counting_recover)
    counts = cudaq.sample(test0)
    assert '00' in counts
    assert qft_alias.accesses == 1
    assert qualified_recovery_calls == 1

    counts = cudaq.sample(test1)
    assert '00' in counts
    assert qft_alias.accesses == 2
    assert qualified_recovery_calls == 2
