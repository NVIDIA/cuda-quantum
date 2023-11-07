# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "density-matrix-cpu"

import pytest

import cudaq

def test_default_target():
    """Tests the default target set by environment variable"""
    assert ("density-matrix-cpu" == cudaq.get_target().name)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '00' in result
    assert '11' in result

def test_target_override():
    """Tests the target set by environment variable is overridden by user setting"""

    cudaq.set_target("qpp-cpu")
    assert("qpp-cpu" == cudaq.get_target().name)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '00' in result
    assert '11' in result

def test_ignore_invalid_target():
    """Tests that if the environment variable is not a valid target, it is ignored"""

    os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "foo"
    assert("foo" != cudaq.get_target().name)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '00' in result
    assert '11' in result
