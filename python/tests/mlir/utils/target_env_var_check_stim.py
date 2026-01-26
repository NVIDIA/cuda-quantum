# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. pytest -rP  %s

import os

os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import pytest

import cudaq
import numpy as np


def test_default_target():
    """Tests the default target set by environment variable"""

    assert ("stim" == cudaq.get_target().name)
    # This is a double-precision simulator
    assert (cudaq.complex() is np.complex128)
    kernel = cudaq.make_kernel()
    # Only stim can do this (200 qubits)
    qubits = kernel.qalloc(200)
    kernel.h(qubits[0])
    for i in range(199):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '0' * 200 in result
    assert '1' * 200 in result


def test_env_var_with_emulate():
    """Tests the target when emulating a hardware backend"""

    assert ("stim" == cudaq.get_target().name)
    cudaq.set_target("quantinuum", emulate=True)
    assert ("quantinuum" == cudaq.get_target().name)
    # The underlying simulator (`stim`) used for emulation is a double-precision simulator
    assert (cudaq.complex() is np.complex128)

    # `Stim` is used for emulation, hence can handle lots of qubits
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(200)
    kernel.h(qubits[0])
    for i in range(199):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '0' * 200 in result
    assert '1' * 200 in result


os.environ.pop("CUDAQ_DEFAULT_SIMULATOR")


# This isn't really an environment variable test, but version checking could
# loosely be interpreted as "environment" checking, so hence placing the test
# here.
def test_version():
    assert "CUDA-Q Version" in cudaq.__version__


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
