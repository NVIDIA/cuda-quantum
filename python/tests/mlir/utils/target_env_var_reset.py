# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. pytest -rP  %s

import os

os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "density-matrix-cpu"

import pytest

import cudaq


def test_env_var_update():
    """Tests that if the environment variable does not take effect on-the-fly : Builder mode"""

    os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "qpp-cpu"
    assert ("qpp-cpu" != cudaq.get_target().name)

    cudaq.set_target("qpp-cpu")
    assert ("qpp-cpu" == cudaq.get_target().name)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    result = cudaq.sample(kernel)
    result.dump()
    assert '00' in result
    assert '11' in result

    cudaq.reset_target()
    assert ("density-matrix-cpu" == cudaq.get_target().name)


def test_env_var_update_kernel():
    """Tests that if the environment variable does not take effect on-the-fly : MLIR mode"""

    @cudaq.kernel
    def simple():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "qpp-cpu"
    assert ("qpp-cpu" != cudaq.get_target().name)

    cudaq.set_target("qpp-cpu")
    assert ("qpp-cpu" == cudaq.get_target().name)

    result = cudaq.sample(simple)
    result.dump()
    assert '00' in result
    assert '11' in result

    cudaq.reset_target()
    assert ("density-matrix-cpu" == cudaq.get_target().name)


os.environ.pop("CUDAQ_DEFAULT_SIMULATOR")

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
