# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest

import cudaq


def test_swap_2q():
    """
    Tests the simple case of swapping the states of two qubits.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Place qubit 0 in the 1-state.
    kernel.x(qubit_0)
    # Swap states with qubit 1.
    kernel.swap(qubit_0, qubit_1)
    # Check their states.
    kernel.mz(qreg)
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:      () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.swap %[[VAL_1]], %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
