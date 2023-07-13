# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq


def test_kernel_2q():
    """
    Test the `cudaq.Kernel` on each two-qubit gate (controlled 
    single qubit gates). We alternate the order of the control and target
    qubits between each successive gate.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # First three gates check the overload for providing a single control
    # qubit as a list of length 1.
    # Test both with and without keyword arguments.
    kernel.ch(controls=[qubit_0], target=qubit_1)
    kernel.cx([qubit_1], qubit_0)
    kernel.cy([qubit_0], qubit_1)
    # Check the overload for providing a single control qubit on its own.
    # Test both with and without keyword arguments.
    kernel.cz(control=qubit_1, target=qubit_0)
    kernel.ct(qubit_0, qubit_1)
    kernel.cs(qubit_1, qubit_0)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.x [%[[VAL_4]]] %[[VAL_3]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.y [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.z [%[[VAL_4]]] %[[VAL_3]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.t [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.s [%[[VAL_4]]] %[[VAL_3]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_kernel_3q():
    """
    Test the `cudaq.Kernel` on each multi-qubit gate (multi-controlled single
    qubit gates). We do this for the case of a 3-qubit kernel. 
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 3.
    qreg = kernel.qalloc(3)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    qubit_2 = qreg[2]
    # Apply each gate to entire register.
    # Note: we alternate between orders to make the circuit less trivial.
    kernel.ch([qubit_0, qubit_1], qubit_2)
    kernel.cx([qubit_2, qubit_0], qubit_1)
    kernel.cy([qubit_1, qubit_2], qubit_0)
    kernel.cz([qubit_0, qubit_1], qubit_2)
    kernel.ct([qubit_2, qubit_0], qubit_1)
    kernel.cs([qubit_1, qubit_2], qubit_0)
    kernel()
    assert (kernel.arguments == [])
    assert (kernel.argument_count == 0)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.h [%[[VAL_4]], %[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.x [%[[VAL_6]], %[[VAL_4]]] %[[VAL_5]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.y [%[[VAL_5]], %[[VAL_6]]] %[[VAL_4]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.z [%[[VAL_4]], %[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.t [%[[VAL_6]], %[[VAL_4]]] %[[VAL_5]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.s [%[[VAL_5]], %[[VAL_6]]] %[[VAL_4]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
