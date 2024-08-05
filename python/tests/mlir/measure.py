# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

import cudaq


def test_kernel_measure_1q():
    """
    Test the measurement instruction for `cudaq.Kernel` by applying
    measurements to qubits one at a time.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Check that we can apply measurements to 1 qubit at a time.
    kernel.mx(qubit_0)
    kernel.mx(qubit_1)
    kernel.my(qubit_0)
    kernel.my(qubit_1)
    kernel.mz(qubit_0)
    kernel.mz(qubit_1)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_5:.*]] = quake.mx %[[VAL_3]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_6:.*]] = quake.mx %[[VAL_4]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_7:.*]] = quake.my %[[VAL_3]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_8:.*]] = quake.my %[[VAL_4]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_9:.*]] = quake.mz %[[VAL_3]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_4]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           return
# CHECK:         }


def test_kernel_measure_qreg():
    """
    Test the measurement instruciton for `cudaq.Kernel` by applying
    measurements to an entire qreg.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(3)
    # Check that we can apply measurements to an entire register.
    kernel.mx(qreg)
    kernel.my(qreg)
    kernel.mz(qreg)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.mx %[[VAL_0]] name "" : (!quake.veq<3>) -> !cc.stdvec<!quake.measure>
# CHECK:           %[[VAL_2:.*]] = quake.my %[[VAL_0]] name "" : (!quake.veq<3>) -> !cc.stdvec<!quake.measure>
# CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_0]] name "" : (!quake.veq<3>) -> !cc.stdvec<!quake.measure>
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
