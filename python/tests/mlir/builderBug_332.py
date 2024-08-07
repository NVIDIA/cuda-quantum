# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_recursive_calls():
    kernel1, qubit1 = cudaq.make_kernel(cudaq.qubit)
    # print(kernel1)

    kernel2, qubit2 = cudaq.make_kernel(cudaq.qubit)
    kernel2.apply_call(kernel1, qubit2)
    # print(kernel2)

    kernel3 = cudaq.make_kernel()
    qreg3 = kernel3.qalloc(1)
    qubit3 = qreg3[0]
    kernel3.apply_call(kernel2, qubit3)

    print(kernel3)


# CHECK-LABEL:  func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:    %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
# CHECK:    %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
# CHECK:    call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_1]]) : (!quake.ref) -> ()
# CHECK:    return
# CHECK:  }
# CHECK:  func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%arg0: !quake.ref) {
# CHECK:    call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%arg0) : (!quake.ref) -> ()
# CHECK:    return
# CHECK:  }
# CHECK:  func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%arg0: !quake.ref) {
# CHECK:    return
# CHECK:  }
# CHECK:}
