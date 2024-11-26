# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
def test_multiple_measurement():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    cudaq.set_target("braket", emulate=True, machine=device_arn)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        mz(qubits[0])
        mz(qubits[1])

    print(kernel)
    cudaq.sample(kernel, shots_count=100).dump()

#test_multiple_measurement()

def test_qvector_slicing():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    cudaq.set_target("braket", emulate=True, machine=device_arn)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        x(q.front(2))
        mz(q)

    print(kernel)
    cudaq.sample(kernel, shots_count=100).dump()

#test_qvector_slicing()

def test_mid_circuit_measurement():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    cudaq.set_target("braket", emulate=True, machine=device_arn)
    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        if mz(q[0]):
            x(q[1])
        mz(q)

    ## error: 'cf.cond_br' op unable to translate op to OpenQASM 2.0
    cudaq.sample(simple, shots_count=100).dump()

test_mid_circuit_measurement()

# module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__kernel = "__nvqpp__mlirgen__kernel_PyKernelEntryPointRewrite"}} {
#   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
#     %0 = quake.alloca !quake.veq<2>
#     %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
#     quake.h %1 : (!quake.ref) -> ()
#     %measOut = quake.mz %1 : (!quake.ref) -> !quake.measure
#     %2 = quake.extract_ref %0[1] : (!quake.veq<2>) -> !quake.ref
#     %measOut_0 = quake.mz %2 : (!quake.ref) -> !quake.measure
#     return
#   }
# }