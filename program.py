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

test_multiple_measurement()

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