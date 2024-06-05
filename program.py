# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def printing():
    @cudaq.kernel()
    def kernel():
        qubits = cudaq.qvector(2)
        x(qubits[0])
        x(qubits[1])
        cnot(qubits[0], qubits[1])

    print("*** DRAW")
    print(cudaq.draw(kernel))

    print("*** MLIR")
    print(kernel)

    print("*** TO_QIR")
    print(cudaq.to_qir(kernel))


    print("*** TO_QIR with profile")
    print(cudaq.to_qir(kernel, profile="qir-base"))


    # New - convert to qasm
    # TODO



printing()