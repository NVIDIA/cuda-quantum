# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# `fmt: off`
#[Begin Docs]
import cudaq
import numpy as np

# Create and test a custom CNOT operation.
cudaq.register_operation("my_cnot", np.array([1, 0, 0, 0,
                                              0, 1, 0, 0,
                                              0, 0, 0, 1,
                                              0, 0, 1, 0]))

@cudaq.kernel
def bell_pair():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    my_cnot(qubits[0], qubits[1]) # `my_cnot(control, target)`


cudaq.sample(bell_pair).dump() # prints { 11:500 00:500 } (exact numbers will be random)


# Construct a custom unitary matrix for X on the first qubit and Y
# on the second qubit.
X = np.array([[0,  1 ], [1 , 0]])
Y = np.array([[0, -1j], [1j, 0]])
XY = np.kron(X, Y)

# Register the custom operation
cudaq.register_operation("my_XY", XY)

@cudaq.kernel
def custom_xy_test():
    qubits = cudaq.qvector(2)
    my_XY(qubits[0], qubits[1])
    y(qubits[1]) # undo the prior Y gate on qubit 1


cudaq.sample(custom_xy_test).dump() # prints { 10:1000 }
#[End Docs]
# `fmt: on`
