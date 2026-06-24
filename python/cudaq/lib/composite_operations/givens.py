# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq


@cudaq.kernel
def givens(theta: float, qubitA: cudaq.qubit, qubitB: cudaq.qubit):
    qubits = [qubitA, qubitB]
    exp_pauli(-.5 * theta, qubits, 'YX')
    exp_pauli(.5 * theta, qubits, 'XY')
