# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq
import math


def test_issue_4898():

    @cudaq.kernel
    def conjugated_rotations(qubits: cudaq.qview):
        for layer in range(1, qubits.size()):
            for branch in range(1 << layer):
                for bit in range(layer):
                    if ((branch >> bit) & 1) == 0:
                        x(qubits[layer - 1 - bit])
                ry(0.1, qubits[layer])
                for bit in range(layer):
                    if ((branch >> bit) & 1) == 0:
                        x(qubits[layer - 1 - bit])

    @cudaq.kernel
    def undo_conjugated_rotations(qubits: cudaq.qview):
        n = qubits.size()
        for reverse_layer in range(1, n):
            layer = n - reverse_layer
            branches = 1 << layer
            for reverse_branch in range(branches):
                branch = branches - 1 - reverse_branch
                for bit in range(layer):
                    if ((branch >> bit) & 1) == 0:
                        x(qubits[layer - 1 - bit])
                ry(-0.1, qubits[layer])
                for bit in range(layer):
                    if ((branch >> bit) & 1) == 0:
                        x(qubits[layer - 1 - bit])

    @cudaq.kernel
    def roundtrip_manual():
        qubits = cudaq.qvector(3)
        conjugated_rotations(qubits)
        undo_conjugated_rotations(qubits)

    @cudaq.kernel
    def roundtrip_autogen():
        qubits = cudaq.qvector(3)
        conjugated_rotations(qubits)
        cudaq.adjoint(conjugated_rotations, qubits)

    manual = np.asarray(cudaq.get_state(roundtrip_manual))
    print(f"hand-written undo:       |<000|state>| = {abs(manual[0]):.16f}")
    #auto = np.asarray(cudaq.get_state(roundtrip_autogen))
    #print(f"cudaq.adjoint roundtrip: |<000|state>| = {abs(auto[0]):.16f}")

    assert math.isclose(manual[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)
    #assert math.isclose(auto[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)
