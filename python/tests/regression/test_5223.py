# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import numpy as np

import cudaq


@cudaq.kernel
def qft(qubits: cudaq.qview):
    qubit_count = len(qubits)
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])


def test_issue_5223_nested_loop_adjoint():
    # `j` is local to the outer loop's body. Allocating it in the function entry
    # block instead makes it a loop-carried value that nothing reads, and the
    # adjoint then fails to synthesize with "control-flow def-use not
    # reversible".

    @cudaq.kernel
    def roundtrip():
        q = cudaq.qvector(3)
        qft(q)
        cudaq.adjoint(qft, q)

    state = np.asarray(cudaq.get_state(roundtrip))
    assert math.isclose(state[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)


def test_issue_5223_adjoint_is_drawable():
    # The original bug where the adjoint alone fails to compile at all.

    @cudaq.kernel
    def entry():
        q = cudaq.qvector(3)
        cudaq.adjoint(qft, q)

    drawing = cudaq.draw(entry)
    assert "h" in drawing
    assert "r1" in drawing
