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
import pytest


def test_issue_4897_corrected():
    # using 1 induction variable
    @cudaq.kernel
    def rotate_each(qubits: cudaq.qview, angles: list[float]):
        for i in range(qubits.size()):
            ry(angles[i], qubits[i])

    @cudaq.kernel
    def undo_rotate_each(qubits: cudaq.qview, angles: list[float]):
        index = qubits.size() - 1
        for i in range(qubits.size()):
            ry(-angles[index], qubits[qubits.size() - 1 - i])
            index -= 1

    @cudaq.kernel
    def roundtrip_autogen(angles: list[float]):
        qubits = cudaq.qvector(3)
        rotate_each(qubits, angles)
        cudaq.adjoint(rotate_each, qubits, angles)

    @cudaq.kernel
    def roundtrip_manual(angles: list[float]):
        qubits = cudaq.qvector(3)
        rotate_each(qubits, angles)
        undo_rotate_each(qubits, angles)

    angles = [0.3, 0.5, 0.7]

    manual = np.asarray(cudaq.get_state(roundtrip_manual, angles))
    print(f"hand-written undo:        |<000|state>| = {abs(manual[0]):.16f}")
    auto = np.asarray(cudaq.get_state(roundtrip_autogen, angles))
    print(f"cudaq.adjoint roundtrip:  |<000|state>| = {abs(auto[0]):.16f}")

    assert math.isclose(manual[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)
    assert math.isclose(auto[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)


skipForNow = pytest.mark.skipif(True, "NYI: compiler does not merge inductions")


@skipForNow
def test_issue_4897_higher_expectations():
    # This test is expecting the compiler to perform an analysis and determine
    # that the variable `index` is a secondary induction.  The compiler does not
    # perform that analysis yet, so this test will fail.  Added note:
    # https://github.com/NVIDIA/cuda-quantum/issues/4897#issuecomment-4939422357
    @cudaq.kernel
    def rotate_each(qubits: cudaq.qview, angles: list[float]):
        index = 0
        for i in range(qubits.size()):
            ry(angles[index], qubits[i])
            index += 1

    @cudaq.kernel
    def undo_rotate_each(qubits: cudaq.qview, angles: list[float]):
        index = qubits.size() - 1
        for i in range(qubits.size()):
            ry(-angles[index], qubits[qubits.size() - 1 - i])
            index -= 1

    @cudaq.kernel
    def roundtrip_autogen(angles: list[float]):
        qubits = cudaq.qvector(3)
        rotate_each(qubits, angles)
        cudaq.adjoint(rotate_each, qubits, angles)

    @cudaq.kernel
    def roundtrip_manual(angles: list[float]):
        qubits = cudaq.qvector(3)
        rotate_each(qubits, angles)
        undo_rotate_each(qubits, angles)

    angles = [0.3, 0.5, 0.7]

    manual = np.asarray(cudaq.get_state(roundtrip_manual, angles))
    print(f"hand-written undo:        |<000|state>| = {abs(manual[0]):.16f}")
    auto = np.asarray(cudaq.get_state(roundtrip_autogen, angles))
    print(f"cudaq.adjoint roundtrip:  |<000|state>| = {abs(auto[0]):.16f}")

    assert math.isclose(manual[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)
    assert math.isclose(auto[0].real, 1.0, rel_tol=1e-9, abs_tol=0.0)
