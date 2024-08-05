# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np

import cudaq
from cudaq import spin


def test_ctrl_x():
    """Tests the accuracy of the overloads for the controlled-X gate."""

    @cudaq.kernel
    def ctrl_x_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        cx(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        cx(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        cx([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        cx(controls, qubits[4])

    counts = cudaq.sample(ctrl_x_kernel)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_ctrl_y():
    """Tests the accuracy of the overloads for the controlled-Y gate."""

    @cudaq.kernel
    def ctrl_y_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        cy(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        cy(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        cy([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        cy(controls, qubits[4])

    counts = cudaq.sample(ctrl_y_kernel)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_ctrl_z():
    """Tests the accuracy of the overloads for the controlled-Z gate."""

    @cudaq.kernel
    def ctrl_z_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        cz(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        cz(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        cz([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        cz(controls, qubits[4])

    counts = cudaq.sample(ctrl_z_kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_ctrl_h():
    """Tests the accuracy of the overloads for the controlled-H gate."""
    cudaq.set_random_seed(4)

    @cudaq.kernel
    def ctrl_h_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        ch(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        ch(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        ch([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        ch(controls, qubits[4])

    counts1 = cudaq.sample(ctrl_h_kernel)
    print(counts1)

    # Our first two qubits remain untouched, while `qubits[2]` is rotated
    # to 1, and `qubits[3]` receives a Hadamard. This results in a nearly 50/50
    # split of measurements on `qubits[3]` between 0 and 1.
    # The controlled Hadamard on `qubits[4]` also results in a 50/50 split of its
    # measurements between 0 and 1.
    assert counts1["0011011"] >= 225 and counts1["0011011"] <= 275
    assert counts1["0011111"] >= 225 and counts1["0011111"] <= 275
    assert counts1["0010011"] >= 225 and counts1["0010011"] <= 275
    assert counts1["0010111"] >= 225 and counts1["0010111"] <= 275
    assert counts1["0011011"] + counts1["0011111"] + counts1[
        "0010011"] + counts1["0010111"] == 1000

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)
        ch(qubits[0], qubits[1])
        x(qubits[2])
        ch(qubits[2], qubits[3])
        ch([controls[0], controls[1]], qubits[4])
        x(controls)
        ch(controls, qubits[4])
        h(qubits[3])
        h(qubits[4])

    counts2 = cudaq.sample(kernel)
    print(counts2)
    assert counts2["0010011"] == 1000


def test_ctrl_s():
    """Tests the accuracy of the overloads for the controlled-S gate."""

    @cudaq.kernel
    def ctrl_s_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        cs(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        cs(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        cz([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        cz(controls, qubits[4])

    counts = cudaq.sample(ctrl_s_kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_ctrl_t():
    """Tests the accuracy of the overloads for the controlled-T gate."""

    @cudaq.kernel
    def ctrl_t_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        ct(qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        x(qubits[2])
        ct(qubits[2], qubits[3])

        # Overload 2: list of control qubits, one target.
        # The last qubit in `qubits` is still in 0-state, as is our
        # `control` register. A controlled gate between them should
        # have no impact.
        cz([controls[0], controls[1]], qubits[4])

        # Overload 3: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        cz(controls, qubits[4])

    counts = cudaq.sample(ctrl_t_kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_cr1_gate():
    """Tests the accuracy of the overloads for the controlled-r1 gate."""

    @cudaq.kernel
    def cr1_kernel(angle: float):
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        # Testing the `QuakeValue` parameter overload.
        cr1(angle, qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        # Testing the `float` parameter overload.
        x(qubits[2])
        cr1(angle, qubits[2], qubits[3])

        # Overload 2: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        # `QuakeValue` parameter.
        cr1(angle, controls, qubits[4])
        # `float` parameter that we set = 0.0 so it doesn't impact state.
        cr1(0.0, controls, qubits[4])

    counts = cudaq.sample(cr1_kernel, np.pi)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_crx_gate():
    """Tests the accuracy of the overloads for the controlled-rx gate."""

    @cudaq.kernel
    def crx_kernel(angle: float):
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        # Testing the `QuakeValue` parameter overload.
        crx(angle, qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        # Testing the `float` parameter overload.
        x(qubits[2])
        crx(angle, qubits[2], qubits[3])

        # Overload 2: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        # `QuakeValue` parameter.
        crx(angle, controls, qubits[4])
        # `float` parameter that we set = 0.0 so it doesn't impact state.
        crx(0.0, controls, qubits[4])

    counts = cudaq.sample(crx_kernel, np.pi)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_cry_gate():
    """Tests the accuracy of the overloads for the controlled-ry gate."""

    @cudaq.kernel
    def cry_kernel(angle: float):
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # Overload 1: one control qubit, one target.
        # 2-qubit controlled operation with control in 0-state.
        # Testing the `QuakeValue` parameter overload.
        cry(angle, qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        # Testing the `float` parameter overload.
        x(qubits[2])
        cry(angle, qubits[2], qubits[3])

        # Overload 2: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        # `QuakeValue` parameter.
        cry(angle, controls, qubits[4])
        # `float` parameter that we set = 0.0 so it doesn't impact state.
        cry(0.0, controls, qubits[4])

    counts = cudaq.sample(cry_kernel, np.pi)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_crz_gate():
    """Tests the accuracy of the overloads for the controlled-rz gate."""

    @cudaq.kernel
    def crz_kernel(angle: float):
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)

        # 2-qubit controlled operation with control in 0-state.
        # Testing the `QuakeValue` parameter overload.
        crz(angle, qubits[0], qubits[1])
        # 2-qubit controlled operation with control in 1-state.
        # Testing the `float` parameter overload.
        x(qubits[2])
        crz(angle, qubits[2], qubits[3])

        # Overload 2: register of control qubits, one target.
        # Now place `control` register in 1-state and perform another
        # controlled operation on the final qubit.
        x(controls)
        # `QuakeValue` parameter.
        crz(angle, controls, qubits[4])
        # `float` parameter that we set = 0.0 so it doesn't impact state.
        crz(0.0, controls, qubits[4])

    counts = cudaq.sample(crz_kernel, np.pi)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000
