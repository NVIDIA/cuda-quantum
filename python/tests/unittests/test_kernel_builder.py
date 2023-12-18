# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is responsible for testing the accuracy of gates within
# the kernel builder.

import pytest
import random
import numpy as np

import cudaq
from cudaq import spin


def test_sdg_0_state():
    """Tests the adjoint S-gate on a qubit starting in the 0-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_sdg_1_state():
    """Tests the adjoint S-gate on a qubit starting in the 1-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superposition.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_sdg_0_state_negate():
    """Tests that the sdg and s gates cancel each other out."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Rotate back around by pi. Will use two gates here, but will
    # also test with a plain Z-gate in the 1-state test.
    kernel.s(qubit)
    kernel.s(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Qubit should still be in 0 state.
    assert counts["0"] == 1000


def test_sdg_1_state_negate():
    """Tests that the sdg and s gates cancel each other out."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superpositoin.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Rotate back by pi.
    kernel.z(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Qubit should still be in 1 state.
    assert counts["1"] == 1000


def test_tdg_0_state():
    """Tests the adjoint T-gate on a qubit starting in the 0-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_tdg_1_state():
    """Tests the adjoint T-gate on a qubit starting in the 1-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superposition.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_tdg_0_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Rotate back by pi.
    kernel.z(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Qubit should remain in 0-state.
    assert counts["0"] == 1000


def test_tdg_1_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superposition.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Rotate back by pi.
    kernel.t(qubit)
    kernel.t(qubit)
    kernel.t(qubit)
    kernel.t(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    print(counts)

    # Qubit should remain in 1-state.
    assert counts["1"] == 1000


def test_rotation_multi_target():
    """
    Tests the accuracy of rotation gates when applied to
    entire qregs.
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)

    # Start in the |1> state.
    kernel.x(qubits)

    # Rotate qubits back to the |0> state.
    kernel.rx(np.pi, qubits)
    # Phase rotation.
    kernel.r1(-np.pi, qubits)
    # Rotate back to |1> state.
    kernel.ry(np.pi, qubits)
    # Phase rotation.
    kernel.rz(np.pi, qubits)

    counts = cudaq.sample(kernel)
    assert counts["111"] == 1000


def test_ctrl_x():
    """Tests the accuracy of the overloads for the controlled-X gate."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.cx(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.cx(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.cx([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.cx(controls, qubits[4])

    counts = cudaq.sample(kernel)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_ctrl_y():
    """Tests the accuracy of the overloads for the controlled-Y gate."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.cy(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.cy(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.cy([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.cy(controls, qubits[4])

    counts = cudaq.sample(kernel)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_ctrl_z():
    """Tests the accuracy of the overloads for the controlled-Z gate."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.cz(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.cz(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.cz([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.cz(controls, qubits[4])

    counts = cudaq.sample(kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_ctrl_h():
    """Tests the accuracy of the overloads for the controlled-H gate."""
    cudaq.set_random_seed(4)
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.ch(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.ch(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.ch([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.ch(controls, qubits[4])
    counts1 = cudaq.sample(kernel)
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
    assert counts1["0011011"] + counts1["0011111"] + counts1["0010011"] + counts1["0010111"] == 1000

    kernel.h(qubits[3])
    kernel.h(qubits[4])
    counts2 = cudaq.sample(kernel)
    print(counts2)
    assert counts2["0010011"] == 1000


def test_ctrl_s():
    """Tests the accuracy of the overloads for the controlled-S gate."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.cs(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.cs(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.cz([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.cz(controls, qubits[4])

    counts = cudaq.sample(kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_ctrl_t():
    """Tests the accuracy of the overloads for the controlled-T gate."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    kernel.ct(qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    kernel.x(qubits[2])
    kernel.ct(qubits[2], qubits[3])

    # Overload 2: list of control qubits, one target.
    # The last qubit in `qubits` is still in 0-state, as is our
    # `control` register. A controlled gate between them should
    # have no impact.
    kernel.cz([controls[0], controls[1]], qubits[4])

    # Overload 3: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    kernel.cz(controls, qubits[4])

    counts = cudaq.sample(kernel)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_cr1_gate():
    """Tests the accuracy of the overloads for the controlled-r1 gate."""
    kernel, angle = cudaq.make_kernel(float)
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)
    angle_value = np.pi

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    # Testing the `QuakeValue` parameter overload.
    kernel.cr1(angle, qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    # Testing the `float` parameter overload.
    kernel.x(qubits[2])
    kernel.cr1(angle_value, qubits[2], qubits[3])

    # Overload 2: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    # `QuakeValue` parameter.
    kernel.cr1(angle, controls, qubits[4])
    # `float` parameter that we set = 0.0 so it doesn't impact state.
    kernel.cr1(0.0, controls, qubits[4])

    counts = cudaq.sample(kernel, angle_value)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


def test_crx_gate():
    """Tests the accuracy of the overloads for the controlled-rx gate."""
    kernel, angle = cudaq.make_kernel(float)
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)
    angle_value = np.pi

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    # Testing the `QuakeValue` parameter overload.
    kernel.crx(angle, qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    # Testing the `float` parameter overload.
    kernel.x(qubits[2])
    kernel.crx(angle_value, qubits[2], qubits[3])

    # Overload 2: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    # `QuakeValue` parameter.
    kernel.crx(angle, controls, qubits[4])
    # `float` parameter that we set = 0.0 so it doesn't impact state.
    kernel.crx(0.0, controls, qubits[4])

    counts = cudaq.sample(kernel, angle_value)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_cry_gate():
    """Tests the accuracy of the overloads for the controlled-ry gate."""
    kernel, angle = cudaq.make_kernel(float)
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)
    angle_value = np.pi

    # Overload 1: one control qubit, one target.
    # 2-qubit controlled operation with control in 0-state.
    # Testing the `QuakeValue` parameter overload.
    kernel.cry(angle, qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    # Testing the `float` parameter overload.
    kernel.x(qubits[2])
    kernel.cry(angle_value, qubits[2], qubits[3])

    # Overload 2: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    # `QuakeValue` parameter.
    kernel.cry(angle, controls, qubits[4])
    # `float` parameter that we set = 0.0 so it doesn't impact state.
    kernel.cry(0.0, controls, qubits[4])

    counts = cudaq.sample(kernel, angle_value)
    print(counts)

    # State of system should now be `|qubits, controls> = |00111 11>`.
    assert counts["0011111"] == 1000


def test_crz_gate():
    """Tests the accuracy of the overloads for the controlled-rz gate."""
    kernel, angle = cudaq.make_kernel(float)
    qubits = kernel.qalloc(5)
    controls = kernel.qalloc(2)
    angle_value = np.pi

    # 2-qubit controlled operation with control in 0-state.
    # Testing the `QuakeValue` parameter overload.
    kernel.crz(angle, qubits[0], qubits[1])
    # 2-qubit controlled operation with control in 1-state.
    # Testing the `float` parameter overload.
    kernel.x(qubits[2])
    kernel.crz(angle_value, qubits[2], qubits[3])

    # Overload 2: register of control qubits, one target.
    # Now place `control` register in 1-state and perform another
    # controlled operation on the final qubit.
    kernel.x(controls)
    # `QuakeValue` parameter.
    kernel.crz(angle, controls, qubits[4])
    # `float` parameter that we set = 0.0 so it doesn't impact state.
    kernel.crz(0.0, controls, qubits[4])

    counts = cudaq.sample(kernel, angle_value)
    print(counts)

    # The phase should not affect the final state of any target qubits,
    # leaving us with the total state: `|qubits, controls> = |00100 11>`.
    assert counts["0010011"] == 1000


@pytest.mark.parametrize("control_count", [1, 2, 3])
def test_cswap_gate_ctrl_list(control_count):
    """Tests the controlled-SWAP operation given a vector of control qubits."""
    kernel = cudaq.make_kernel()
    controls = [kernel.qalloc() for _ in range(control_count)]
    first = kernel.qalloc()
    second = kernel.qalloc()

    kernel.x(first)
    # All controls in the |0> state, no SWAP should occur.
    kernel.cswap(controls, first, second)
    # If we have multiple controls, place a random control qubit
    # in the |1> state. This check ensures that our controlled
    # SWAP's are performed if and only if all controls are in the
    # |1> state.
    if (len(controls) != 1):
        random_index = random.randint(0, control_count - 1)
        kernel.x(controls[random_index])
        # Not all controls in the in |1>, no SWAP.
        kernel.cswap(controls, first, second)
        # Rotate that random control back to |0>.
        kernel.x(controls[random_index])

    # Now place all of the controls in |1>.
    for control in controls:
        kernel.x(control)
    # Should now SWAP our `first` and `second` qubits.
    kernel.cswap(controls, first, second)

    counts = cudaq.sample(kernel)
    print(counts)

    controls_state = "1" * control_count
    want_state = controls_state + "01"
    assert counts[want_state] == 1000


def test_cswap_gate_mixed_ctrls():
    """
    Tests the controlled-SWAP gate given a list of a mix of ctrl
    qubits and registers.
    """
    kernel = cudaq.make_kernel()
    controls_vector = [kernel.qalloc() for _ in range(2)]
    controls_register = kernel.qalloc(2)
    first = kernel.qalloc()
    second = kernel.qalloc()

    # `first` in |1> state.
    kernel.x(first)
    # `controls_register` in |1> state.
    kernel.x(controls_register)

    # `controls_vector` in |0>, no SWAP.
    kernel.cswap(controls_vector, first, second)
    # `controls_register` in |1>, SWAP.
    kernel.cswap(controls_register, first, second)
    # Pass the vector and register as the controls. The vector is still in |0>, so
    # no SWAP.
    kernel.cswap([controls_vector[0], controls_vector[1], controls_register],
                 first, second)
    # Place the vector in |1>, should now get a SWAP.
    kernel.x(controls_vector[0])
    kernel.x(controls_vector[1])
    kernel.cswap([controls_vector[0], controls_vector[1], controls_register],
                 first, second)

    counts = cudaq.sample(kernel)
    print(counts)

    controls_state = "1111"
    # The SWAP's should make the targets end up back in |10>.
    want_state = controls_state + "10"
    assert counts[want_state] == 1000


def test_crx_control_list():
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()
    q3 = kernel.qalloc()

    # Place a subset of controls in 1-state.
    kernel.x(q1)
    kernel.x(q2)

    # Using different orientations of our control qubits
    # to make kernel less trivial.
    # Overload 1: `QuakeValue` parameter. All controls are in |1>,
    # so this should rotate our `target`.
    kernel.crx(value, [q1, q2], target)
    # Overload 2: `float` parameter. `q3` is still in |0>, so this
    # should not rotate our `target`.
    kernel.crx(np.pi, [q3, q2, q1], target)

    print(kernel)

    result = cudaq.sample(kernel, np.pi)
    print(result)

    # Target is still in 1-state, while q1 = q2 = 1, and q3 = 0
    assert result["1110"] == 1000


def test_cry_control_list():
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()
    q3 = kernel.qalloc()

    # Place a subset of controls in 1-state.
    kernel.x(q1)
    kernel.x(q2)

    # Using different orientations of our control qubits
    # to make kernel less trivial.
    # Overload 1: `QuakeValue` parameter. All controls are in |1>,
    # so this should rotate our `target`.
    kernel.cry(value, [q1, q2], target)
    # Overload 2: `float` parameter. `q3` is still in |0>, so this
    # should not rotate our `target`.
    kernel.cry(np.pi, [q3, q2, q1], target)

    result = cudaq.sample(kernel, np.pi)
    print(result)

    # Target is still in 1-state, while q1 = q2 = 1, and q3 = 0
    assert result["1110"] == 1000


def test_crz_control_list():
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()
    q3 = kernel.qalloc()

    # Place controls in 1-state.
    kernel.x(q1)
    kernel.x(q2)
    kernel.x(q3)

    # Hadamard our target.
    kernel.h(target)

    # Overload 1: `QuakeValue` parameter.
    kernel.crz(value, [q1, q2, q3], target)
    # Overload 2: `float` parameter.
    kernel.crz(-np.pi / 2, [q3, q2, q1], target)

    # Another hadamard to our target.
    kernel.h(target)

    result = cudaq.sample(kernel, -np.pi / 2)
    print(result)

    # The phase rotation on our target by -pi should mean
    # we measure it in the 1-state.
    assert result["1111"] == 1000


def test_cr1_control_list():
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()
    q3 = kernel.qalloc()

    # Place controls in 1-state.
    kernel.x(q1)
    kernel.x(q2)
    kernel.x(q3)

    # Hadamard our target.
    kernel.h(target)

    # Overload 1: `QuakeValue` parameter.
    kernel.cr1(value, [q1, q2, q3], target)
    # Overload 2: `float` parameter.
    kernel.cr1(-np.pi / 2, [q3, q2, q1], target)

    # Another hadamard to our target.
    kernel.h(target)

    result = cudaq.sample(kernel, -np.pi / 2)
    print(result)

    # The phase rotation on our target by -pi should mean
    # we measure it in the 1-state.
    assert result["1111"] == 1000


def test_ctrl_rotation_integration():
    """
    Tests more complex controlled rotation kernels, including
    pieces that will only run in quantinuum emulation.
    """
    cudaq.set_random_seed(4)
    cudaq.set_target("quantinuum", emulate=True)

    kernel = cudaq.make_kernel()
    ctrls = kernel.qalloc(4)
    ctrl = kernel.qalloc()
    target = kernel.qalloc()

    # Subset of `ctrls` in |1> state.
    kernel.x(ctrls[0])
    kernel.x(ctrls[1])

    # Multi-controlled rotation with that qreg should have
    # no impact on our target, since not all `ctrls` are |1>.
    kernel.cry(1.0, ctrls, target)

    # Flip the rest of our `ctrls` to |1>.
    kernel.x(ctrls[2])
    kernel.x(ctrls[3])

    # Multi-controlled rotation should now flip our target.
    kernel.crx(np.pi / 4., ctrls, target)

    # Test (1) (only works in emulation): mixed list of veqs and qubits.
    # Has no impact because `ctrl` = |0>
    kernel.crx(1.0, [ctrls, ctrl], target)
    # Test (2): Flip `ctrl` and try again.
    kernel.x(ctrl)
    kernel.crx(np.pi / 4., [ctrls, ctrl], target)

    result = cudaq.sample(kernel)
    print(result)

    # The `target` should be in a 50/50 mix between |0> and |1>.
    extra_mapping_qubits = "0000"
    want_1_state = extra_mapping_qubits + "111111"
    want_0_state = extra_mapping_qubits + "111110"
    assert result[want_1_state] == 505
    assert result[want_0_state] == 495


def test_can_progressively_build():
    """Tests that a kernel can be build progressively."""
    cudaq.reset_target()
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    print(kernel)
    state = cudaq.get_state(kernel)
    assert np.isclose(1. / np.sqrt(2.), state[0].real)
    assert np.isclose(0., state[1].real)
    assert np.isclose(1. / np.sqrt(2.), state[2].real)
    assert np.isclose(0., state[3].real)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts

    # Continue building the kernel
    kernel.cx(q[0], q[1])
    print(kernel)
    state = cudaq.get_state(kernel)
    assert np.isclose(1. / np.sqrt(2.), state[0].real)
    assert np.isclose(0., state[1].real)
    assert np.isclose(0., state[2].real)
    assert np.isclose(1. / np.sqrt(2.), state[3].real)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_from_state():
    cudaq.reset_target()
    state = np.asarray([.70710678, 0., 0., 0.70710678])
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)

    cudaq.from_state(kernel, qubits, state)

    print(kernel)
    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    kernel = cudaq.from_state(state)
    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    state = np.asarray([0., .292786, .956178, 0.])
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    cudaq.from_state(kernel, qubits, state)
    energy = cudaq.observe(kernel, hamiltonian).expectation()
    assert np.isclose(-1.748, energy, 1e-3)

    ss = cudaq.get_state(kernel)
    for i in range(4):
        assert np.isclose(ss[i], state[i], 1e-3)


def test_exp_pauli():
    cudaq.reset_target()
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(4)
    kernel.x(qubits[0])
    kernel.x(qubits[1])
    print(type(.11))
    kernel.exp_pauli(.11, qubits, "XXXY")
    print(kernel)
    h2_data = [
        3, 1, 1, 3, 0.0454063, 0, 2, 0, 0, 0, 0.17028, 0, 0, 0, 2, 0, -0.220041,
        -0, 1, 3, 3, 1, 0.0454063, 0, 0, 0, 0, 0, -0.106477, 0, 0, 2, 0, 0,
        0.17028, 0, 0, 0, 0, 2, -0.220041, -0, 3, 3, 1, 1, -0.0454063, -0, 2, 2,
        0, 0, 0.168336, 0, 2, 0, 2, 0, 0.1202, 0, 0, 2, 0, 2, 0.1202, 0, 2, 0,
        0, 2, 0.165607, 0, 0, 2, 2, 0, 0.165607, 0, 0, 0, 2, 2, 0.174073, 0, 1,
        1, 3, 3, -0.0454063, -0, 15
    ]
    h = cudaq.SpinOperator(h2_data, 4)
    want_exp = cudaq.observe(kernel, h).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)

    kernel, theta = cudaq.make_kernel(float)
    qubits = kernel.qalloc(4)
    kernel.x(qubits[0])
    kernel.x(qubits[1])
    kernel.exp_pauli(theta, qubits, "XXXY")
    want_exp = cudaq.observe(kernel, h, .11).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)


def test_givens_rotation_op():
    cudaq.reset_target()
    angle = 0.2
    c = np.cos(angle)
    s = np.sin(angle)
    test_01 = cudaq.make_kernel()
    qubits_01 = test_01.qalloc(2)
    test_01.x(qubits_01[0])
    test_01.givens_rotation(angle, qubits_01[0], qubits_01[1])
    ss_01 = cudaq.get_state(test_01)
    assert np.isclose(ss_01[1], -s, 1e-3)
    assert np.isclose(ss_01[2], c, 1e-3)

    test_10 = cudaq.make_kernel()
    qubits_10 = test_10.qalloc(2)
    test_10.x(qubits_10[1])
    test_10.givens_rotation(angle, qubits_10[0], qubits_10[1])
    ss_10 = cudaq.get_state(test_10)
    assert np.isclose(ss_10[1], c, 1e-3)
    assert np.isclose(ss_10[2], s, 1e-3)


def test_fermionic_swap_op():
    cudaq.reset_target()
    angle = 0.2
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    test_01 = cudaq.make_kernel()
    qubits_01 = test_01.qalloc(2)
    test_01.x(qubits_01[0])
    test_01.fermionic_swap(angle, qubits_01[0], qubits_01[1])
    ss_01 = cudaq.get_state(test_01)
    val1 = np.abs(ss_01[1] - (-1j * np.exp(1j * angle / 2.0) * s))
    val2 = np.abs(ss_01[2] - (np.exp(1j * angle / 2.0) * c))
    assert np.isclose(val1, 0.0, atol=1e-6)
    assert np.isclose(val2, 0.0, atol=1e-6)

    test_10 = cudaq.make_kernel()
    qubits_10 = test_10.qalloc(2)
    test_10.x(qubits_10[1])
    test_10.fermionic_swap(angle, qubits_10[0], qubits_10[1])
    ss_10 = cudaq.get_state(test_10)
    val3 = np.abs(ss_10[1] - (np.exp(1j * angle / 2.0) * c))
    val4 = np.abs(ss_10[2] - (-1j * np.exp(1j * angle / 2.0) * s))
    assert np.isclose(val3, 0.0, atol=1e-6)
    assert np.isclose(val4, 0.0, atol=1e-6)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
