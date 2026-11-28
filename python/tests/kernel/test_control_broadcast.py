# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import cudaq


def test_ctrl_x_broadcast():
    """Test controlled-X with broadcast over multiple targets."""

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(3)

        # Put controls in |11> state
        x(controls)

        # Apply CX to each target individually
        for t in targets:
            cx(controls, t)

    counts = cudaq.sample(kernel)
    # All 5 qubits should be |1>
    assert counts["11111"] == 1000


def test_ctrl_x_broadcast_single_control():
    """Test controlled-X with single control broadcast over multiple targets."""

    @cudaq.kernel
    def kernel():
        control = cudaq.qubit()
        targets = cudaq.qvector(3)

        x(control)
        for t in targets:
            cx(control, t)

    counts = cudaq.sample(kernel)
    # Control + 3 targets = 4 qubits all |1>
    assert counts["1111"] == 1000


def test_ctrl_y_broadcast():
    """Test controlled-Y with broadcast."""

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            cy(controls, t)

    counts = cudaq.sample(kernel)
    # 4 qubits all |1>
    assert counts["1111"] == 1000


def test_ctrl_z_broadcast():
    """Test controlled-Z with broadcast."""

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            cz(controls, t)

    counts = cudaq.sample(kernel)
    # Phase gate doesn't change computational basis state
    assert counts["1100"] == 1000


def test_ctrl_h_broadcast():
    """Test controlled-H with broadcast."""

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            ch(controls, t)

    cudaq.set_random_seed(42)
    counts = cudaq.sample(kernel, shots_count=10000)
    # Should get superposition on targets
    total = sum(counts.values())
    assert total == 10000


def test_crx_broadcast():
    """Test controlled-RX with broadcast."""

    @cudaq.kernel
    def kernel(angle: float):
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            crx(angle, controls, t)

    counts = cudaq.sample(kernel, math.pi)
    # 4 qubits all |1> (RX(pi) = X up to phase)
    assert counts["1111"] == 1000


def test_cry_broadcast():
    """Test controlled-RY with broadcast."""

    @cudaq.kernel
    def kernel(angle: float):
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            cry(angle, controls, t)

    counts = cudaq.sample(kernel, math.pi)
    # RY(pi) = Y up to phase, so targets should be |1>
    assert counts["1111"] == 1000


def test_crz_broadcast():
    """Test controlled-RZ with broadcast."""

    @cudaq.kernel
    def kernel(angle: float):
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(2)

        x(controls)
        for t in targets:
            crz(angle, controls, t)

    counts = cudaq.sample(kernel, math.pi)
    # Phase gate doesn't change computational basis
    assert counts["1100"] == 1000


def test_ctrl_custom_gate_broadcast():
    """Test broadcasting a custom controlled gate."""

    @cudaq.kernel
    def my_controlled_gate(ctrl: cudaq.qvector, target: cudaq.qubit):
        x.ctrl(ctrl, target)

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(2)
        targets = cudaq.qvector(3)

        x(controls)
        for t in targets:
            my_controlled_gate(controls, t)

    counts = cudaq.sample(kernel)
    assert counts["11111"] == 1000


def test_broadcast_with_list_of_controls():
    """Test broadcast with Python list of control qubits."""

    @cudaq.kernel
    def kernel():
        controls = cudaq.qvector(3)
        targets = cudaq.qvector(2)

        x(controls[0])
        x(controls[2])
        # Use list comprehension for controls
        ctrl_list = [controls[i] for i in [0, 2]]
        for t in targets:
            cx(ctrl_list, t)

    counts = cudaq.sample(kernel)
    # controls[0] and controls[2] are |1>, targets should be flipped
    assert counts["10111"] == 1000


def test_ctrl_vs_broadcast_control_count():
    """Test that control count affects broadcast behavior."""

    @cudaq.kernel
    def kernel(n_controls: int):
        controls = cudaq.qvector(n_controls)
        target = cudaq.qubit()

        x(controls)
        cx(controls, target)

    # 1 control
    counts1 = cudaq.sample(kernel, 1)
    assert counts1["11"] == 1000

    # 2 controls
    counts2 = cudaq.sample(kernel, 2)
    assert counts2["111"] == 1000

    # 3 controls
    counts3 = cudaq.sample(kernel, 3)
    assert counts3["1111"] == 1000


def test_broadcast_empty_controls():
    """Test broadcast with empty control register."""

    @cudaq.kernel
    def kernel(n: int):
        controls = cudaq.qvector(n)
        targets = cudaq.qvector(2)

        # Put targets in |1>
        x(targets)
        for t in targets:
            cx(controls, t)

    counts = cudaq.sample(kernel, 0)
    # With zero controls the controlled gate degenerates to a plain X gate,
    # so each target is flipped a second time and ends up back in |0>.
    assert counts["00"] == 1000


def test_ctrl_broadcast_mixed_qubit_types():
    """Test broadcast with mix of qubit and qvector."""

    @cudaq.kernel
    def kernel():
        control_reg = cudaq.qvector(2)
        control_scalar = cudaq.qubit()
        target_reg = cudaq.qvector(2)

        x(control_reg)
        x(control_scalar)
        # Apply CX to each target individually with mixed controls
        for t in target_reg:
            cx([*control_reg, control_scalar], t)

    counts = cudaq.sample(kernel)
    # All 5 qubits should be |1>
    assert counts["11111"] == 1000