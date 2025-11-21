# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest
import numpy as np
import cudaq


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_builtin_controlled_gates():

    @cudaq.kernel
    def control_simple_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        cx(~c, q)
        cx(c, q)

    counts = cudaq.sample(control_simple_gate)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_simple_gate():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[3])
        cx(c[0], ~c[1], ~c[2], c[3], q)

    counts = cudaq.sample(multi_control_simple_gate)
    assert counts["10011"] == 1000

    @cudaq.kernel
    def control_rotation_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        cry(np.pi, ~c, q)
        cry(np.pi, c, q)

    counts = cudaq.sample(control_rotation_gate)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_rotation_gate():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[3])
        cry(np.pi, c[0], ~c[1], ~c[2], c[3], q)

    counts = cudaq.sample(multi_control_rotation_gate)
    assert counts["10011"] == 1000

    # Note: u3, swap, and exp_pauli do not have a built-in
    # c<gatename> version at the time of writing this.


def test_ctrl_attribute():

    @cudaq.kernel
    def control_simple_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        x.ctrl(~c, q)
        x.ctrl(c, q)

    counts = cudaq.sample(control_simple_gate)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_simple_gate():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[3])
        x.ctrl(c[0], ~c[1], ~c[2], c[3], q)

    counts = cudaq.sample(multi_control_simple_gate)
    assert counts["10011"] == 1000

    @cudaq.kernel
    def control_rotation_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        ry.ctrl(np.pi, ~c, q)
        ry.ctrl(np.pi, c, q)

    counts = cudaq.sample(control_rotation_gate)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_rotation_gate():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[3])
        ry.ctrl(np.pi, c[0], ~c[1], ~c[2], c[3], q)

    counts = cudaq.sample(multi_control_rotation_gate)
    assert counts["10011"] == 1000

    @cudaq.kernel
    def control_swap_gate():
        c, q1, q2 = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q1)
        swap.ctrl(~c, q1, q2)
        swap.ctrl(c, q1, q2)

    counts = cudaq.sample(control_swap_gate)
    assert counts["001"] == 1000

    @cudaq.kernel
    def multi_control_swap_gate():
        c, q1, q2 = cudaq.qvector(4), cudaq.qubit(), cudaq.qubit()
        x(q1)
        x(c[0], c[3])
        swap.ctrl(c[0], ~c[1], ~c[2], c[3], q1, q2)

    counts = cudaq.sample(multi_control_swap_gate)
    assert counts["100101"] == 1000

    @cudaq.kernel
    def control_u3_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        t, p, l = np.pi, 0., 0.
        u3.ctrl(t, p, l, ~c, q)
        u3.ctrl(t, p, l, c, q)

    counts = cudaq.sample(control_u3_gate)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_u3_gate():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[2])
        t, p, l = np.pi, 0., 0.
        u3.ctrl(t, p, l, c[0], ~c[1], c[2], ~c[3], q)

    counts = cudaq.sample(multi_control_u3_gate)
    assert counts["10101"] == 1000

    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def control_registered_operation():
        c, q = cudaq.qubit(), cudaq.qubit()
        custom_x.ctrl(~c, q)
        custom_x.ctrl(c, q)

    counts = cudaq.sample(control_registered_operation)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_registered_operation():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[2])
        custom_x.ctrl(c[0], ~c[1], c[2], ~c[3], q)

    counts = cudaq.sample(multi_control_registered_operation)
    assert counts["10101"] == 1000


def test_cudaq_control():

    @cudaq.kernel
    def custom_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def control_kernel():
        c, q = cudaq.qubit(), cudaq.qubit()
        cudaq.control(custom_x, ~c, q)
        cudaq.control(custom_x, c, q)

    counts = cudaq.sample(control_kernel)
    assert counts["01"] == 1000

    @cudaq.kernel
    def multi_control_kernel():
        c, q = cudaq.qvector(4), cudaq.qubit()
        x(c[0], c[3])
        cudaq.control(custom_x, c[0], ~c[1], ~c[2], c[3], q)

    counts = cudaq.sample(multi_control_kernel)
    assert counts["10011"] == 1000

    # Note: calling cudaq.control on a registered operation
    # or on a built-in gate is not supported at the time of writing this


def test_unsupported_calls():

    # If we add support for any of these, add the corresponding
    # tests above and remove the notes.

    @cudaq.kernel
    def cu3_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        t, p, l = 0., 0., np.pi
        cu3(t, p, l, ~c, q)
        cu3(t, p, l, c, q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(cu3_gate)
    assert "unhandled function call - cu3" in str(e.value)

    @cudaq.kernel
    def cswap_gate():
        c, q1, q2 = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q1)
        cswap(~c, q1, q2)
        cswap(c, q1, q2)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(cswap_gate)
    assert "unhandled function call - cswap" in str(e.value)

    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def control_registered_operation():
        c, q = cudaq.qubit(), cudaq.qubit()
        cudaq.control(custom_x, ~c, q)
        cudaq.control(custom_x, c, q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(control_registered_operation)
    assert "calling cudaq.control or cudaq.adjoint on a globally registered operation is not supported" in str(
        e.value)

    @cudaq.kernel
    def control_rotation_gate():
        c, q = cudaq.qubit(), cudaq.qubit()
        cudaq.control(ry, ~c, np.pi, q)
        cudaq.control(ry, c, np.pi, q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(control_rotation_gate)
    assert "calling cudaq.control or cudaq.adjoint on a built-in gate is not supported" in str(
        e.value)

    @cudaq.kernel
    def control_simple_gate():
        c, q = cudaq.qvector(3), cudaq.qubit()
        cx(~c, q)
        x(c[0])
        cx(c, q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(control_simple_gate)
    assert "unary operator ~ is only supported for values of type qubit" in str(
        e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
