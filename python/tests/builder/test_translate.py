# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math
import re
import numpy as np
import cudaq


def make_bell_builder():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    return kernel


def test_translate_builder_qir():
    kernel = make_bell_builder()
    qir = cudaq.translate(kernel, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 2)" in qir
    assert "__quantum__qis__h" in qir
    assert "__quantum__qis__x__ctl" in qir


def test_translate_builder_openqasm():
    kernel = make_bell_builder()
    asm = cudaq.translate(kernel, format="openqasm2")
    assert "OPENQASM 2.0;" in asm
    assert "qreg var0[2];" in asm
    assert "h var0[0];" in asm
    assert "cx var0[0], var0[1];" in asm


def test_translate_builder_qir_base():
    kernel = make_bell_builder()
    qir = cudaq.translate(kernel, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_builder_qir_adaptive():
    kernel = make_bell_builder()
    qir = cudaq.translate(kernel, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir


def test_translate_builder_with_params_qir():
    kernel, n = cudaq.make_kernel(int)
    q = kernel.qalloc(n)
    kernel.h(q[0])
    qir = cudaq.translate(kernel, 3, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 3)" in qir
    assert "__quantum__qis__h" in qir


def test_draw_builder():
    kernel = make_bell_builder()
    drawing = cudaq.draw(kernel)
    expected_str = '''     ╭───╮     
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯
'''
    assert drawing == expected_str


def test_get_unitary_builder():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    kernel.h(q[0])
    unitary = cudaq.get_unitary(kernel)
    expected = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]],
                                           dtype=np.complex128)
    np.testing.assert_allclose(unitary, expected, atol=1e-12)


def _adjoint_openqasm(build_inner):
    # build_inner() returns the sub-kernel to embed via adjoint().
    # Builds a one-qubit outer kernel, applies the adjoint, and translates to OpenQASM 2.
    outer = cudaq.make_kernel()
    reg = outer.qalloc(1)
    outer.adjoint(build_inner(), reg[0])
    return cudaq.translate(outer, format="openqasm2")


def _parse_qasm2_rotation_ops(qasm):
    # Return list of (gate_name, angle) for every rx/ry/rz in the circuit
    # body (after the qreg declaration, skipping gate definition blocks).
    body = qasm[qasm.index('qreg'):]
    return [(m.group(1), float(m.group(2)))
            for m in re.finditer(r'\b(r[xyz])\(([^)]+)\)', body)]


def test_translate_builder_adjoint_s_openqasm():
    # adjoint(s) should produce rz(-pi/2), the s-dagger equivalent.
    def inner():
        k, q = cudaq.make_kernel(cudaq.qubit)
        k.s(q)
        return k

    asm = _adjoint_openqasm(inner)
    assert "OPENQASM 2.0;" in asm
    ops = _parse_qasm2_rotation_ops(asm)
    assert len(ops) == 1
    gate, angle = ops[0]
    assert gate == "rz"
    assert math.isclose(angle, -math.pi / 2, rel_tol=1e-5)


def test_translate_builder_adjoint_rx_openqasm():
    # adjoint(rx(pi/3)) should negate the angle to -pi/3.
    def inner():
        k, q = cudaq.make_kernel(cudaq.qubit)
        k.rx(math.pi / 3, q)
        return k

    asm = _adjoint_openqasm(inner)
    assert "OPENQASM 2.0;" in asm
    ops = _parse_qasm2_rotation_ops(asm)
    assert len(ops) == 1
    gate, angle = ops[0]
    assert gate == "rx"
    assert math.isclose(angle, -math.pi / 3, rel_tol=1e-5)


def test_translate_builder_adjoint_ry_openqasm():
    # adjoint(ry(pi/4)) should negate the angle to -pi/4.
    def inner():
        k, q = cudaq.make_kernel(cudaq.qubit)
        k.ry(math.pi / 4, q)
        return k

    asm = _adjoint_openqasm(inner)
    assert "OPENQASM 2.0;" in asm
    ops = _parse_qasm2_rotation_ops(asm)
    assert len(ops) == 1
    gate, angle = ops[0]
    assert gate == "ry"
    assert math.isclose(angle, -math.pi / 4, rel_tol=1e-5)


def test_translate_builder_adjoint_rz_openqasm():
    # adjoint(rz(pi/6)) should negate the angle to -pi/6.
    def inner():
        k, q = cudaq.make_kernel(cudaq.qubit)
        k.rz(math.pi / 6, q)
        return k

    asm = _adjoint_openqasm(inner)
    assert "OPENQASM 2.0;" in asm
    ops = _parse_qasm2_rotation_ops(asm)
    assert len(ops) == 1
    gate, angle = ops[0]
    assert gate == "rz"
    assert math.isclose(angle, -math.pi / 6, rel_tol=1e-5)


def test_translate_builder_adjoint_t_openqasm():
    # adjoint(t) should produce rz(-pi/4), the t-dagger equivalent.
    def inner():
        k, q = cudaq.make_kernel(cudaq.qubit)
        k.t(q)
        return k

    asm = _adjoint_openqasm(inner)
    assert "OPENQASM 2.0;" in asm
    ops = _parse_qasm2_rotation_ops(asm)
    assert len(ops) == 1
    gate, angle = ops[0]
    assert gate == "rz"
    assert math.isclose(angle, -math.pi / 4, rel_tol=1e-5)
