# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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
