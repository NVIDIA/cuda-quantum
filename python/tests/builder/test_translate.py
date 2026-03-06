# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq


def test_translate_builder_qir():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    qir = cudaq.translate(kernel, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_builder_openqasm():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    asm = cudaq.translate(kernel, format="openqasm2")
    assert "qreg" in asm


def test_translate_builder_qir_base():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    qir = cudaq.translate(kernel, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_builder_qir_adaptive():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    qir = cudaq.translate(kernel, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir


def test_translate_builder_with_params_qir():
    kernel, n = cudaq.make_kernel(int)
    q = kernel.qalloc(n)
    kernel.h(q[0])
    qir = cudaq.translate(kernel, 3, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 3)" in qir


def test_translate_builder_invalid_extra_args():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.mz(q)
    with pytest.raises(RuntimeError) as e:
        cudaq.translate(kernel, 5, format="qir")
    assert "Invalid number of argu" in repr(e)


def test_translate_builder_invalid_type():
    with pytest.raises(RuntimeError) as e:
        cudaq.translate("not_a_kernel", format="qir")
    assert "kernel is invalid type" in repr(e)


# Smoke tests for other APIs sharing mk_decorator path
def test_draw_builder():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    drawing = cudaq.draw(kernel)
    assert "h" in drawing
    assert "x" in drawing


def test_get_unitary_builder():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    kernel.h(q[0])
    unitary = cudaq.get_unitary(kernel)
    assert unitary.shape == (2, 2)
    assert np.isclose(abs(unitary[0, 0]), 1.0 / np.sqrt(2.0), atol=1e-6)
