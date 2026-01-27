# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq

import numpy as np


@cudaq.kernel
def bell_pair():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)


@cudaq.kernel
def kernel_loop_params(numQubits: int):
    q = cudaq.qvector(numQubits)
    h(q)
    for i in range(numQubits - 1):
        cx(q[i], q[i + 1])
    for i in range(numQubits):
        mz(q[i])


@cudaq.kernel
def kernel_loop():
    numQubits = 5
    q = cudaq.qvector(numQubits)
    h(q)
    for i in range(4):
        cx(q[i], q[i + 1])
    for i in range(numQubits):
        mz(q[i])


@cudaq.kernel
def kernel_vector():
    c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    q = cudaq.qvector(c)
    mz(q)


@cudaq.kernel
def kernel_with_call():

    def inner():
        q = cudaq.qvector(2)

    inner()


def test_translate_openqasm():
    asm = cudaq.translate(bell_pair, format="openqasm2")
    assert "qreg var0[2];" in asm


def test_translate_openqasm_with_ignored_args():
    with pytest.raises(RuntimeError) as e:
        asm = cudaq.translate(bell_pair, 5, format="openqasm2")
    assert 'Invalid number of argu' in repr(e)


def test_translate_openqasm_loop():
    asm = cudaq.translate(kernel_loop, format="openqasm2")
    assert "qreg var0[5];" in asm


def test_translate_openqasm_vector():
    asm = cudaq.translate(kernel_vector, format="openqasm2")
    assert 'translation failed' in asm


def test_translate_openqasm_with_args():
    with pytest.raises(RuntimeError) as e:
        print(cudaq.translate(kernel_loop_params, 5, format="openqasm2"))
    assert 'Use synthesize before translate' in repr(e)


def test_translate_openqasm_synth():
    synth = cudaq.synthesize(kernel_loop_params, 4)
    asm = cudaq.translate(synth, format="openqasm2")
    assert "measure var0[3] -> var8[0]" in asm


def test_translate_openqasm_call():
    asm = cudaq.translate(kernel_with_call, format="openqasm2")
    assert 'qreg var0[2];' in asm


def test_translate_qir():
    qir = cudaq.translate(bell_pair, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_qir_ignored_args():
    with pytest.raises(RuntimeError) as e:
        qir = cudaq.translate(bell_pair, 5, format="qir")
    assert 'Invalid number of argu' in repr(e)


def test_translate_qir_with_args():
    qir = cudaq.translate(kernel_loop_params, 5, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 5)" in qir


def test_translate_qir_call():
    qir = cudaq.translate(kernel_with_call, format="qir")
    assert "@__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_qir_base():
    qir = cudaq.translate(bell_pair, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_qir_base_ignored_args():
    with pytest.raises(RuntimeError) as e:
        qir = cudaq.translate(bell_pair, 5, format="qir-base")
    assert 'Invalid number of argu' in repr(e)


def test_translate_qir_base_args():
    with pytest.raises(RuntimeError) as e:
        synth = cudaq.synthesize(kernel_loop_params, 5)
        qir = cudaq.translate(synth, 5, format="qir-base")
    assert 'Invalid number of argu' in repr(e)


def test_translate_qir_adaptive():
    qir = cudaq.translate(bell_pair, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir


def test_translate_qir_adaptive_ignored_args():
    with pytest.raises(RuntimeError) as e:
        qir = cudaq.translate(bell_pair, 5, format="qir-adaptive")
    assert 'Invalid number of argu' in repr(e)


def test_translate_qir_adaptive_args():
    with pytest.raises(RuntimeError) as e:
        synth = cudaq.synthesize(kernel_loop_params, 5)
        qir = cudaq.translate(synth, 5, format="qir-adaptive")
    assert 'Invalid number of argu' in repr(e)
