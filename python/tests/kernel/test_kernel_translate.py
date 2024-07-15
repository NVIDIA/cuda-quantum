# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


@cudaq.kernel
def bell_pair():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)


@cudaq.kernel
def kernel(numQubits: int):
    q = cudaq.qvector(numQubits)
    h(q)
    for i in range(numQubits - 1):
        cx(q[i], q[i + 1])
    for i in range(numQubits):
        mz(q[i])


@cudaq.kernel
def kernel_with_call():

    def inner():
        q = cudaq.qvector(2)

    inner()


def test_translate_openqasm():
    asm = cudaq.translate(bell_pair, format="openqasm2")
    assert "qreg var0[2];" in asm


def test_translate_openqasm_with_ignored_args():
    asm = cudaq.translate(bell_pair, 5, format="openqasm2")
    assert "qreg var0[2];" in asm


def test_translate_openqasm_with_args():
    with pytest.raises(RuntimeError) as e:
        print(cudaq.translate(kernel, 5, format="openqasm2"))
    assert 'Cannot translate function with arguments to OpenQASM 2.0.' in repr(
        e)


def test_translate_openqasm_synth():
    synth = cudaq.synthesize(kernel, 4)

    asm = cudaq.translate(synth, format="openqasm2")
    assert "measure var0[3] -> var8[0]" in asm


def test_translate_openqasm_call():
    # error: 'cc.instantiate_callable' op unable to translate op to OpenQASM 2.0
    with pytest.raises(RuntimeError) as e:
        print(cudaq.translate(kernel_with_call, format="openqasm2"))
    assert 'getASM: failed to translate to OpenQASM.' in repr(e)


def test_translate_qir():
    qir = cudaq.translate(bell_pair, format="qir")
    assert "%1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_qir_ignored_args():
    qir = cudaq.translate(bell_pair, 5, format="qir")
    assert "%1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_qir_with_args():
    qir = cudaq.translate(kernel, 5, format="qir")
    assert "%2 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 %0)" in qir


def test_translate_qir_call():
    qir = cudaq.translate(kernel_with_call, format="qir")
    assert "%2 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)" in qir


def test_translate_qir_base():
    qir = cudaq.translate(bell_pair, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_qir_base_ignored_args():
    qir = cudaq.translate(bell_pair, 5, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_qir_base_args():
    synth = cudaq.synthesize(kernel, 5)
    qir = cudaq.translate(synth, 5, format="qir-base")
    assert '"qir_profiles"="base_profile"' in qir


def test_translate_qir_adaptive():
    qir = cudaq.translate(bell_pair, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir


def test_translate_qir_adaptive_ignored_args():
    qir = cudaq.translate(bell_pair, 5, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir


def test_translate_qir_adaptive_args():
    synth = cudaq.synthesize(kernel, 5)
    qir = cudaq.translate(synth, 5, format="qir-adaptive")
    assert '"qir_profiles"="adaptive_profile"' in qir
