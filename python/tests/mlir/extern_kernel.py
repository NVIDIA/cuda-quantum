# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import numpy as np
import pytest

import cudaq


@cudaq.extern_kernel
def wait(duration: float, q: cudaq.qubit) -> None:
    ...


@cudaq.extern_kernel("__qm__wait_function")
def renamed_wait(duration: float, q: cudaq.qubit) -> None:
    ...


def test_extern_kernel_call():
    """The call is emitted in reference form with a bodyless declaration."""

    @cudaq.kernel
    def ramsey(d: float):
        q = cudaq.qubit()
        rx(np.pi / 2, q)
        wait(d, q)
        rx(np.pi / 2, q)
        mz(q)

    print(ramsey)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__ramsey
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rx
# CHECK:           call @wait(%{{.*}}, %[[VAL_0]]) : (f64, !quake.ref) -> ()
# CHECK:           quake.rx
# CHECK:         func.func private @wait(f64, !quake.ref)


def test_extern_kernel_backend_symbol():
    """The backend symbol may differ from the Python name."""

    @cudaq.kernel
    def renamed(d: float):
        q = cudaq.qubit()
        renamed_wait(d, q)

    print(renamed)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__renamed
# CHECK:           call @__qm__wait_function(%{{.*}}, %{{.*}}) : (f64, !quake.ref) -> ()
# CHECK:         func.func private @__qm__wait_function(f64, !quake.ref)


def test_extern_kernel_declaration_errors():
    with pytest.raises(RuntimeError) as e:

        @cudaq.extern_kernel
        def returns_a_value(d: float, q: cudaq.qubit) -> int:
            ...

    assert 'must return None' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.extern_kernel
        def takes_a_qvector(d: float, q: cudaq.qvector) -> None:
            ...

    assert 'takes a qvector' in str(e.value)


def test_extern_kernel_call_errors():
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def too_few(d: float):
            q = cudaq.qubit()
            wait(d)

        print(too_few)

    assert 'takes 2 argument(s), but 1 were given' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def wrong_types(d: float):
            q = cudaq.qubit()
            wait(q, d)

        print(wrong_types)

    assert 'cannot convert value of type' in str(e.value)


def test_extern_kernel_not_callable_from_python():
    with pytest.raises(RuntimeError) as e:
        wait(1.0, None)

    assert 'can only be called from inside a CUDA-Q kernel' in str(e.value)
