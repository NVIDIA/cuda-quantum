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


@cudaq.kernel(external=True)
def wait(duration: float, q: cudaq.qubit) -> None:
    ...


@cudaq.kernel(external=True, backend_symbol="__qm__wait_function")
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


@cudaq.kernel(external=True)
def probe(q: cudaq.qubit) -> float:
    ...


def test_extern_kernel_returning_a_value():
    """A declared return type is carried through to the call."""

    @cudaq.kernel
    def measure_and_rotate():
        q = cudaq.qubit()
        h(q)
        angle = probe(q)
        ry(angle, q)
        mz(q)

    print(measure_and_rotate)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__measure_and_rotate
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h
# CHECK:           %[[VAL_1:.*]] = call @probe(%[[VAL_0]]) : (!quake.ref) -> f64
# CHECK:           quake.ry (%[[VAL_1]])
# CHECK:         func.func private @probe(!quake.ref) -> f64


@cudaq.kernel(external=True)
def wait_vec(q: cudaq.qvector, duration: float) -> None:
    ...


def test_extern_kernel_taking_a_qvector():
    """A qvector argument needs no size in the declaration."""

    @cudaq.kernel
    def vectored(d: float):
        q = cudaq.qvector(3)
        h(q[0])
        wait_vec(q, d)
        mz(q)

    print(vectored)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__vectored
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<3>) -> !quake.veq<?>
# CHECK:           call @wait_vec(%[[VAL_1]], %{{.*}}) : (!quake.veq<?>, f64) -> ()
# CHECK:         func.func private @wait_vec(!quake.veq<?>, f64)


def test_extern_kernel_module_alias():
    """An annotation written against a `cudaq` alias resolves."""
    import cudaq as cq

    @cq.kernel(external=True)
    def aliased_wait(q: cq.qubit, duration: float) -> None:
        ...

    @cq.kernel
    def aliased(d: float):
        q = cq.qubit()
        aliased_wait(q, d)
        mz(q)

    print(aliased)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__aliased
# CHECK:           call @aliased_wait(%{{.*}}, %{{.*}}) : (!quake.ref, f64) -> ()
# CHECK:         func.func private @aliased_wait(!quake.ref, f64)


def test_extern_kernel_declaration_errors():
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(external=True)
        def no_return_annotation(d: float, q: cudaq.qubit):
            ...

    assert 'missing a return type annotation' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(external=True)
        def returns_a_qubit(d: float, q: cudaq.qubit) -> cudaq.qubit:
            ...

    assert 'cannot return a quantum type' in str(e.value)


def test_extern_kernel_option_errors():
    """The two options only make sense together."""
    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(backend_symbol="__qm__wait_function")
        def not_external(d: float):
            ...

    assert 'requires `external=True`' in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel(external=True, verbose=True)
        def compiled_option(d: float, q: cudaq.qubit) -> None:
            ...

    assert 'takes no other `cudaq.kernel` options' in str(e.value)


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
