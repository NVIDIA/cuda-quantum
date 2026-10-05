# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP %s | FileCheck %s

import cudaq


@cudaq.kernel(atomic_quantum_region=True)
def atomic_h(q: cudaq.qubit):
    h(q)


@cudaq.kernel
def ordinary_h(q: cudaq.qubit):
    h(q)


def test_atomic_quantum_region_decorator_ir():
    print("DECORATOR_ATOMIC")
    print(atomic_h)
    print("DECORATOR_ORDINARY")
    print(ordinary_h)


# CHECK-LABEL: DECORATOR_ATOMIC
# CHECK:       func.func @__nvqpp__mlirgen__atomic_h{{.*}}(
# CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-kernel"} {

# CHECK-LABEL: DECORATOR_ORDINARY
# CHECK:       func.func @__nvqpp__mlirgen__ordinary_h{{.*}}(
# CHECK-SAME:    attributes {"cudaq-kernel"} {


def test_atomic_quantum_region_builder_ir():
    atomic, atomic_target = cudaq.make_kernel(cudaq.qubit)
    atomic.atomic_quantum_region()
    atomic.h(atomic_target)

    ordinary, ordinary_target = cudaq.make_kernel(cudaq.qubit)
    ordinary.h(ordinary_target)

    direct = cudaq.make_kernel()
    direct_target = direct.qalloc()
    direct.apply_call(atomic, direct_target)
    direct.apply_call(atomic, direct_target)

    controlled = cudaq.make_kernel()
    controlled_qubits = controlled.qalloc(2)
    controlled.control(atomic, controlled_qubits[0], controlled_qubits[1])
    controlled.control(atomic, controlled_qubits[0], controlled_qubits[1])

    adjoint = cudaq.make_kernel()
    adjoint_target = adjoint.qalloc()
    adjoint.h(adjoint_target)
    adjoint.adjoint(atomic, adjoint_target)

    print("BUILDER_ATOMIC")
    print(atomic)
    print("BUILDER_ORDINARY")
    print(ordinary)
    print("BUILDER_DIRECT")
    print(direct)
    print("BUILDER_CONTROLLED")
    print(controlled)
    print("BUILDER_ADJOINT")
    print(adjoint)


# CHECK-LABEL: BUILDER_ATOMIC
# CHECK:       func.func @{{.*}}(
# CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-entrypoint", "cudaq-kernel"} {

# CHECK-LABEL: BUILDER_ORDINARY
# CHECK:       func.func @{{.*}}(
# CHECK-SAME:    attributes {"cudaq-entrypoint", "cudaq-kernel"} {

# CHECK-LABEL: BUILDER_DIRECT
# CHECK:       call @[[DIRECT_HELPER:[^(]+]](
# CHECK-NEXT:  call @[[DIRECT_HELPER]](
# CHECK:       func.func @[[DIRECT_HELPER]](
# CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-kernel"} {

# CHECK-LABEL: BUILDER_CONTROLLED
# CHECK:       quake.apply @[[CONTROLLED_HELPER:[^ ]+]]
# CHECK-NEXT:  quake.apply @[[CONTROLLED_HELPER]]
# CHECK:       func.func @[[CONTROLLED_HELPER]](
# CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-kernel"} {

# CHECK-LABEL: BUILDER_ADJOINT
# CHECK:       quake.apply<adj> @[[ADJOINT_HELPER:[^( ]+]]
# CHECK:       func.func @[[ADJOINT_HELPER]](
# CHECK-SAME:    attributes {atomic_quantum_region, "cudaq-kernel"} {
