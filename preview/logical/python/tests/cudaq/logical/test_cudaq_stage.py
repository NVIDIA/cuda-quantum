# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q's ``quake-unmapped`` stage prepares every kernel Logical imports.

A kernel reaches Logical's import through a Logical target, such as
``cudaq.estimate`` on ``cudaq.logical.targets.estimator``, or directly through
``cudaq.logical.compiler.import_cudaq``. Both compile with the same compile
target, so the behavior tests use ``import_cudaq``, and one test checks that
the two paths agree.
"""

import pytest

import cudaq
import cudaq.logical as cql
from cudaq.logical._native import native

pytestmark = pytest.mark.skipif(
    not native.has_quake_import,
    reason="QLX was built without CUDA-Q Quake support",
)


def _actions(build):
    return cql.analysis.estimate(build, tier=cql.analysis.Tier.LOGICAL).actions


@cudaq.kernel
def loops():
    qubits = cudaq.qvector(1100)
    for _ in range(50):
        for i in range(4):
            x.ctrl(qubits[i], qubits[i + 1])
    for i in range(1100):
        h(qubits[i])
    for i in range(4):
        rz(0.1 * (i + 1), qubits[0])
    mz(qubits)


def test_only_loops_that_use_their_index_unroll():
    # The repeat loop stays folded, with the index loop inside it unrolled.
    # The 1100-iteration loop and the angle loop unroll, so they leave no
    # second `cflow.repeat`.
    build = cql.compiler.import_cudaq(loops)
    assert build.to_mlir().count("cflow.repeat") == 1
    actions = _actions(build)
    assert actions["qlx_standard_cx"] == 50 * 4
    assert actions["qlx_standard_h"] == 1100


@cudaq.kernel
def ccrx():
    qubits = cudaq.qvector(3)
    rx.ctrl(0.7, qubits[0], qubits[1], qubits[2])
    mz(qubits)


def test_import_cudaq_matches_the_target_whatever_target_is_installed():
    cudaq.set_target(cql.targets.estimator)
    try:
        through_target = cudaq.estimate(ccrx).annotations["LOGICAL"]
    finally:
        cudaq.reset_target()
    # `import_cudaq` compiles with a Logical target's compile target, never
    # with the installed one, even when that target is emulated.
    cudaq.set_target("quantinuum", emulate=True)
    try:
        direct = cql.compiler.import_cudaq(ccrx)
    finally:
        cudaq.reset_target()
    assert cql.analysis.estimate(
        direct, tier=cql.analysis.Tier.LOGICAL).to_dict() == through_target


@cudaq.kernel
def noisy():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1, qubits[0])
    mz(qubits)


def test_a_noisy_kernel_fails_to_import():
    # Prep keeps `quake.apply_noise`, so Logical never estimates the kernel
    # as noiseless, and the import rejects it.
    with pytest.raises(RuntimeError, match="supports only value-semantics"):
        cql.compiler.import_cudaq(noisy)
