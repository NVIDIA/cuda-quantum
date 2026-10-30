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
``cudaq.logical.compiler.import_cudaq``. Both compile it with the target's
compile target, so most tests run on both paths through the ``import_kernel``
fixture.
"""

import pytest

import cudaq
import cudaq.logical as cql
from cudaq.logical._native import native

pytestmark = pytest.mark.skipif(
    not native.has_quake_import,
    reason="QLX was built without CUDA-Q Quake support",
)


def _estimate_through_target(kernel, monkeypatch):
    """Run ``cudaq.estimate`` on the estimator target and return the Build
    that Logical imported."""
    builds = []
    compile_build = cql.targets.ProgramBackend.compile

    def recording(self, source, **options):
        builds.append(compile_build(self, source, **options))
        return builds[-1]

    monkeypatch.setattr(cql.targets.ProgramBackend, "compile", recording)
    cudaq.set_target(cql.targets.estimator)
    try:
        cudaq.estimate(kernel)
    finally:
        cudaq.reset_target()
    return builds[-1]


@pytest.fixture(params=["target", "import_cudaq"])
def import_kernel(request, monkeypatch):
    """Import a kernel on one of the two paths and return the Build."""
    if request.param == "import_cudaq":
        return cql.compiler.import_cudaq
    return lambda kernel: _estimate_through_target(kernel, monkeypatch)


def _actions(build):
    return cql.analysis.estimate(build, tier=cql.analysis.Tier.LOGICAL).actions


@cudaq.kernel
def repeat_around_index_loop():
    qubits = cudaq.qvector(5)
    for _ in range(50):
        for i in range(4):
            x.ctrl(qubits[i], qubits[i + 1])
    mz(qubits)


@cudaq.kernel
def long_index_loop():
    qubits = cudaq.qvector(1100)
    for i in range(1100):
        h(qubits[i])
    mz(qubits)


@cudaq.kernel
def angle_loop():
    qubit = cudaq.qubit()
    for i in range(4):
        rz(0.1 * (i + 1), qubit)
        h(qubit)
    mz(qubit)


def test_a_repeat_loop_keeps_its_unrolled_index_loop_inside(import_kernel):
    build = import_kernel(repeat_around_index_loop)
    assert build.to_mlir().count("cflow.repeat") == 1
    assert _actions(build)["qlx_standard_cx"] == 50 * 4


def test_an_index_loop_past_1024_iterations_unrolls(import_kernel):
    build = import_kernel(long_index_loop)
    assert "cflow.repeat" not in build.to_mlir()
    assert _actions(build)["qlx_standard_h"] == 1100


def test_a_loop_whose_index_sets_an_angle_unrolls(import_kernel):
    build = import_kernel(angle_loop)
    assert "cflow.repeat" not in build.to_mlir()


@cudaq.kernel
def ccrx():
    qubits = cudaq.qvector(3)
    rx.ctrl(0.7, qubits[0], qubits[1], qubits[2])
    mz(qubits)


@cudaq.kernel
def foldable_phases():
    qubits = cudaq.qvector(2)
    rz(0.3, qubits[0])
    x.ctrl(qubits[0], qubits[1])
    x.ctrl(qubits[0], qubits[1])
    rz(0.4, qubits[0])
    t(qubits[1])
    t.adj(qubits[1])
    mz(qubits)


@pytest.mark.parametrize("kernel", [ccrx, foldable_phases],
                         ids=lambda kernel: kernel.name)
def test_the_target_and_import_cudaq_give_the_same_counts(kernel, monkeypatch):
    # A doubly controlled `rx` decomposes into gates the import accepts, and
    # the global phase that decomposition leaves is lowered. The
    # foldable-phases kernel checks that both paths run the same optimizer.
    through_target = _estimate_through_target(kernel, monkeypatch)
    assert _actions(
        cql.compiler.import_cudaq(kernel)) == _actions(through_target)


def test_import_cudaq_ignores_an_installed_emulated_target(monkeypatch):
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime

    expected = _actions(cql.compiler.import_cudaq(ccrx))
    compile_module = cudaq_runtime.compile_module

    def checked(*args, **kwargs):
        compiled = compile_module(*args, **kwargs)
        # Logical's endpoint reads the MLIR, so the compile builds no JIT
        # engine.
        assert compiled.entry_point is None
        return compiled

    monkeypatch.setattr(cudaq_runtime, "compile_module", checked)
    cudaq.set_target("quantinuum", emulate=True)
    try:
        assert _actions(cql.compiler.import_cudaq(ccrx)) == expected
    finally:
        cudaq.reset_target()


@cudaq.kernel
def noisy():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1, qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


def test_a_noisy_kernel_fails_to_import(import_kernel):
    # Prep keeps `quake.apply_noise`, so Logical never estimates the kernel
    # as noiseless, and the import rejects it.
    with pytest.raises(RuntimeError, match="supports only value-semantics"):
        import_kernel(noisy)
