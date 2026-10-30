# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the compiler-bench-nisq target: decomposition to a CZ and rotation
# basis, with optional routing onto a device.

import re
from pathlib import Path

import cudaq
import numpy as np
import pytest
from test_helpers import derive_compile_target

cudaq._register_target_config(
    str(
        Path(__file__).resolve().parents[1] / "targets" /
        "compiler-bench-nisq.yml"))

NISQ_TARGET = 'compiler-bench-nisq'

ALLOWED_NISQ_OPS = {'h', 'rx', 'ry', 'rz', 'x', 'cz', 'mx', 'my', 'mz'}


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def set_target_on_device(device):
    """Route onto `device` by deriving the target's compile target."""
    cudaq.set_target(NISQ_TARGET,
                     compile_target=derive_compile_target(NISQ_TARGET,
                                                          device=device))


def test_swap_decomposition():
    """SWAP decomposes into 3 CZ gates under compiler-bench-nisq."""
    cudaq.set_target(NISQ_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.gate_count_for_arity(2) == 3
    assert resources.depth_for_arity(2) == 3


def test_swap_no_decomposition_default_target():
    """SWAP stays as a single 2Q gate on qpp-cpu (no decomposition)."""
    cudaq.set_target('qpp-cpu')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.gate_count_for_arity(2) == 1
    assert resources.depth_for_arity(2) == 1


def test_custom_unitary_produces_2q_gates():
    """Registered SU(4) unitary must produce entangling gates after synthesis.

    This validates SU(4) handling through the target pipeline. Without helper
    inlining, resource counting sees only 1Q gates for this case.
    """
    cudaq.set_target(NISQ_TARGET)

    rng = np.random.default_rng(42)
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q_mat, r = np.linalg.qr(z)
    d = np.diag(r)
    mat = q_mat * (d / np.abs(d))

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    cudaq.register_operation("test_su4_pipeline", mat.flatten().tolist())
    kernel.test_su4_pipeline(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    ops = resources.to_dict()
    two_q = resources.gate_count_for_arity(2)
    assert 'custom_op' not in ops, f"Custom SU(4) was not synthesized: {ops}"
    assert ops.get('cz', 0) >= 1, f"Custom SU(4) did not lower to CZ: {ops}"
    assert two_q >= 1, (f"Random SU(4) produced 0 2Q gates. Gates: {ops}")
    assert two_q <= 6, (
        f"KAK produces at most 3 CX (6 CZ after basis change), got {two_q}")


def test_ccx_fully_decomposed():
    """CCX (Toffoli) must decompose to CZ basis, not remain as ccx.

    The decomposition pass must select CCXToCCZ and CCZToCX patterns
    even when t and s are not directly in the basis. Requires unbounded
    (n) registration for SToR1/TToR1 and wildcard matching in the
    pattern selection graph.
    """
    cudaq.set_target(NISQ_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    kernel.cx([q[0], q[1]], q[2])

    resources = cudaq.estimate_resources(kernel)
    ops = resources.to_dict()
    assert 'ccx' not in ops, f"CCX not decomposed: {ops}"
    assert resources.gate_count_for_arity(2) > 0


def test_axis_measurements_normalize_to_nisq_basis():
    cudaq.set_target(NISQ_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.mx(q[0])
    kernel.my(q[1])
    kernel.rz(0.7853981633974483, q[2])
    kernel.mz(q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert ops.get('mx', 0) == 1
    assert ops.get('my', 0) == 1
    assert ops.get('mz', 0) == 1
    assert ops.get('rz', 0) == 1
    assert 's' not in ops and 't' not in ops, (
        f"NISQ target should keep a native rotation, not S or T: {ops}")
    assert set(ops).issubset(ALLOWED_NISQ_OPS), (
        f"Unexpected NISQ operations after normalization: {ops}")


def test_negated_control_lowers_to_cz_basis():
    """Negated controls lower to the target CZ basis."""
    cudaq.set_target(NISQ_TARGET)

    @cudaq.kernel
    def kernel():
        c = cudaq.qubit()
        q = cudaq.qubit()
        cx(~c, q)
        cx(c, q)

    resources = cudaq.estimate_resources(kernel)
    ops = resources.to_dict()
    assert ops.get('x', 0) == 2, f"Negated control was not expanded: {ops}"
    assert ops.get('cz', 0) == 2, f"CX gates did not lower to CZ: {ops}"
    assert resources.gate_count_for_arity(2) == 2


# On path(4), `x.ctrl(q[0], q[3])` needs swaps, so these kernels prepare 1101
# on device qubits in another order.


@cudaq.kernel
def entangle(a: cudaq.qubit, b: cudaq.qubit):
    x.ctrl(a, b)


@cudaq.kernel
def prepare_1101():
    q = cudaq.qvector(4)
    x(q[0])
    entangle(q[0], q[3])
    x(q[1])


@cudaq.kernel
def prepare_1101_measure_each():
    q = cudaq.qvector(4)
    x(q[0])
    x.ctrl(q[0], q[3])
    x(q[1])
    mz(q[0])
    mz(q[1])
    mz(q[2])
    mz(q[3])


@cudaq.kernel
def rolled_loop():
    q = cudaq.qvector(4)
    x(q[0])
    for i in range(2000):
        x.ctrl(q[0], q[3])
        h(q[1])
        h(q[2])


def test_routed_sample_in_kernel_order():
    """`sample` applies the mapper's reorder index, except with
    `explicit_measurements`, whose results are in execution order. `q[2]` is
    idle, so the level-1 `dqe` would remove it unless `sample` measured first.
    A loop longer than the unroll limit carries device qubits into the
    simulator; it runs an even number of times."""
    set_target_on_device('path(4)')
    counts = cudaq.sample(prepare_1101, shots_count=10)
    assert counts.most_probable() == '1101'
    assert len(counts) == 1
    counts = cudaq.sample(prepare_1101_measure_each,
                          shots_count=10,
                          explicit_measurements=True)
    assert counts.get_sequential_data() == ['1101'] * 10
    assert cudaq.sample(rolled_loop, shots_count=10).most_probable() == '1000'


@cudaq.kernel
def entangle_ends_4():
    q = cudaq.qvector(4)
    h(q)
    x.ctrl(q[0], q[3])


def test_translate_from_wire_set():
    """`translate` compiles against the installed compile target, so it
    routes, and it converts the wires borrowed from the device's wire set for
    full QIR and for OpenQASM. On path(4), every two-qubit gate acts on
    neighboring device qubits."""
    set_target_on_device('path(4)')
    qir = cudaq.translate(entangle_ends_4, format='qir')
    assert '__quantum__qis__z__ctl' in qir
    qasm = cudaq.translate(entangle_ends_4, format='openqasm2')
    pairs = re.findall(r'cx var0\[(\d)\], var0\[(\d)\];', qasm)
    assert len(pairs) > 1
    assert all(abs(int(a) - int(b)) == 1 for a, b in pairs)


@cudaq.kernel
def entangle_ends(n: int):
    q = cudaq.qvector(n)
    h(q)
    x.ctrl(q[0], q[n - 1])


@cudaq.kernel
def entangled_end() -> bool:
    q = cudaq.qvector(4)
    x(q[0])
    x.ctrl(q[0], q[3])
    return mz(q[3])


def test_routed_kernel_forms(capfd):
    """Without a device nothing is routed. On one, an argument-sized register,
    a callee with qubit arguments, and a kernel that returns a value compile
    and route, and a kernel with more qubits than the device fails."""
    cudaq.set_target(NISQ_TARGET)
    assert cudaq.estimate_resources(entangle_ends,
                                    5).gate_count_for_arity(2) == 1

    set_target_on_device('path(5)')
    routed = cudaq.estimate_resources(entangle_ends, 5)
    assert routed.gate_count_for_arity(2) > 1
    assert routed.num_qubits == 5
    assert routed.num_used_qubits == 5

    set_target_on_device('path(4)')
    assert cudaq.estimate_resources(prepare_1101).gate_count_for_arity(2) > 1
    assert cudaq.run(entangled_end, shots_count=3) == [True] * 3
    with pytest.raises(RuntimeError, match='lowering failed'):
        cudaq.estimate_resources(entangle_ends, 5)
    assert 'Too many qubits' in capfd.readouterr().err
