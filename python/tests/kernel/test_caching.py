# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Correctness tests for the automatic per-kernel JIT cache living behind
`PyKernelDecorator._compiled_module` / `PyKernel._compiled_module`."""

import numpy as np
import pytest

import cudaq


def assert_cached(kernel):
    """A cacheable launch must leave the kernel's slot populated with a module
    that was actually compiled (non-default name)."""
    assert hasattr(kernel, '_compiled_module'), \
        "no _compiled_module slot was installed after launch"
    assert kernel._compiled_module.name != "", \
        "_compiled_module slot was installed but never written"


# ---------------------------------------------------------------------------
# Cacheable launch modes — one test per launch path.
# ---------------------------------------------------------------------------


def test_cache_mode_call():
    """Direct invocation."""

    @cudaq.kernel
    def flip() -> bool:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    for _ in range(3):
        assert flip() is True
    assert_cached(flip)


def test_cache_mode_sample():
    """cudaq.sample drives the kernel via __call__, sharing the 'call' slot."""

    @cudaq.kernel
    def ones():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    for _ in range(3):
        assert cudaq.sample(ones, shots_count=1).count("111") == 1
    assert_cached(ones)


def test_cache_mode_draw():
    """cudaq.draw."""

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    drawn = cudaq.draw(bell)
    assert "h" in drawn
    # Repeated draws should be stable.
    assert cudaq.draw(bell) == drawn
    assert cudaq.draw(bell) == drawn
    assert_cached(bell)


def test_cache_mode_get_state():
    """cudaq.get_state."""

    @cudaq.kernel
    def fixed():
        qubits = cudaq.qvector(2)
        x(qubits[0])

    s1 = np.array(cudaq.get_state(fixed))
    s2 = np.array(cudaq.get_state(fixed))
    np.testing.assert_allclose(s1, s2)
    assert_cached(fixed)


def test_cache_mode_get_unitary():
    """cudaq.get_unitary."""

    @cudaq.kernel
    def h_kernel():
        q = cudaq.qubit()
        h(q)

    u1 = cudaq.get_unitary(h_kernel)
    u2 = cudaq.get_unitary(h_kernel)
    np.testing.assert_allclose(u1, u2)


def test_cache_mode_run():
    """cudaq.run."""

    @cudaq.kernel
    def count_ones(n: int) -> int:
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)
        total = 0
        for i in range(n):
            if mz(qubits[i]):
                total += 1
        return total

    # Same arg → cache hit.
    for _ in range(3):
        assert all(r == 3 for r in cudaq.run(count_ones, 3, shots_count=2))
    # Different arg → cache turnover; result must still be correct.
    assert all(r == 6 for r in cudaq.run(count_ones, 6, shots_count=2))
    assert all(r == 3 for r in cudaq.run(count_ones, 3, shots_count=2))


def test_cache_mode_builder():
    """PyKernel (builder) uses its own _compiled_module slot."""

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(3)
    kernel.x(qreg)
    kernel.mz(qreg)

    for _ in range(3):
        assert cudaq.sample(kernel, shots_count=1).count("111") == 1


# ---------------------------------------------------------------------------
# Per-kernel cache isolation.
# ---------------------------------------------------------------------------


def test_independent_caches_per_kernel():
    """Two kernels must not share a cache slot."""

    @cudaq.kernel
    def all_zero():
        cudaq.qvector(3)

    @cudaq.kernel
    def all_one():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    # Interleave so a shared cache would corrupt results.
    assert cudaq.sample(all_zero, shots_count=1).count("000") == 1
    assert cudaq.sample(all_one, shots_count=1).count("111") == 1
    assert cudaq.sample(all_zero, shots_count=1).count("000") == 1
    assert cudaq.sample(all_one, shots_count=1).count("111") == 1


def test_different_args_correct_after_cache_turnover():
    """Different concrete args force a re-JIT; results must remain correct."""

    @cudaq.kernel
    def all_one(n: int):
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)

    assert cudaq.sample(all_one, 3, shots_count=1).count("111") == 1
    assert cudaq.sample(all_one, 5, shots_count=1).count("11111") == 1
    assert cudaq.sample(all_one, 3, shots_count=1).count("111") == 1


# ---------------------------------------------------------------------------
# Synthesis bypass.
# ---------------------------------------------------------------------------


def test_synthesized_kernel_correctness():
    """Two syntheses of the same parent kernel must not share cache state."""

    @cudaq.kernel
    def all_one(n: int):
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)

    synth_3 = cudaq.synthesize(all_one, 3)
    synth_5 = cudaq.synthesize(all_one, 5)

    assert cudaq.sample(synth_3, shots_count=1).count("111") == 1
    assert cudaq.sample(synth_5, shots_count=1).count("11111") == 1
    # Repeat in reverse order — independent slots must keep results intact.
    assert cudaq.sample(synth_5, shots_count=1).count("11111") == 1
    assert cudaq.sample(synth_3, shots_count=1).count("111") == 1
    # Parent kernel still works with arbitrary args.
    assert cudaq.sample(all_one, 4, shots_count=1).count("1111") == 1


def test_redefined_kernel_does_not_hit_stale_cache():
    """Rebinding a kernel name yields a fresh decorator and a fresh JIT."""

    @cudaq.kernel
    def k():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    assert cudaq.sample(k, shots_count=1).count("111") == 1

    # Rebind the same Python name to a kernel with a different body. Under a
    # per-name (rather than per-decorator) cache this would still run the
    # all-ones body.
    @cudaq.kernel
    def k():
        cudaq.qvector(3)

    assert cudaq.sample(k, shots_count=1).count("000") == 1


def test_observe_with_different_spin_operators():
    """The Python observe path reads ctx->spin at runtime, not JIT time, so the
    cache stays valid across spin-op changes."""

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qubit()
        rx(theta, q)

    # <X> on |+_x rotation> kernel: rx(pi/2)|0> = |-iY> direction.
    # Pick two Hamiltonians whose expectation values must differ on this state.
    z = cudaq.spin.z(0)
    x = cudaq.spin.x(0)

    theta = np.pi / 2
    ez = cudaq.observe(ansatz, z, theta).expectation()
    ex = cudaq.observe(ansatz, x, theta).expectation()

    # rx(pi/2)|0> ≈ (|0> - i|1>) / sqrt(2): <Z>=0, <X>=0 here actually.
    # Use a more distinguishing angle.
    theta = 0.3
    ez = cudaq.observe(ansatz, z, theta).expectation()
    ex = cudaq.observe(ansatz, x, theta).expectation()
    # For rx(theta)|0>: <Z> = cos(theta), <X> = 0.
    assert ez == pytest.approx(np.cos(theta), abs=1e-6)
    assert ex == pytest.approx(0.0, abs=1e-6)

    # Now swap the order -- if the JIT was cached on the previous launch, the
    # second observe below would get the stale ansatz from the first one.
    ex2 = cudaq.observe(ansatz, x, theta).expectation()
    ez2 = cudaq.observe(ansatz, z, theta).expectation()
    assert ex2 == pytest.approx(0.0, abs=1e-6)
    assert ez2 == pytest.approx(np.cos(theta), abs=1e-6)


def test_synthesized_kernel_does_not_cache():
    """Synthesized kernels bypass the cache (is_cachable arity check fails);
    interleaved launches must remain independent."""

    @cudaq.kernel
    def all_one(n: int):
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)

    synth_a = cudaq.synthesize(all_one, 3)
    synth_b = cudaq.synthesize(all_one, 4)
    # Interleave — a stale cache hit from synth_a would corrupt synth_b's
    # measurement, and vice versa.
    for _ in range(2):
        assert cudaq.sample(synth_a, shots_count=1).count("111") == 1
        assert cudaq.sample(synth_b, shots_count=1).count("1111") == 1


def test_kernel_with_unused_argument():
    """A kernel that takes an argument but never uses it must still run
    correctly for varying arg values. (The current predicate marks this kind
    of kernel as fully-synthesized and bypasses the cache; correctness is
    preserved, only the perf optimization is lost.)"""

    @cudaq.kernel
    def k(n: int) -> bool:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    assert k(5) is True
    assert k(7) is True
    # Documented limitation: cache slot stays empty because every formal
    # arg is dead, so isFullySynthesized() reports true.
    assert (not hasattr(k, '_compiled_module') or k._compiled_module.name == "")


def test_captured_kernel_change_reflected_after_first_launch():
    """A kernel that captures another kernel from its enclosing scope must
    not cache: rebinding the captured name to a different body has to take
    effect on the next call, not be masked by a stale JIT artifact."""

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def outer() -> bool:
        q = cudaq.qubit()
        inner(q)
        return mz(q)

    # v1: inner flips |0> -> |1>.
    assert outer() is True

    # Rebind `inner` to a no-op body. The lifted capture in `outer` must
    # resolve to this new definition on the next launch.
    @cudaq.kernel
    def inner(q: cudaq.qubit):
        pass

    assert outer() is False

    # The parent kernel must not have been cached — caching would freeze the
    # captured-kernel body inside the JIT artifact and the rebind above would
    # silently no-op.
    assert (not hasattr(outer, '_compiled_module') or
            outer._compiled_module.name == "")
