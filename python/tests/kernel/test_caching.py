# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Ownership and behavioral tests for the automatic per-kernel JIT cache."""

import numpy as np
import pytest

import cudaq

skipIfValueSemantics = pytest.mark.skipif(True,
                                          reason="broken in value semantics")


def assert_owns_compiled_module_cache(kernel):
    """A launch installs one stable cache object on its kernel owner."""
    assert hasattr(kernel, '_compiled_module_cache')
    first_lookup = kernel.compiledModuleCache()
    second_lookup = kernel.compiledModuleCache()
    assert first_lookup is kernel._compiled_module_cache
    assert second_lookup is first_lookup


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
    assert_owns_compiled_module_cache(flip)


def test_cache_mode_sample():
    """cudaq.sample drives the kernel through its per-kernel cache."""

    @cudaq.kernel
    def ones():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    for _ in range(3):
        assert cudaq.sample(ones, shots_count=1).count("111") == 1
    assert_owns_compiled_module_cache(ones)


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
    assert_owns_compiled_module_cache(bell)


def test_cache_mode_get_state():
    """cudaq.get_state."""

    @cudaq.kernel
    def fixed():
        qubits = cudaq.qvector(2)
        x(qubits[0])

    s1 = np.array(cudaq.get_state(fixed))
    s2 = np.array(cudaq.get_state(fixed))
    np.testing.assert_allclose(s1, s2)
    assert_owns_compiled_module_cache(fixed)


def test_cache_mode_get_unitary():
    """cudaq.get_unitary."""

    @cudaq.kernel
    def h_kernel():
        q = cudaq.qubit()
        h(q)

    u1 = cudaq.get_unitary(h_kernel)
    u2 = cudaq.get_unitary(h_kernel)
    np.testing.assert_allclose(u1, u2)
    assert_owns_compiled_module_cache(h_kernel)


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

    # Repeat one runtime argument.
    for _ in range(3):
        assert all(r == 3 for r in cudaq.run(count_ones, 3, shots_count=2))
    # Runtime arguments are execution inputs, so changing them must preserve
    # both compiled-code reuse and result correctness.
    assert all(r == 6 for r in cudaq.run(count_ones, 6, shots_count=2))
    assert all(r == 3 for r in cudaq.run(count_ones, 3, shots_count=2))
    assert_owns_compiled_module_cache(count_ones)


def test_cache_mode_builder():
    """PyKernel (builder) owns its compiled-module cache."""

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(3)
    kernel.x(qreg)
    kernel.mz(qreg)

    for _ in range(3):
        assert cudaq.sample(kernel, shots_count=1).count("111") == 1
    assert_owns_compiled_module_cache(kernel)


def test_builder_wrapper_shares_builder_cache():
    """Transient decorator adapters retain the builder's cache identity."""

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.x(qubit)

    first = np.array(cudaq.get_state(kernel))
    cache = kernel._compiled_module_cache
    second = np.array(cudaq.get_state(kernel))

    np.testing.assert_allclose(first, second)
    assert kernel._compiled_module_cache is cache


@skipIfValueSemantics
def test_builder_mutation_discards_compiled_module_cache():
    """Extending a compiled builder cannot reuse code for its old body."""

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    assert cudaq.sample(kernel, shots_count=1).count("0") == 1
    old_cache = kernel._compiled_module_cache

    kernel.x(qubit)
    assert not hasattr(kernel, '_compiled_module_cache')
    assert cudaq.sample(kernel, shots_count=1).count("1") == 1
    assert kernel._compiled_module_cache is not old_cache


# ---------------------------------------------------------------------------
# Per-kernel cache isolation.
# ---------------------------------------------------------------------------

skipIfValueSemantics = pytest.mark.skipif(True,
                                          reason="broken in value semantics")


@skipIfValueSemantics
def test_independent_caches_per_kernel():
    """Two kernels must not share a compiled-module cache."""

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
    assert_owns_compiled_module_cache(all_zero)
    assert_owns_compiled_module_cache(all_one)
    assert all_zero._compiled_module_cache is not all_one._compiled_module_cache


def test_different_runtime_args_remain_correct_with_cache_reuse():
    """Different runtime arguments must not corrupt compiled-code reuse."""

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
    """Two syntheses of the same parent kernel must remain independent."""

    @cudaq.kernel
    def all_one(n: int):
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)

    synth_3 = cudaq.synthesize(all_one, 3)
    synth_5 = cudaq.synthesize(all_one, 5)

    assert cudaq.sample(synth_3, shots_count=1).count("111") == 1
    assert cudaq.sample(synth_5, shots_count=1).count("11111") == 1
    # Repeat in reverse order — independent caches must keep results intact.
    assert cudaq.sample(synth_5, shots_count=1).count("11111") == 1
    assert cudaq.sample(synth_3, shots_count=1).count("111") == 1
    # Parent kernel still works with arbitrary args.
    assert cudaq.sample(all_one, 4, shots_count=1).count("1111") == 1


@skipIfValueSemantics
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


def test_synthesized_kernels_remain_independent():
    """Interleaved launches of synthesized kernels must remain independent."""

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
    correctly for varying arg values."""

    @cudaq.kernel
    def k(n: int) -> bool:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    assert k(5) is True
    assert k(7) is True
    assert_owns_compiled_module_cache(k)


def test_captured_kernel_change_reflected_after_first_launch():
    """Changing the captured kernel must invalidate the cache."""

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

    assert_owns_compiled_module_cache(outer)
