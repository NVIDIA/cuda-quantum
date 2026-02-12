# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


@pytest.fixture(autouse=True)
def clear_registries():
    yield
    cudaq.__clearKernelRegistries()


# ---------------------------------------------------------------------------
# Basic deferred vs eager compilation
# ---------------------------------------------------------------------------


class TestCompilationModes:
    """Verify that kernels compile lazily by default and eagerly on request."""

    def test_deferred_by_default(self):
        """Kernels declared with `@cudaq.kernel` are *not* compiled immediately."""

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            h(q)

        assert not k.is_compiled()

    def test_compiled_on_first_call(self):
        """Deferred kernels compile transparently when first invoked."""

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            h(q)

        assert not k.is_compiled()
        cudaq.sample(k)
        assert k.is_compiled()

    def test_eager_compilation(self):
        """Setting `defer_compilation=False` compiles at declaration time."""

        @cudaq.kernel(defer_compilation=False)
        def k():
            q = cudaq.qubit()
            h(q)

        assert k.is_compiled()

    def test_explicit_compile(self):
        """Calling `.compile()` on a deferred kernel compiles it."""

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            h(q)

        assert not k.is_compiled()
        k.compile()
        assert k.is_compiled()


# ---------------------------------------------------------------------------
# Out‑of‑order definition (forward references)
# ---------------------------------------------------------------------------


class TestOutOfOrderDefinition:
    """
    With deferred compilation a caller can reference a callee that is defined
    *after* the caller, since compilation of the caller is deferred until the
    first call.
    """

    def test_simple_forward_reference(self):
        """Caller uses a callee that is defined after it."""

        @cudaq.kernel
        def caller():
            q = cudaq.qubit()
            apply_x(q)

        @cudaq.kernel
        def apply_x(q: cudaq.qubit):
            x(q)

        caller()

    def test_forward_reference_with_return(self):
        """Forward-referenced callee returns a value used by the caller."""

        @cudaq.kernel
        def caller() -> bool:
            q = cudaq.qubit()
            apply_h(q)
            return mz(q)

        @cudaq.kernel
        def apply_h(q: cudaq.qubit):
            h(q)

        result = caller()
        assert isinstance(result, bool)

    def test_chain_of_forward_references(self):
        """A -> B -> C, where each kernel is defined after its caller."""

        @cudaq.kernel
        def a():
            q = cudaq.qubit()
            b(q)

        @cudaq.kernel
        def b(q: cudaq.qubit):
            c(q)

        @cudaq.kernel
        def c(q: cudaq.qubit):
            x(q)

        a()

    def test_recursive(self):
        """Test a kernel calling itself recursively."""

        # We currently don't support recursive kernel calls.
        @cudaq.kernel
        def a(i: int) -> int:
            # cudaq fails without this, see https://github.com/NVIDIA/cuda-quantum/issues/3963
            a(0)
            if i == 0:
                return 0
            else:
                return a(i - 1) + 1

        with pytest.raises(RuntimeError) as e:
            assert a(3) == 3

        assert "recursive kernel call" in str(e.value)

    def test_recursive_depth_2(self):
        """Test a kernel calling itself recursively indirectly."""

        @cudaq.kernel
        def b(i: int) -> int:
            return a(i)

        @cudaq.kernel
        def a(i: int) -> int:
            # cudaq fails without this, see https://github.com/NVIDIA/cuda-quantum/issues/3963
            b(0)
            if i == 0:
                return b(i - 1) + 1
            else:
                return 0

        with pytest.raises(RuntimeError) as e:
            assert a(3) == 3

        assert "recursive kernel call" in str(e.value)


# ---------------------------------------------------------------------------
# Kernel composition (caller/callee)
# ---------------------------------------------------------------------------


class TestKernelComposition:
    """Deferred compilation works correctly for kernels calling other kernels."""

    def test_caller_callee_same_scope(self):
        """Both kernels defined in normal (top-down) order."""

        @cudaq.kernel
        def callee(qubits: cudaq.qview):
            h(qubits)

        @cudaq.kernel
        def caller(n: int):
            q = cudaq.qvector(n)
            callee(q)

        counts = cudaq.sample(caller, 3)

    def test_callee_resolved_from_definition_scope(self):
        """
        A kernel defined in a nested scope calls another kernel from that
        same scope.  When `.compile()` is invoked from *outside* that scope
        (where the callee is not visible), the callee must still be resolved
        via the captured parentVariables from definition time.
        """

        def _make_kernels():

            @cudaq.kernel
            def foo(q: cudaq.qubit):
                x(q)

            @cudaq.kernel
            def bar():
                q = cudaq.qubit()
                foo(q)

            return [bar, foo]

        bar, hidden_foo = _make_kernels()
        # `foo` is not in scope here, but `bar` should still compile
        # because `foo` is visible in the definition scope of `bar`.
        assert not bar.is_compiled()
        bar.compile()
        assert bar.is_compiled()
        assert hidden_foo.is_compiled()


# ---------------------------------------------------------------------------
# __str__ representation
# ---------------------------------------------------------------------------


class TestStringRepresentation:
    """The `__str__` output should reflect compilation state."""

    def test_str_before_compilation(self):

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            h(q)

        s = str(k)
        assert "Uncompiled" in s
        assert "k" in s

    def test_str_after_compilation(self):

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            h(q)

        k.compile()
        s = str(k)
        assert "Compiled" in s
        assert "k" in s
