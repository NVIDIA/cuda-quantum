# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Demo/test for dynamically defining a QPU in Python.
#
# A custom Python QPU is a plain object implementing the
# ``cudaq.qpu.SupportsSampleQPU`` and/or ``cudaq.qpu.SupportsObserveQPU`` protocols.
# Registering it via ``cudaq.set_target(<qpu_instance>)`` selects the
# "python-dynamic" target and wraps the object in the internal C++
# ``PyDynamicQPU``, which forwards compile/launch calls to the Python methods.
#
# See docs/sphinx/using/extending/python_qpu.rst for the full protocol reference.

import cudaq
import cudaq.mlir.ir as mlir
import pytest

from cudaq.qpu import (
    SupportsSampleQPU,
    SupportsObserveQPU,
    CompileTarget,
    SampleResult,
    ObserveResult,
)


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


class DemoQPU:
    """Minimal custom QPU implementing the sample and observe protocols.

    Fill in the bodies to showcase how a dynamic Python QPU intercepts the
    compile and launch steps. A real backend would submit ``module`` to an
    external service/simulator inside the launch methods.
    """

    def __init__(self):
        self.traced_calls = []
        self.mlir_module = None

    # --- SupportsSampleQPU ---

    def get_compile_target_sample(self) -> CompileTarget:
        self.traced_calls.append("get_compile_target_sample")
        return CompileTarget.default_sample()

    def launch_sample(self, module, args):
        self.traced_calls.append(f"launch_sample({args})")
        self.mlir_module = module.mlir_module
        return SampleResult()

    # --- SupportsObserveQPU ---

    def get_compile_target_observe(self) -> CompileTarget:
        self.traced_calls.append("get_compile_target_observe")
        return CompileTarget.default_observe()

    def launch_observe(self, module, args):
        self.traced_calls.append(f"launch_observe({args})")
        self.mlir_module = module.mlir_module
        return ObserveResult(1.0, cudaq.spin.x(0), SampleResult())


def test_protocol_conformance():
    """The demo QPU should satisfy both runtime-checkable protocols."""
    qpu = DemoQPU()
    assert isinstance(qpu, SupportsSampleQPU)
    assert isinstance(qpu, SupportsObserveQPU)


def test_set_target_registers_python_dynamic():
    """Registering the QPU instance should activate the python-dynamic target."""
    qpu = DemoQPU()
    cudaq.set_target(qpu)
    assert cudaq.get_target().name == "python-dynamic"


@cudaq.kernel
def kernel(n_qubits: int, array: list[int]):
    qbs = cudaq.qvector(n_qubits)
    h(qbs[0])


def test_sample_launch():
    """`cudaq.sample` should route through the custom QPU's launch_sample."""
    qpu = DemoQPU()
    cudaq.set_target(qpu)

    assert qpu.traced_calls == []
    cudaq.sample(kernel, 1, [1, 2, 3])
    assert qpu.traced_calls == [
        "get_compile_target_sample",
        'launch_sample(KernelArgs([1, <instance of !cc.stdvec<i64>>]))'
    ]
    assert isinstance(qpu.mlir_module, mlir.Module)


def test_observe_launch():
    """`cudaq.observe` should route through the custom QPU's launch_observe."""
    qpu = DemoQPU()
    cudaq.set_target(qpu)

    assert qpu.traced_calls == []
    cudaq.observe(kernel, cudaq.spin.x(0), 2, [])
    assert qpu.traced_calls == [
        "get_compile_target_observe",
        'launch_observe(KernelArgs([2, <instance of !cc.stdvec<i64>>]))'
    ]
    assert isinstance(qpu.mlir_module, mlir.Module)


def test_sample_twice():
    qpu = DemoQPU()
    cudaq.set_target(qpu)
    cudaq.sample(kernel, 1, [1, 2, 3])
    cudaq.sample(kernel, 2, [1, 2, 3])
    assert qpu.traced_calls == [
        "get_compile_target_sample",
        'launch_sample(KernelArgs([1, <instance of !cc.stdvec<i64>>]))',
        "get_compile_target_sample",
        'launch_sample(KernelArgs([1, <instance of !cc.stdvec<i64>>]))'
    ]


class SampleOnlyQPU:
    """QPU that does not support observe"""

    def get_compile_target_sample(self) -> CompileTarget:
        pass

    def launch_sample(self, module, args):
        pass


class ObserveOnlyQPU:
    """QPU that does not support sample"""

    def get_compile_target_observe(self) -> CompileTarget:
        pass

    def launch_observe(self, module, args):
        pass


def test_sample_fails_when_observe_only():
    qpu = ObserveOnlyQPU()
    cudaq.set_target(qpu)
    with pytest.raises(
            RuntimeError,
            match="QPU does not implement the SupportsSampleQPU protocol"):
        cudaq.sample(kernel, 1, [1, 2, 3])


def test_observe_fails_when_sample_only():
    qpu = SampleOnlyQPU()
    cudaq.set_target(qpu)
    with pytest.raises(
            RuntimeError,
            match="QPU does not implement the SupportsObserveQPU protocol"):
        cudaq.observe(kernel, cudaq.spin.x(0), 2, [])


# leave in place so `pytest test_python_dynamic_qpu.py` behaves like the rest of
# the backends suite when run standalone
if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
