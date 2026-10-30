# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
import cudaq
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.kernel.utils import get_optimization_level_attr

OPT_LEVELS = [0, 1, 2, 3]


@cudaq.kernel
def bell() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    return mz(q[0])


@cudaq.kernel
def empty():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@pytest.fixture
def override_pykernel_method(monkeypatch):
    """Record the optimization level on the module handed to the runtime,
    then abort before compilation/execution."""

    def fn(method_name, new_method):
        monkeypatch.setattr(PyKernelDecorator, method_name, new_method)

    return fn


op = cudaq.spin.z(0) * cudaq.spin.z(1)
LAUNCHES = {
    "call":
        lambda l: bell(optimization_level=l),
    "sample":
        lambda l: cudaq.sample(empty, shots_count=10, optimization_level=l),
    "sample_async":
        lambda l: cudaq.sample_async(
            empty, shots_count=10, optimization_level=l),
    "observe":
        lambda l: cudaq.observe(empty, op, shots_count=10, optimization_level=l
                               ),
    "observe_async":
        lambda l: cudaq.observe_async(
            empty, op, shots_count=10, optimization_level=l),
    "run":
        lambda l: cudaq.run(bell, shots_count=10, optimization_level=l),
    "run_async":
        lambda l: cudaq.run_async(bell, shots_count=10, optimization_level=l),
    "get_state":
        lambda l: cudaq.get_state(empty, optimization_level=l),
    "get_state_async":
        lambda l: cudaq.get_state_async(empty, optimization_level=l),
    "draw":
        lambda l: cudaq.draw(empty, optimization_level=l),
    "get_unitary":
        lambda l: cudaq.get_unitary(empty, optimization_level=l),
    "estimate_resources":
        lambda l: cudaq.estimate_resources(empty, optimization_level=l),
}


@pytest.mark.parametrize("level", OPT_LEVELS)
@pytest.mark.parametrize("launch", LAUNCHES)
def test_prepare_call_stamps_optimization_level(override_pykernel_method,
                                                launch, level):
    # For all launch types, test that the `optimization_level` option gets
    # stamped onto the IR module and that the call gets routed to `prepare_call`
    # before execution.
    class ThowModule(Exception):

        def __init__(self, module, name):
            self.module = module
            self.name = name

    orig = PyKernelDecorator.prepare_call

    def mock_prepare_call(self, *args, **kwargs):
        _, module = orig(self, *args, **kwargs)
        raise ThowModule(module, self.uniqName)

    # Hijack the PyKernelDecorator.prepare_call method to throw the module after execution
    override_pykernel_method("prepare_call", mock_prepare_call)

    with pytest.raises(ThowModule) as e:
        LAUNCHES[launch](level)

    assert level == get_optimization_level_attr(e.value.module, e.value.name)


@pytest.mark.parametrize("level", [-1, 4, True, 1.5, "2"])
def test_invalid_levels(level):
    with pytest.raises(ValueError):
        bell(optimization_level=level)
    with pytest.raises(ValueError):
        cudaq.sample(empty, shots_count=10, optimization_level=level)


def test_kernel_level_0_wins_over_launch_level():
    # A kernel marked with `disable_quantum_optimization` stays at level 0,
    # whatever the launch asks for (the annotation wins, like in C++).
    @cudaq.kernel(disable_quantum_optimization=True)
    def marked():
        q = cudaq.qubit()
        h(q)

    _, module = marked.prepare_call(optimization_level=2)
    assert get_optimization_level_attr(module, marked.uniqName) == 0
    cudaq.sample(marked, shots_count=10, optimization_level=2)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_run_entry_point_has_launch_level(level):
    # `cudaq.run` launches the `.run` entry point, not the kernel itself.
    _, module = bell.prepare_call(optimization_level=level)
    assert get_optimization_level_attr(module, bell.uniqName + ".run") == level
    assert len(cudaq.run(bell, shots_count=2, optimization_level=level)) == 2


def test_run_entry_point_has_kernel_level_0():

    @cudaq.kernel(disable_quantum_optimization=True)
    def marked() -> bool:
        q = cudaq.qubit()
        h(q)
        h(q)
        return mz(q)

    _, module = marked.prepare_call(optimization_level=2)
    assert get_optimization_level_attr(module, marked.uniqName + ".run") == 0
    assert cudaq.run(marked, shots_count=2) == [False, False]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
