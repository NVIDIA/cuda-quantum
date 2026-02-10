# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from typing import List

import cudaq


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("orca-photonics")
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_qudit():

    @cudaq.kernel
    def kernel():
        q = qudit(level=4)
        create(q)
        create(q)
        create(q)
        mz(q)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert '3' in counts

    state = cudaq.StateMemoryView(cudaq.get_state(kernel))
    state.dump()
    assert 4 == state.__len__()


def test_qudit_list():

    @cudaq.kernel
    def kernel():
        qumodes = [qudit(3) for _ in range(2)]
        create(qumodes[0])
        create(qumodes[1])
        mz(qumodes)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert '11' in counts


def test_qudit_invalid():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel():
            q = [i for i in range(2)]
            create(q[0])

        cudaq.sample(kernel)
    assert "Qudit level not set" in repr(e)


def test_supported_gates():

    @cudaq.kernel
    def kernel():
        qumodes = [qudit(5) for _ in range(3)]

        create(qumodes[0])
        create(qumodes[1])
        create(qumodes[2])

        phase_shift(qumodes[1], 0.5)
        beam_splitter(qumodes[0], qumodes[1], 1.3)

        mz(qumodes)

    counts = cudaq.sample(kernel)
    counts.dump()


def test_kernel_with_args():
    """Test that `PhotonicsHandler` supports basic arguments. 
       The check here is that all the test kernels run successfully."""

    @cudaq.kernel
    def kernel_1f(theta: float):
        q = qudit(4)
        create(q)
        phase_shift(q, theta)
        mz(q)

    result = cudaq.sample(kernel_1f, 0.5)
    result.dump()

    state = cudaq.StateMemoryView(cudaq.get_state(kernel_1f, 0.5))
    state.dump()

    @cudaq.kernel
    def kernel_2f(theta: float, phi: float):
        qumodes = [qudit(3) for _ in range(2)]
        create(qumodes[0])
        phase_shift(qumodes[0], theta)
        beam_splitter(qumodes[0], qumodes[1], phi)
        mz(qumodes)

    result = cudaq.sample(kernel_2f, 0.7854, 0.3927)
    result.dump()

    state = cudaq.StateMemoryView(cudaq.get_state(kernel_2f, 0.7854, 0.3927))
    state.dump()

    @cudaq.kernel
    def kernel_list(angles: List[float]):
        qumodes = [qudit(2) for _ in range(3)]
        create(qumodes[0])
        phase_shift(qumodes[1], angles[0])
        phase_shift(qumodes[2], angles[1])
        mz(qumodes)

    result = cudaq.sample(kernel_list, [0.5236, 1.0472])
    result.dump()

    state = cudaq.StateMemoryView(cudaq.get_state(kernel_list,
                                                  [0.5236, 1.0472]))
    state.dump()


def test_target_change():

    @cudaq.kernel
    def kernel():
        q = qudit(level=2)
        create(q)
        mz(q)

    res = cudaq.sample(kernel)
    assert len(res) == 1 and '1' in res

    cudaq.reset_target()

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])

    res = cudaq.sample(bell_pair)
    assert len(res) == 2 and '00' in res and '11' in res

    cudaq.set_target("orca-photonics")
    res = cudaq.sample(kernel)
    assert len(res) == 1 and '1' in res

    cudaq.reset_target()
    res = cudaq.sample(bell_pair)
    assert len(res) == 2 and '00' in res and '11' in res


def test_unsupported_gates():

    with pytest.raises(NameError) as e:

        @cudaq.kernel
        def kernel():
            q = qudit(6)
            h(q)

        cudaq.sample(kernel)
    assert "name 'h' is not defined" in repr(e)


@pytest.mark.skip(
    reason=
    "This failing test causes a crash on subsequent test when running in CI. \
    (python/tests/kernel/test_adjoint_operations.py::test_sdg_1_state) \
    Need to check if the fix for https://github.com/NVIDIA/cuda-quantum/issues/1717 also solves this."
)
def test_unsupported_types():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel1():
            q = cudaq.qubit()
            create(q)

        cudaq.sample(kernel1)
    assert "Qudit level not set" in repr(e)

    with pytest.raises(Exception) as e:

        @cudaq.kernel
        def kernel2():
            q = cudaq.qvector(2)
            create(q[0])

        cudaq.sample(kernel2)
    assert "Qudit level not set" in repr(e)


def test_target_handler():
    cudaq.reset_target()

    with pytest.raises(RuntimeError):

        @cudaq.kernel
        def kernel():
            q = qudit(level=3)
            create(q)
            mz(q)

        cudaq.sample(kernel)


def test_qudit_level():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel():
            q1 = qudit(2)
            q2 = qudit(3)

        cudaq.sample(kernel)
    assert "qudits must be of same level" in repr(e)


def test_run_unsupported():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel():
            q = qudit(level=2)
            create(q)
            mz(q)

        cudaq.run(kernel, shots_count=10)
    assert "Unsupported target" in repr(e)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
