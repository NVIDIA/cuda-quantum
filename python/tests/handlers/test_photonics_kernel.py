# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("photonics")
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def test_qudit():

    @cudaq.kernel
    def kernel():
        q = qudit(level=4)
        plus(q)
        plus(q)
        plus(q)
        mz(q)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert '3' in counts

    state = cudaq.get_state(kernel)
    state.dump()
    # TODO: Add check for 'state' object


def test_qudit_list():

    @cudaq.kernel
    def kernel():
        qutrits = [qudit(3) for _ in range(2)]
        plus(qutrits[0])
        plus(qutrits[1])
        mz(qutrits)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert '11' in counts


def test_qudit_invalid():

    @cudaq.kernel
    def kernel():
        q = [i for i in range(2)]
        plus(q[0])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert "Qudit level not set" in repr(e)


def test_supported_gates():

    @cudaq.kernel
    def kernel():
        quds = [qudit(5) for _ in range(3)]

        plus(quds[0])
        plus(quds[1])
        plus(quds[2])

        phase_shift(quds[1], 0.5)
        beam_splitter(quds[0], quds[1], 1.3)

        mz(quds)

    counts = cudaq.sample(kernel)
    counts.dump()


def test_kernel_with_args():

    @cudaq.kernel
    def kernel(theta: float):
        q = qudit(4)
        plus(q)
        phase_shift(q, theta)
        mz(q)

    result = cudaq.sample(kernel, 0.5)
    result.dump()


def test_target_change():

    @cudaq.kernel
    def kernel():
        q = qudit(level=2)
        plus(q)
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

    cudaq.set_target("photonics")
    res = cudaq.sample(kernel)
    assert len(res) == 1 and '1' in res

    cudaq.reset_target()
    res = cudaq.sample(bell_pair)
    assert len(res) == 2 and '00' in res and '11' in res


def test_unsupported_gates():

    @cudaq.kernel
    def kernel():
        q = qudit(6)
        h(q)

    with pytest.raises(NameError) as e:
        cudaq.sample(kernel)
    assert "name 'h' is not defined" in repr(e)


@pytest.mark.skip(
    reason=
    "This failing test causes a crash on subsequent test when running in CI. \
    (python/tests/kernel/test_adjoint_operations.py::test_sdg_1_state) \
    Need to check if the fix for https://github.com/NVIDIA/cuda-quantum/issues/1717 also solves this."
)
def test_unsupported_types():

    @cudaq.kernel
    def kernel1():
        q = cudaq.qubit()
        plus(q)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel1)
    assert "Qudit level not set" in repr(e)

    @cudaq.kernel
    def kernel2():
        q = cudaq.qvector(2)
        plus(q[0])

    with pytest.raises(Exception) as e:
        cudaq.sample(kernel2)
    assert "Qudit level not set" in repr(e)


def test_target_handler():
    cudaq.reset_target()

    @cudaq.kernel
    def kernel():
        q = qudit(level=3)
        plus(q)
        mz(q)

    with pytest.raises(RuntimeError):
        cudaq.sample(kernel)


def test_qudit_level():

    @cudaq.kernel
    def kernel():
        q1 = qudit(2)
        q2 = qudit(3)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert "qudits must be of same level" in repr(e)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
