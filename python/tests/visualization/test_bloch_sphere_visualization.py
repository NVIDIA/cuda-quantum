# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pytest
import numpy as np

import cudaq

qutip = pytest.importorskip("qutip")

import io
from contextlib import redirect_stdout


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


# write sample kernels


@cudaq.kernel
def single_qubit_kernel():
    qubit = cudaq.qubit()
    h(qubit)
    x(qubit)
    mz(qubit)


@cudaq.kernel
def two_qubit_kernel():
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(np.pi / 2., qvector[1])
    mz(qvector)


# basic tests


def test_visualization_bad_state():
    with pytest.raises(Exception) as err:
        invalid_state = np.array([1, 0])
        cudaq.add_to_bloch_sphere(invalid_state)


def test_visualization_invalid_state():
    with pytest.raises(Exception) as err:
        two_qubit_state = cudaq.get_state(two_qubit_kernel)
        cudaq.add_to_bloch_sphere(two_qubit_state)


def test_visualization_single_qubit_no_sphere():
    single_qubit_state = cudaq.get_state(single_qubit_kernel)
    b = cudaq.add_to_bloch_sphere(single_qubit_state)
    assert isinstance(b, qutip.Bloch)


def test_visualization_single_qubit_shere():
    sph = qutip.Bloch()
    # generate a random density matrix with qutip and add to sphere
    sph.add_states(qutip.rand_dm(2))

    b = cudaq.add_to_bloch_sphere(cudaq.get_state(single_qubit_kernel),
                                  existing_sphere=sph)
    assert isinstance(b, qutip.Bloch)


# TODO: refactor this part later, but run the same tests as before with density matrix backend

cudaq.set_target("density-matrix-cpu")


def test_visualization_bad_state_dm():
    with pytest.raises(Exception) as err:
        # valid density matrix, but it is not a cudaq.State type.
        invalid_state = np.array([[0.5, 0], [0, 0.5]])
        cudaq.add_to_bloch_sphere(invalid_state)


def test_visualization_invalid_state_dm():
    with pytest.raises(Exception) as err:
        two_qubit_state = cudaq.get_state(two_qubit_kernel)
        cudaq.add_to_bloch_sphere(two_qubit_state)


def test_visualization_single_qubit_no_sphere_dm():
    single_qubit_state = cudaq.get_state(single_qubit_kernel)
    b = cudaq.add_to_bloch_sphere(single_qubit_state)
    assert isinstance(b, qutip.Bloch)


def test_visualization_single_qubit_shere_dm():
    sph = qutip.Bloch()
    # generate a random density matrix with qutip and add to sphere
    sph.add_states(qutip.rand_dm(2))

    b = cudaq.add_to_bloch_sphere(cudaq.get_state(single_qubit_kernel),
                                  existing_sphere=sph)
    assert isinstance(b, qutip.Bloch)


# reset target to prevent interference with further tests
cudaq.reset_target()


def test_show_bloch_no_data():
    # must return nothing when nothing is supplied
    op = 'ERR'
    with io.StringIO() as buf, redirect_stdout(buf):
        cudaq.show()
        op = buf.getvalue()

    assert "Nothing to display." in op


def test_show_bloch_bad_sphere():
    with pytest.raises(TypeError) as err:
        cudaq.show(np.array([0, 1]))


def test_show_bloch_bad_rows():
    # make a few dummy spheres
    sphList = []
    for _ in range(6):
        sph = qutip.Bloch()
        sph.add_states(qutip.rand_dm(2))
        sphList.append(sph)

    # insufficient space to show all spheres
    with pytest.raises(Exception) as err:
        cudaq.show(sphList, ncols=2, nrows=2)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
