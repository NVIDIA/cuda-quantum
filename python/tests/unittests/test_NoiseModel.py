# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq


def test_Depol():

    cudaq.set_qpu('dm')

    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)

    depol = cudaq.DepolarizationChannel(.1)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], depol)

    counts = cudaq.sample(circuit, noise_model=noise, shots_count=100)
    assert (len(counts) == 2)
    assert ('0' in counts)
    assert ('1' in counts)
    assert (counts.count('0') + counts.count('1') == 100)

    counts = cudaq.sample(circuit)
    assert (len(counts) == 1)
    assert ('1' in counts)

    cudaq.set_qpu('qpp')
    counts = cudaq.sample(circuit)
    assert (len(counts) == 1)
    assert ('1' in counts)


def test_Kraus():

    cudaq.set_qpu('dm')

    k0 = np.array([[0.05773502691896258, 0.0], [0., -0.05773502691896258]],
                  dtype=np.complex128)
    k1 = np.array([[0., 0.05773502691896258], [0.05773502691896258, 0.]],
                  dtype=np.complex128)
    k2 = np.array([[0., -0.05773502691896258j], [0.05773502691896258j, 0.]],
                  dtype=np.complex128)
    k3 = np.array([[0.99498743710662, 0.0], [0., 0.99498743710662]],
                  dtype=np.complex128)

    depol = cudaq.KrausChannel([k0, k1, k2, k3])

    assert ((depol[0] == k0).all())
    assert ((depol[1] == k1).all())
    assert ((depol[2] == k2).all())
    assert ((depol[3] == k3).all())

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], depol)
    cudaq.set_noise(noise)
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)

    counts = cudaq.sample(circuit)
    assert (len(counts) == 2)
    assert ('0' in counts)
    assert ('1' in counts)

    cudaq.set_qpu('qpp')
    cudaq.unset_noise()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
