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


def test_simpleObserveN_QNN():
    qubit_count = 2
    samples_count = 5000
    h = cudaq.spin.z(0)
    parameters_count = qubit_count * 3
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(samples_count,
                                                         parameters_count))
    np.random.seed(1)

    # Build up a parameterized kernel composed of a layer of
    # Rx, Ry, Rz, then CZs.
    kernel, params = cudaq.make_kernel(list)
    qubits = kernel.qalloc(qubit_count)
    qubits_list = list(range(qubit_count))
    for i in range(qubit_count):
        kernel.rx(params[i], qubits[i])
    for i in range(qubit_count):
        kernel.ry(params[i + qubit_count], qubits[i])
    for i in range(qubit_count):
        kernel.rz(params[i + qubit_count * 2], qubits[i])
    for q1, q2 in zip(qubits_list[0::2], qubits_list[1::2]):
        kernel.cz(qubits[q1], qubits[q2])

    exp_vals = cudaq.observe_n(kernel, h, parameters)
    assert len(exp_vals) == samples_count
    data = np.asarray([e.expectation_z() for e in exp_vals])
    # Test a collection of computed exp vals.
    assert np.isclose(data[0], 0.44686141)
    assert np.isclose(data[1], 0.5014559)
    assert np.isclose(data[2], 0.6815774)
    assert np.isclose(data[-3], 0.50511996)
    assert np.isclose(data[-2], 0.54314517)
    assert np.isclose(data[-1], 0.33752631)
