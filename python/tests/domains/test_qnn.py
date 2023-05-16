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
    n_qubits = 2
    n_samples = 5000
    h = cudaq.spin.z(0)
    n_parameters = n_qubits * 3
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(n_samples,
                                                         n_parameters))
    np.random.seed(1)

    kernel, params = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    qubits_list = list(range(n_qubits))
    for i in range(n_qubits):
        kernel.rx(params[i], qubits[i])
    for i in range(n_qubits):
        kernel.ry(params[i + n_qubits], qubits[i])
    for i in range(n_qubits):
        kernel.rz(params[i + n_qubits * 2], qubits[i])
    for q1, q2 in zip(qubits_list[0::2], qubits_list[1::2]):
        kernel.cz(qubits[q1], qubits[q2])

    exp_vals = cudaq.observe_n(kernel, h, parameters)
    assert len(exp_vals) == n_samples 
    data = np.asarray([e.expectation_z() for e in exp_vals])
    # Test a collection of computed exp vals.
    assert np.isclose(data[0], 0.44686141)
    assert np.isclose(data[1], 0.5014559)
    assert np.isclose(data[2], 0.6815774)
    assert np.isclose(data[-3], 0.50511996)
    assert np.isclose(data[-2], 0.54314517)
    assert np.isclose(data[-1], 0.33752631)
