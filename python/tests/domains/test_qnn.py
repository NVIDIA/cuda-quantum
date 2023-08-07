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

skipIfNoMQPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-mqpu')),
    reason="nvidia-mqpu backend not available"
)


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

    exp_vals = cudaq.observe(kernel, h, parameters)
    assert len(exp_vals) == samples_count
    data = np.asarray([e.expectation_z() for e in exp_vals])
    # Test a collection of computed exp vals.
    assert np.isclose(data[0], 0.44686141)
    assert np.isclose(data[1], 0.5014559)
    assert np.isclose(data[2], 0.6815774)
    assert np.isclose(data[-3], 0.50511996)
    assert np.isclose(data[-2], 0.54314517)
    assert np.isclose(data[-1], 0.33752631)

@skipIfNoMQPU
def test_observeAsync_QNN():
    target = cudaq.get_target('nvidia-mqpu')

    cudaq.set_target(target)
    num_qpus = target.num_qpus()

    n_qubits = 2
    n_samples = 2
    h = cudaq.spin.z(0)

    n_parameters = n_qubits * 3
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(n_samples,
                                                         n_parameters))

    kernel, params = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    qubits_list = list(range(n_qubits))
    for i in range(n_qubits):
        kernel.rx(params[i], qubits[i])
    for i in range(n_qubits):
        kernel.ry(params[i + n_qubits], qubits[i])
    for i in range(n_qubits):
        kernel.rz(params[i + n_qubits * 2], qubits[i])

    xi = np.split(parameters, num_qpus)
    asyncresults = []
    for i in range(len(xi)):
        for j in range(xi[i].shape[0]):
            asyncresults.append(
                cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=i))

    expvals = []
    for res in asyncresults:
        expvals.append(res.get().expectation_z())

    assert np.allclose(np.asarray([0.44686155, 0.50145603]),
                       np.asarray(expvals))
    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])