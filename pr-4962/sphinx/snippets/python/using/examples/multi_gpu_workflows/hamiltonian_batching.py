# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import cudaq
from cudaq import spin

if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")
    exit(0)

cudaq.set_target("nvidia", option="mqpu")
cudaq.mpi.initialize()

qubit_count = 15
term_count = 100000


@cudaq.kernel
def kernel(n_qubits: int):

    qubits = cudaq.qvector(n_qubits)

    h(qubits[0])
    for i in range(1, n_qubits):
        x.ctrl(qubits[0], qubits[i])


# Create a random Hamiltonian
hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows calculation of the the expectation value of the Hamiltonian with respect to a specified kernel.

# Single node, single GPU.
result = cudaq.observe(kernel, hamiltonian, qubit_count)
result.expectation()

# If multiple GPUs/ QPUs are available, the computation can parallelize with the addition of an argument in the observe call.

# Single node, multi-GPU.
result = cudaq.observe(kernel,
                       hamiltonian,
                       qubit_count,
                       execution=cudaq.parallel.thread)
result.expectation()

# Multi-node, multi-GPU.
result = cudaq.observe(kernel,
                       hamiltonian,
                       qubit_count,
                       execution=cudaq.parallel.mpi)
result.expectation()

cudaq.mpi.finalize()
#[End Docs]
