# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin

cudaq.set_target("nvidia-mqpu")

cudaq.mpi.initialize()
num_ranks = cudaq.mpi.num_ranks()
rank = cudaq.mpi.rank()

print('mpi is initialized? ', cudaq.mpi.is_initialized())
print('rank', rank, 'num_ranks', num_ranks)

qubit_count = 15
term_count = 100000

kernel = cudaq.make_kernel()
qubits = kernel.qalloc(qubit_count)
kernel.h(qubits[0])
for i in range(1, qubit_count):
    kernel.cx(qubits[0], qubits[i])

# We create a random Hamiltonian
hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows us to calculate the expectation value of the Hamiltonian with respect to a specified kernel.

# Single node, single GPU.
result = cudaq.observe(kernel, hamiltonian)
result.expectation()

# If we have multiple GPUs/ QPUs available, we can parallelize the workflow with the addition of an argument in the observe call.

# Single node, multi-GPU.
result = cudaq.observe(kernel, hamiltonian, execution=cudaq.parallel.thread)
result.expectation()

# Multi-node, multi-GPU.
result = cudaq.observe(kernel, hamiltonian, execution=cudaq.parallel.mpi)
result.expectation()

cudaq.mpi.finalize()
