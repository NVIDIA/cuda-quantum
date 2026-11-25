# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.      #
# ============================================================================ #

import cudaq
import numpy as np

# Grover's algorithm is a quantum algorithm that finds with high probability 
# the unique input to a black box function that produces a particular output 
# value, using just O(sqrt(N)) evaluations of the function, where N is the 
# size of the function's domain.

@cudaq.kernel
def reflect_about_uniform(qs: cudaq.qview):
    """
    Reflection about the uniform superposition state.
    This is also known as the Grover diffusion operator.
    """
    ctrlQubits = qs.front(qs.size() - 1)
    lastQubit = qs.back()

    # compute_action(U, V) performs U V U_adjoint
    cudaq.compute_action(lambda: (h(qs), x(qs)), 
                         lambda: z.ctrl(ctrlQubits, lastQubit))

@cudaq.kernel
def oracle(qs: cudaq.qview, target_state: int):
    """
    The oracle marks the target state by flipping its phase.
    """
    def prepare_oracle():
        for i in range(qs.size()):
            # If the bit in target_state is 0, apply X to that qubit
            # so that the multi-controlled Z applies to the |0> state.
            if not ((target_state >> (qs.size() - i - 1)) & 1):
                x(qs[i])

    ctrlQubits = qs.front(qs.size() - 1)
    cudaq.compute_action(prepare_oracle, 
                         lambda: z.ctrl(ctrlQubits, qs.back()))

@cudaq.kernel
def grover(n_qubits: int, target_state: int):
    # Allocate the qubits
    qs = cudaq.qreg(n_qubits)

    # Calculate the number of iterations
    n_iterations = int(np.round(0.25 * np.pi * np.sqrt(2**n_qubits)))

    # Start in uniform superposition
    h(qs)

    # Iteratively apply oracle and diffusion operator
    for _ in range(n_iterations):
        oracle(qs, target_state)
        reflect_about_uniform(qs)

    # Measure to find the target state
    mz(qs)

# --- Execution ---

n_qubits = 4
target_state = 0b1011  # We are searching for 11

print(f"Searching for target state: {bin(target_state)}")

# Sample the kernel
result = cudaq.sample(grover, n_qubits, target_state)

# Output the results
measured_state = result.most_probable()
print(f"Measured state: {measured_state}")

# Validation
if int(measured_state, 2) == target_state:
    print("Success!")
else:
    print("Failure.")
