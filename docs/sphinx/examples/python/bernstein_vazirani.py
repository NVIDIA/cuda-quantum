# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

import cudaq
import random

# The Bernstein-Vazirani algorithm allows us to find a hidden bitstring
# `s` in a single query to an oracle that computes f(x) = s · x (mod 2).

def generate_random_bitstring(length):
    return [random.randint(0, 1) for _ in range(length)]

@cudaq.kernel
def oracle(qvector: cudaq.qview, auxillary_qubit: cudaq.qubit, hidden_bitstring: list[int]):
    for i, bit in enumerate(hidden_bitstring):
        if bit == 1:
            # Apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(qvector[i], auxillary_qubit)

@cudaq.kernel
def bernstein_vazirani(hidden_bitstring: list[int]):
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qvector = cudaq.qreg(len(hidden_bitstring))
    
    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()

    # Prepare the auxillary qubit in the |-> state.
    x(auxillary_qubit)
    h(auxillary_qubit)

    # Place the rest of the qubits in a superposition state.
    h(qvector)

    # Query the oracle.
    oracle(qvector, auxillary_qubit, hidden_bitstring)

    # Apply another set of Hadamards to the qubits to 
    # extract the hidden bitstring.
    h(qvector)

    # Measure the qubits.
    mz(qvector)

# Setup the experiment
qubit_count = 5
hidden_bitstring = generate_random_bitstring(qubit_count)
print(f"Encoded bitstring  = {''.join(map(str, hidden_bitstring))}")

# Sample the kernel
result = cudaq.sample(bernstein_vazirani, hidden_bitstring)

# Output the results
measured_bitstring = result.most_probable()
print(f"Measured bitstring = {measured_bitstring}")

# Validation
if measured_bitstring == ''.join(map(str, hidden_bitstring)):
    print("Success!")
else:
    print("Failure.")
