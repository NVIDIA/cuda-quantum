#============================================================================== #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                          #
#                                                                               #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.      #
# The QAOA-GPT implementation in CUDA-Q is based on this paper:                 #
# https://arxiv.org/pdf/2504.16350                                              #
# Usage or reference of this code or algorithms requires citation of the paper: #
# Ilya Tyagin, Marwa Farag, Kyle Sherbert, Karunya Shirali, Yuri Alexeev,       #
# Ilya Safro "QAOA-GPT: Efficient Generation of Adaptive and Regular Quantum    #
# Approximate Optimization Algorithm Circuits", IEEE International Conference   #
# on Quantum Computing and Engineering (QCE), 2025.                             #
# ============================================================================= #

from cudaq import spin
import cudaq


def max_cut_ham(edges):
    """
    Generate a Hamiltonian for the Max-Cut problem.
    Args:
        edges: List of edges in the graph.
        weight: List of weights for each edge.
    Returns:
        Hamiltonian for the Max-Cut problem.
    """
    ham = 0.0

    for edge in range(len(edges)):

        qubitu = edges[edge][0]
        qubitv = edges[edge][1]
        weight = edges[edge][2]
        # Add a term to the Hamiltonian for the edge (u,v)
        ham += 0.5 * (weight * spin.z(qubitu) * spin.z(qubitv) -
                      weight * spin.i(qubitu) * spin.i(qubitv))

    return ham


# Collect coefficients from a spin operator so we can pass them to a kernel
def term_coefficients(ham: cudaq.SpinOperator) -> list[complex]:
    result = []
    for term in ham:
        result.append(term.evaluate_coefficient())
    return result


# Collect Pauli words from a spin operator so we can pass them to a kernel
def term_words(ham: cudaq.SpinOperator, qubits_num) -> list[str]:
    # Our kernel uses these words to apply exp_pauli to the entire state.
    # we hence ensure that each pauli word covers the entire space.

    result = []
    for term in ham:
        result.append(term.get_pauli_word(qubits_num))
    return result
