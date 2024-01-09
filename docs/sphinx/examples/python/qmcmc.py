# To run with the default parameters use:
# python qmcmc.py
#
# For non-default values of the parameters use:
# python3 qmcmc.py --num_iterations 30 --nqubits 13 --temperature 0.08 --shots_count 20
#
# This code is based on the quantum enhanced Markov
# Chain Monte Carlo (QMCMC) algorithm presented in the
# paper https://arxiv.org/pdf/2203.12497.pdf
# The Hamiltonian here is exponentiated using the first
# order Trotterization.

import cudaq
from cudaq import spin
import numpy as np
import random
import argparse


# Generate a random bitstring of given length
def generate_random_bitstring(length):
    return ''.join(random.choice('01') for _ in range(length))


# Initialize the circuit with a given bitstring
def initial_state(qc, q, bitstring):
    for i, bit in enumerate(bitstring):
        if bit == '1':
            qc.x(q[i])
    return qc


# Calculate energy for a spin configuration
def calculate_energy(bitstring, J, h):
    # Map the bitstring to the spin configuration first
    nqubits = len(bitstring)
    s = [-1 if bit == '1' else 1 for bit in bitstring]
    sum_E = 0.0
    for k in range(1, nqubits):
        for j in range(k + 1, nqubits):
            sum_E -= J[j][k] * int(s[j]) * int(s[k])

    for j in range(1, nqubits):
        sum_E -= h[j] * int(s[j])

    return sum_E


# Create a list of Hamiltonians if order matters
def generate_H(gamma, nqubits, J, h):
    # Create an empty list for storing the Hamiltonians
    H_list = []
    # Initialize Hamiltonians with 0 coefficients
    # These are needed to calculate alpha
    H_prob = 0 * spin.i(0)
    H_mix = 0 * spin.i(0)
    count_problem_terms = 0

    # Problem Hamiltonian
    for k in range(1, nqubits):
        for j in range(k + 1, nqubits):
            H_prob -= J[j][k] * spin.z(j) * spin.z(k)
            H_list.append((gamma - 1.0) * J[j][k] * spin.z(j) * spin.z(k))
            count_problem_terms = count_problem_terms + 1

    for j in range(1, nqubits):
        H_prob -= h[j] * spin.z(j)
        H_list.append((gamma - 1.0) * h[j] * spin.z(j))
        count_problem_terms = count_problem_terms + 1

    # Mixer Hamiltonian
    for j in range(1, nqubits):
        H_mix += spin.x(j)
        H_list.append(gamma * spin.x(j))

    # Final Hamiltonian
    alpha = np.linalg.norm(H_mix.to_matrix()) / np.linalg.norm(
        H_prob.to_matrix())
    for i in range(count_problem_terms):
        H_list[i] = H_list[i] * alpha

    ordered_H = []
    for term in H_list:
        adjTerm = term
        for i in range(term.get_qubit_count(), nqubits):
            adjTerm = term * spin.i(i)
        ordered_H.append(adjTerm)

    return ordered_H


# This is the first-order Trotter gate decomposition for
# an ordered list of Hamiltonians
def trotter_circuit(kernel, qreg, hk, dt, n_qubits):
    for term in hk:
        kernel.exp_pauli(dt * term.get_coefficient().real, qreg, term)

    return kernel


def main():

    random.seed(41)

    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Quantum Enhanced Markov Chain Monte Carlo')
    parser.add_argument('--num_iterations',
                        type=int,
                        default=20,
                        help='Number of iterations')
    parser.add_argument('--nqubits',
                        type=int,
                        default=10,
                        help='Number of qubits')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.1,
                        help='Temperature')
    parser.add_argument('--shots_count',
                        type=int,
                        default=10,
                        help='Number of shots')
    args = parser.parse_args()

    num_iterations = args.num_iterations
    nqubits = args.nqubits
    T = args.temperature
    shots_count = args.shots_count

    # Specify couplings J's and fields h's.
    J = np.ones((nqubits, nqubits))
    h = np.ones(nqubits)

    # Generate a random initial bitstring instead of a spin config
    # Convert it to its equivalent spin config later when needed
    s = generate_random_bitstring(nqubits)

    # Iteration loop
    iter = 0
    while iter < num_iterations:
        # Create a circuit and allocate qubits
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(nqubits)

        # Propose jump (quantum step 1)
        gamma = np.random.uniform(0.25, 0.6)
        t = np.random.uniform(2, 20)
        # Create a list of ordered Hamiltonian terms
        H_list = generate_H(gamma, nqubits, J, h)

        # Load initial bitstring into the circuit
        kernel = initial_state(kernel, q, s)
        # First order Trotterization
        kernel = trotter_circuit(kernel, q, H_list, t, nqubits)

        # Sample the distribution
        result = cudaq.sample(kernel, shots_count=shots_count)
        s_prime = result.most_probable()

        # Accept/reject jump (classical step 2)
        es = calculate_energy(s, J, h)
        es_prime = calculate_energy(s_prime, J, h)
        print(s, es, s_prime, es_prime)
        A = min(1, np.exp((es - es_prime)) / T)
        if (A >= random.uniform(0, 1)):
            s = s_prime

        # Increment the iteration count
        iter = iter + 1


if __name__ == '__main__':
    main()
