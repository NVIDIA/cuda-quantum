# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq


def uccsd_get_excitation_list(n_electrons, n_qubits):

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    singles_alpha = []
    singles_beta = []
    doubles_mixed = []
    doubles_alpha = []
    doubles_beta = []

    if n_electrons % 2 != 0:
        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [
            i * 2 + n_electrons + 1 for i in range(n_virtual)
        ]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied - 1)]
        virtual_beta_indices = [2 * n_occupied - 1]
        virtual_beta_indices += [
            i * 2 + 1 + n_electrons + 1 for i in range(n_virtual)
        ]

    else:
        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [i * 2 + n_electrons for i in range(n_virtual)]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
        virtual_beta_indices = [
            i * 2 + 1 + n_electrons for i in range(n_virtual)
        ]

    #occupied orbital is alpha, virtual orbital is alpha
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_alpha.append((p, q))

    # occupied orbital is beta and virtual orbital is beta
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_beta.append((p, q))

    #Mixed spin double excitation
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_mixed.append((p, q, r, s))

    # same spin double excitation
    n_occ_alpha = len(occupied_alpha_indices)
    n_occ_beta = len(occupied_beta_indices)
    n_virt_alpha = len(virtual_alpha_indices)
    n_virt_beta = len(virtual_beta_indices)

    for p in range(n_occ_alpha - 1):
        for q in range(p + 1, n_occ_alpha):
            for r in range(n_virt_alpha - 1):
                for s in range(r + 1, n_virt_alpha):

                    # Same spin: all alpha
                    doubles_alpha.append((occupied_alpha_indices[p],occupied_alpha_indices[q],\
                                     virtual_alpha_indices[r],virtual_alpha_indices[s]))

    for p in range(n_occ_beta - 1):
        for q in range(p + 1, n_occ_beta):
            for r in range(n_virt_beta - 1):
                for s in range(r + 1, n_virt_beta):

                    # Same spin: all beta
                    doubles_beta.append((occupied_beta_indices[p],occupied_beta_indices[q],\
                                     virtual_beta_indices[r],virtual_beta_indices[s]))

    return singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta


#########################################################################


def uccsd_num_parameters(n_electrons, n_qubits):
    # Compute the size of theta parameters for all UCCSD excitation.

    singles_alpha,singles_beta,doubles_mixed,doubles_alpha,doubles_beta=\
        uccsd_get_excitation_list(n_electrons, n_qubits)

    length_alpha_singles = len(singles_alpha)
    length_beta_singles = len(singles_beta)
    length_mixed_doubles = len(doubles_mixed)
    length_alpha_doubles = len(doubles_alpha)
    length_beta_doubles = len(doubles_beta)

    singles = length_alpha_singles + length_beta_singles
    doubles = length_mixed_doubles + length_alpha_doubles + length_beta_doubles
    total = singles + doubles

    return sum((singles, doubles, total))


def single_excitation_gate(kernel, qubits: cudaq.qvector, p_occ: int,
                           q_virt: int, theta: list[float]):

    # Y_p X_q
    kernel.rx(np.pi / 2.0, qubits[p_occ])
    kernel.h(qubits[q_virt])

    for i in range(p_occ, q_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.h(qubits[q_virt])
    kernel.rx(-np.pi / 2.0, qubits[p_occ])

    # -X_p Y_q
    kernel.h(qubits[p_occ])
    kernel.rx(np.pi / 2.0, qubits[q_virt])

    for i in range(p_occ, q_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(-0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.rx(-np.pi / 2.0, qubits[q_virt])
    kernel.h(qubits[p_occ])


def double_excitation_gate_opt(kernel, qubits: cudaq.qvector, p_occ: int,
                               q_occ: int, r_virt: int, s_virt: int,
                               theta: list[float]):

    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ, j_occ = p_occ, q_occ
        a_virt, b_virt = r_virt, s_virt

    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ, j_occ = q_occ, p_occ
        a_virt, b_virt = s_virt, r_virt

    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ, j_occ = p_occ, q_occ
        a_virt, b_virt = s_virt, r_virt
        theta *= -1.0

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ, j_occ = q_occ, p_occ
        a_virt, b_virt = r_virt, s_virt
        theta *= -1.0

    #Block I: x_i x_j x_a y_b + x_i x_j y_a x_b + x_i y_i y_a y_b - x_i y_j x_a x_b
    #Block II: - y_i x_j x_a x_b +y_i x_j y_a y_b - y_i x_j x_a x_b - y_i y_j y_a x_b

    kernel.h(qubits[i_occ])
    kernel.h(qubits[j_occ])
    kernel.h(qubits[a_virt])
    kernel.rx(np.pi / 2.0, qubits[b_virt])

    for i in range(i_occ, j_occ):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])

    kernel.rx(-np.pi / 2.0, qubits[b_virt])
    kernel.h(qubits[a_virt])

    kernel.rx(np.pi / 2.0, qubits[a_virt])
    kernel.h(qubits[b_virt])

    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.rx(-np.pi / 2.0, qubits[a_virt])
    kernel.h(qubits[j_occ])

    kernel.rx(np.pi / 2.0, qubits[j_occ])
    kernel.h(qubits[a_virt])

    for i in range(i_occ, j_occ):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])

    kernel.h(qubits[b_virt])
    kernel.h(qubits[a_virt])

    kernel.rx(np.pi / 2.0, qubits[a_virt])
    kernel.rx(np.pi / 2.0, qubits[b_virt])

    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.rx(-np.pi / 2.0, qubits[j_occ])
    kernel.h(qubits[i_occ])

    kernel.rx(np.pi / 2.0, qubits[i_occ])
    kernel.h(qubits[j_occ])

    for i in range(i_occ, j_occ):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])

    kernel.rx(-np.pi / 2.0, qubits[b_virt])
    kernel.rx(-np.pi / 2.0, qubits[a_virt])

    kernel.h(qubits[a_virt])
    kernel.h(qubits[b_virt])

    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.h(qubits[b_virt])
    kernel.h(qubits[j_occ])

    kernel.rx(np.pi / 2.0, qubits[j_occ])
    kernel.rx(np.pi / 2.0, qubits[b_virt])

    for i in range(i_occ, j_occ):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])

    kernel.rx(-np.pi / 2.0, qubits[b_virt])
    kernel.h(qubits[a_virt])

    kernel.rx(np.pi / 2.0, qubits[a_virt])
    kernel.h(qubits[b_virt])

    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        kernel.cx(qubits[i], qubits[i + 1])

    kernel.rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        kernel.cx(qubits[i - 1], qubits[i])
    kernel.cx(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        kernel.cx(qubits[i - 1], qubits[i])

    kernel.h(qubits[b_virt])
    kernel.rx(-np.pi / 2.0, qubits[a_virt])
    kernel.rx(-np.pi / 2.0, qubits[j_occ])
    kernel.rx(-np.pi / 2.0, qubits[i_occ])


def uccsd(kernel, qubits: cudaq.qvector, thetas: list[float], n_electrons: int,
          n_qubits: int):

    # This function generates a quantum circuit for the VQE-UCCSD ansatz
    # To construct an efficient quantum circuit with minimum number of `cnot`,
    # we use gate cancellation.

    # Generate the relevant UCCSD excitation list indices
    singles_alpha,singles_beta,doubles_mixed,doubles_alpha,doubles_beta=\
        uccsd_get_excitation_list(n_electrons, n_qubits)

    n_alpha_singles = len(singles_alpha)
    n_beta_singles = len(singles_beta)
    n_mixed_doubles = len(doubles_mixed)
    n_alpha_doubles = len(doubles_alpha)
    n_beta_doubles = len(doubles_beta)
    total = n_alpha_singles + n_beta_singles + n_mixed_doubles + n_beta_doubles + n_alpha_doubles

    thetaCounter = 0
    for i in range(n_alpha_singles):
        single_excitation_gate(kernel, qubits, singles_alpha[i][0],
                               singles_alpha[i][1], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_singles):
        single_excitation_gate(kernel, qubits, singles_beta[i][0],
                               singles_beta[i][1], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_mixed_doubles):
        double_excitation_gate_opt(kernel, qubits, doubles_mixed[i][0],
                                   doubles_mixed[i][1], doubles_mixed[i][2],
                                   doubles_mixed[i][3], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_alpha_doubles):
        double_excitation_gate_opt(kernel, qubits, doubles_alpha[i][0],
                                   doubles_alpha[i][1], doubles_alpha[i][2],
                                   doubles_alpha[i][3], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_doubles):
        double_excitation_gate_opt(kernel, qubits, doubles_beta[i][0],
                                   doubles_beta[i][1], doubles_beta[i][2],
                                   doubles_beta[i][3], thetas[thetaCounter])
        thetaCounter += 1
