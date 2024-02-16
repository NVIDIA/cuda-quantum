# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq
from ...kernels.uccsd import uccsd_num_parameters

@cudaq.kernel
def single_excitation_gate(qubits: cudaq.qview, p_occ: int, q_virt: int,
                           theta: float):

    # Y_p X_q
    rx(np.pi / 2.0, qubits[p_occ])
    h(qubits[q_virt])

    for i in range(p_occ, q_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[q_virt])
    rx(-np.pi / 2.0, qubits[p_occ])

    # -X_p Y_q
    h(qubits[p_occ])
    rx(np.pi / 2.0, qubits[q_virt])

    for i in range(p_occ, q_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-np.pi / 2.0, qubits[q_virt])
    h(qubits[p_occ])

@cudaq.kernel
def double_excitation_gate_opt(qubits: cudaq.qview, p_occ: int,
                               q_occ: int, r_virt: int, s_virt: int,
                               theta: float):
    i_occ, j_occ = (0, 0)
    a_virt, b_virt = (0, 0)

    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ, j_occ = (p_occ, q_occ)
        a_virt, b_virt = (r_virt, s_virt)

    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ, j_occ = (q_occ, p_occ)
        a_virt, b_virt = (s_virt, r_virt)

    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ, j_occ = (p_occ, q_occ)
        a_virt, b_virt = (s_virt, r_virt)
        theta = -theta

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ, j_occ = (q_occ, p_occ)
        a_virt, b_virt = (r_virt, s_virt)
        theta = -theta

    #Block I: x_i x_j x_a y_b + x_i x_j y_a x_b + x_i y_i y_a y_b - x_i y_j x_a x_b
    #Block II: - y_i x_j x_a x_b +y_i x_j y_a y_b - y_i x_j x_a x_b - y_i y_j y_a x_b

    h(qubits[i_occ])
    h(qubits[j_occ])
    h(qubits[a_virt])
    rx(np.pi / 2.0, qubits[b_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-np.pi / 2.0, qubits[b_virt])
    h(qubits[a_virt])

    rx(np.pi / 2.0, qubits[a_virt])
    h(qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-np.pi / 2.0, qubits[a_virt])
    h(qubits[j_occ])

    rx(np.pi / 2.0, qubits[j_occ])
    h(qubits[a_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    h(qubits[b_virt])
    h(qubits[a_virt])

    rx(np.pi / 2.0, qubits[a_virt])
    rx(np.pi / 2.0, qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-np.pi / 2.0, qubits[j_occ])
    h(qubits[i_occ])

    rx(np.pi / 2.0, qubits[i_occ])
    h(qubits[j_occ])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-np.pi / 2.0, qubits[b_virt])
    rx(-np.pi / 2.0, qubits[a_virt])

    h(qubits[a_virt])
    h(qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[b_virt])
    h(qubits[j_occ])

    rx(np.pi / 2.0, qubits[j_occ])
    rx(np.pi / 2.0, qubits[b_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-np.pi / 2.0, qubits[b_virt])
    h(qubits[a_virt])

    rx(np.pi / 2.0, qubits[a_virt])
    h(qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * theta, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[b_virt])
    rx(-np.pi / 2.0, qubits[a_virt])
    rx(-np.pi / 2.0, qubits[j_occ])
    rx(-np.pi / 2.0, qubits[i_occ])


@cudaq.kernel
def uccsd(qubits: cudaq.qview, thetas: list[float], n_electrons: int,
          n_qubits: int):

    # This function generates a quantum circuit for the VQE-UCCSD ansatz
    # To construct an efficient quantum circuit with minimum number of `cnot`,
    # we use gate cancellation.

    # Generate the relevant UCCSD excitation list indices
    singles_alpha,singles_beta,doubles_mixed,doubles_alpha,doubles_beta=\
        cudaq.uccsd_get_excitation_list(n_electrons, n_qubits)

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
