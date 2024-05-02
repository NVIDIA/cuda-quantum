# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations
import numpy as np
import cudaq

## [PYTHON_VERSION_FIX]
## To support Python v3.8, using `typing.List[float]` instead of `list[float]`
from typing import List


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


def uccsd_num_parameters(n_electrons, n_qubits):
    """
    For the given number of electrons and qubits, return the required number
    of UCCSD parameters."
    """
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
    return singles + doubles


@cudaq.kernel
def single_excitation(qubits: cudaq.qview, p_occ: int, q_virt: int,
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
def double_excitation_opt(qubits: cudaq.qview, p_occ: int, q_occ: int,
                          r_virt: int, s_virt: int, theta: float):

    i_occ = 0
    j_occ = 0
    a_virt = 0
    b_virt = 0
    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = r_virt
        b_virt = s_virt

    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = s_virt
        b_virt = r_virt

    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = s_virt
        b_virt = r_virt
        # theta *= -1.0 FIXME
        theta *= -1.

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = r_virt
        b_virt = s_virt
        theta *= -1.0
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
def uccsd_odd_electrons(qubits: cudaq.qview, thetas: List[float],
                        n_electrons: int, n_qubits: int):
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
    virtual_alpha_indices = [i * 2 + n_electrons + 1 for i in range(n_virtual)]

    occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied - 1)]
    virtual_beta_indices = [0 for k in range(n_virtual + 1)]
    virtual_beta_indices[0] = 2 * n_occupied - 1
    for i in range(n_virtual):
        virtual_beta_indices[i + 1] = i * 2 + 1 + n_electrons

    lenOccA = len(occupied_alpha_indices)
    lenOccB = len(occupied_beta_indices)
    lenVirtA = len(virtual_alpha_indices)
    lenVirtB = len(virtual_beta_indices)

    singles_a = [[0, 0] for k in range(lenOccA * lenVirtA)]
    counter = 0
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_a[counter] = [p, q]
            counter = counter + 1

    counter = 0
    singles_b = [[0, 0] for k in range(lenOccB * lenVirtB)]
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_b[counter] = [p, q]
            counter = counter + 1

    counter = 0
    doubles_m = [
        [0, 0, 0, 0] for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)
    ]
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_m[counter] = [p, q, r, s]
                    counter = counter + 1

    counter = 0
    nEle = 0
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            for r in range(lenVirtA - 1):
                for s in range(r + 1, lenVirtA):
                    nEle = nEle + 1

    counter = 0
    doubles_a = [[0, 0, 0, 0] for k in range(nEle)]
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            for r in range(lenVirtA - 1):
                for s in range(r + 1, lenVirtA):
                    doubles_a[counter] = [occupied_alpha_indices[p],occupied_alpha_indices[q],\
                                     virtual_alpha_indices[r],virtual_alpha_indices[s]]
                    counter = counter + 1

    counter = 0
    nEle = 0
    for p in range(lenOccB - 1):
        for q in range(p + 1, lenOccB):
            for r in range(lenVirtB - 1):
                for s in range(r + 1, lenVirtB):
                    nEle = nEle + 1
    doubles_b = [[0, 0, 0, 0] for k in range(nEle)]
    for p in range(lenOccB - 1):
        for q in range(p + 1, lenOccB):
            for r in range(lenVirtB - 1):
                for s in range(r + 1, lenVirtB):
                    doubles_b[counter] = [occupied_beta_indices[p],occupied_beta_indices[q],\
                                     virtual_beta_indices[r],virtual_beta_indices[s]]
                    counter = counter + 1

    n_alpha_singles = len(singles_a)
    n_beta_singles = len(singles_b)
    n_mixed_doubles = len(doubles_m)
    n_alpha_doubles = len(doubles_a)
    n_beta_doubles = len(doubles_b)

    thetaCounter = 0
    for i in range(n_alpha_singles):
        single_excitation(qubits, singles_a[i][0], singles_a[i][1],
                          thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_singles):
        single_excitation(qubits, singles_b[i][0], singles_b[i][1],
                          thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_mixed_doubles):
        double_excitation_opt(qubits, doubles_m[i][0], doubles_m[i][1],
                              doubles_m[i][2], doubles_m[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_alpha_doubles):
        double_excitation_opt(qubits, doubles_a[i][0], doubles_a[i][1],
                              doubles_a[i][2], doubles_a[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_doubles):
        double_excitation_opt(qubits, doubles_b[i][0], doubles_b[i][1],
                              doubles_b[i][2], doubles_b[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1


@cudaq.kernel
def uccsd_even_electrons(qubits: cudaq.qview, thetas: List[float],
                         n_electrons: int, n_qubits: int):
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
    virtual_alpha_indices = [i * 2 + n_electrons for i in range(n_virtual)]

    occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
    virtual_beta_indices = [i * 2 + 1 + n_electrons for i in range(n_virtual)]

    lenOccA = len(occupied_alpha_indices)
    lenOccB = len(occupied_beta_indices)
    lenVirtA = len(virtual_alpha_indices)
    lenVirtB = len(virtual_beta_indices)

    singles_a = [[0, 0] for k in range(lenOccA * lenVirtA)]
    counter = 0
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_a[counter] = [p, q]
            counter = counter + 1

    counter = 0
    singles_b = [[0, 0] for k in range(lenOccB * lenVirtB)]
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_b[counter] = [p, q]
            counter = counter + 1

    counter = 0
    doubles_m = [
        [0, 0, 0, 0] for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)
    ]
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_m[counter] = [p, q, r, s]
                    counter = counter + 1

    counter = 0
    nEle = 0
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            for r in range(lenVirtA - 1):
                for s in range(r + 1, lenVirtA):
                    nEle = nEle + 1

    counter = 0
    doubles_a = [[0, 0, 0, 0] for k in range(nEle)]
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            for r in range(lenVirtA - 1):
                for s in range(r + 1, lenVirtA):
                    doubles_a[counter] = [occupied_alpha_indices[p],occupied_alpha_indices[q],\
                                     virtual_alpha_indices[r],virtual_alpha_indices[s]]
                    counter = counter + 1

    counter = 0
    nEle = 0
    for p in range(lenOccB - 1):
        for q in range(p + 1, lenOccB):
            for r in range(lenVirtB - 1):
                for s in range(r + 1, lenVirtB):
                    nEle = nEle + 1
    doubles_b = [[0, 0, 0, 0] for k in range(nEle)]
    for p in range(lenOccB - 1):
        for q in range(p + 1, lenOccB):
            for r in range(lenVirtB - 1):
                for s in range(r + 1, lenVirtB):
                    doubles_b[counter] = [occupied_beta_indices[p],occupied_beta_indices[q],\
                                     virtual_beta_indices[r],virtual_beta_indices[s]]
                    counter = counter + 1

    n_alpha_singles = len(singles_a)
    n_beta_singles = len(singles_b)
    n_mixed_doubles = len(doubles_m)
    n_alpha_doubles = len(doubles_a)
    n_beta_doubles = len(doubles_b)

    thetaCounter = 0
    for i in range(n_alpha_singles):
        single_excitation(qubits, singles_a[i][0], singles_a[i][1],
                          thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_singles):
        single_excitation(qubits, singles_b[i][0], singles_b[i][1],
                          thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_mixed_doubles):
        double_excitation_opt(qubits, doubles_m[i][0], doubles_m[i][1],
                              doubles_m[i][2], doubles_m[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_alpha_doubles):
        double_excitation_opt(qubits, doubles_a[i][0], doubles_a[i][1],
                              doubles_a[i][2], doubles_a[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_doubles):
        double_excitation_opt(qubits, doubles_b[i][0], doubles_b[i][1],
                              doubles_b[i][2], doubles_b[i][3],
                              thetas[thetaCounter])
        thetaCounter += 1


@cudaq.kernel
def uccsd(qubits: cudaq.qview, thetas: List[float], n_electrons: int,
          n_qubits: int):
    """
    Generate the unitary coupled cluster singlet doublet CUDA-Q kernel.

    Args:
        qubits (:class:`qview`): Pre-allocated qubits
        thetas (List[float]): List of parameters
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits
    """

    if n_electrons % 2 == 0:
        uccsd_even_electrons(qubits, thetas, n_electrons, n_qubits)
    else:
        uccsd_odd_electrons(qubits, thetas, n_electrons, n_qubits)
