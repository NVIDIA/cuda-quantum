# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq

## NOTE: https://docs.python.org/3.9/whatsnew/3.9.html#type-hinting-generics-in-standard-collections
## To support Python v3.8, using `typing.List[float]` instead of `list[float]`
from typing import List


@cudaq.kernel
def genExcitations(occ_a: list[int], virt_a: list[int], occ_b: list[int],
                   virt_b: list[int], singles_a: list[list[int]],
                   singles_b: list[list[int]], doubles_m: list[list[int]],
                   doubles_a: list[list[int]], doubles_b: list[list[int]]):

    counter = 0
    for p in occ_a:
        for q in virt_a:
            singles_a[counter] = [p, q]
            counter = counter + 1

    counter = 0

    # occupied orbital is beta and virtual orbital is beta
    for p in occ_b:
        for q in virt_b:
            singles_b[counter] = [p, q]
            counter = counter + 1

    counter = 0
    #Mixed spin double excitation
    for p in occ_a:
        for q in occ_b:
            for r in virt_b:
                for s in virt_a:
                    doubles_m[counter] = [p, q, r, s]
                    counter = counter + 1

    # same spin double excitation
    n_occ_alpha = len(occ_a)
    n_occ_beta = len(occ_b)
    n_virt_alpha = len(virt_a)
    n_virt_beta = len(virt_b)
    counter = 0

    for p in range(n_occ_alpha - 1):
        for q in range(p + 1, n_occ_alpha):
            for r in range(n_virt_alpha - 1):
                for s in range(r + 1, n_virt_alpha):

                    # Same spin: all alpha
                    doubles_a[counter] = [
                        occ_a[p], occ_a[q], virt_a[r], virt_a[s]
                    ]
                    counter = counter + 1

    counter = 0
    for p in range(n_occ_beta - 1):
        for q in range(p + 1, n_occ_beta):
            for r in range(n_virt_beta - 1):
                for s in range(r + 1, n_virt_beta):

                    # Same spin: all beta
                    doubles_b[counter] = [
                        occ_b[p], occ_b[q], virt_b[r], virt_b[s]
                    ]
                    counter = counter + 1

    return


@cudaq.kernel
def test_excitations(n_electrons: int, n_qubits: int, m: int, n: int,
                      choice: int) -> int:
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    if n_electrons % 2 != 0:
        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [
            i * 2 + n_electrons + 1 for i in range(n_virtual)
        ]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied - 1)]
        virtual_beta_indices = [0 for k in range(n_virtual + 1)]
        virtual_beta_indices[0] = 2 * n_occupied - 1
        for i in range(1, n_virtual + 1):
            virtual_beta_indices[i] = i * 2 + 1 + n_electrons + 1

        singles_a = [[0, 0] for k in range(n_occupied * n_virtual)]
        singles_b = [[0, 0] for k in range(
            len(occupied_beta_indices) * len(virtual_beta_indices))]
        doubles_m = [[0, 0, 0, 0] for k in range(
            len(occupied_beta_indices) * len(virtual_beta_indices) *
            len(occupied_alpha_indices) * len(virtual_alpha_indices))]

        nEle = 0
        for p in range(n_occupied - 1):
            for q in range(p + 1, n_occupied):
                for r in range(n_virtual - 1):
                    for s in range(r + 1, n_virtual):
                        nEle = nEle + 1

        doubles_a = [[0, 0, 0, 0] for k in range(nEle)]

        nEle = 0
        for p in range(n_occupied - 2):
            for q in range(p + 1, n_occupied - 2):
                for r in range(n_virtual):
                    for s in range(r + 1, n_virtual + 1):
                        nEle = nEle + 1

        doubles_b = [[0, 0, 0, 0] for k in range(nEle)]
        genExcitations(occupied_alpha_indices, virtual_alpha_indices,
                       occupied_beta_indices, virtual_beta_indices, singles_a,
                       singles_b, doubles_m, doubles_a, doubles_b)

        if choice == 0:
            return singles_a[m][n]
        elif choice == 1:
            return singles_b[m][n]
        elif choice == 2:
            return doubles_m[m][n]
        elif choice == 3:
            return doubles_a[m][n]
        elif choice == 4:
            return doubles_b[m][n]

        return -1

    occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
    virtual_alpha_indices = [i * 2 + n_electrons for i in range(n_virtual)]

    occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
    virtual_beta_indices = [i * 2 + 1 + n_electrons for i in range(n_virtual)]

    singles_a = [[0, 0] for k in range(n_occupied * n_virtual)]
    singles_b = [[
        0, 0
    ] for k in range(len(occupied_beta_indices) * len(virtual_beta_indices))]
    doubles_m = [[0, 0, 0, 0] for k in range(
        len(occupied_beta_indices) * len(virtual_beta_indices) *
        len(occupied_alpha_indices) * len(virtual_alpha_indices))]

    nEle = 0
    for p in range(n_occupied - 1):
        for q in range(p + 1, n_occupied):
            for r in range(n_virtual - 1):
                for s in range(r + 1, n_virtual):
                    nEle = nEle + 1

    doubles_a = [[0, 0, 0, 0] for k in range(nEle)]

    n_occ_alpha = len(occupied_alpha_indices)
    n_occ_beta = len(occupied_beta_indices)
    n_virt_alpha = len(virtual_alpha_indices)
    n_virt_beta = len(virtual_beta_indices)

    nEle = 0
    for p in range(n_occ_alpha - 1):
        for q in range(p + 1, n_occ_alpha):
            for r in range(n_virt_alpha - 1):
                for s in range(r + 1, n_virt_alpha):
                    nEle = nEle + 1

    doubles_b = [[0, 0, 0, 0] for k in range(nEle)]
    genExcitations(occupied_alpha_indices, virtual_alpha_indices,
                   occupied_beta_indices, virtual_beta_indices, singles_a,
                   singles_b, doubles_m, doubles_a, doubles_b)
    if choice == 0:
        a = singles_a[m][n]
        return a
    elif choice == 1:
        return singles_b[m][n]
    elif choice == 2:
        return doubles_m[m][n]
    elif choice == 3:
        return doubles_a[m][n]
    elif choice == 4:
        return doubles_b[m][n]
    return -1


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


def single_excitation_gate(kernel, qubits: cudaq.qview, p_occ: int, q_virt: int,
                           theta: float):

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


def double_excitation_gate_opt(kernel, qubits: cudaq.qview, p_occ: int,
                               q_occ: int, r_virt: int, s_virt: int,
                               theta: float):

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
        theta = theta * -1.

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ = q_occ 
        j_occ = p_occ
        a_virt = r_virt
        b_virt = s_virt
        theta = theta * -1.0

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
        theta = theta * -1.

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ = q_occ 
        j_occ = p_occ
        a_virt = r_virt
        b_virt = s_virt
        theta = theta * -1.0
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
def uccsd_odd_electrons(qubits: cudaq.qview, thetas: List[float], n_electrons: int,
          n_qubits: int):
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
    virtual_alpha_indices = [
        i * 2 + n_electrons + 1 for i in range(n_virtual)
    ]

    occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied - 1)]
    virtual_beta_indices = [0 for k in range(n_virtual + 1)]
    virtual_beta_indices[0] = 2 * n_occupied - 1
    for i in range(1, n_virtual + 1):
        virtual_beta_indices[i] = i * 2 + 1 + n_electrons + 1

    singles_a = [[0, 0] for k in range(n_occupied * n_virtual)]
    singles_b = [[0, 0] for k in range(
        len(occupied_beta_indices) * len(virtual_beta_indices))]
    doubles_m = [[0, 0, 0, 0] for k in range(
        len(occupied_beta_indices) * len(virtual_beta_indices) *
        len(occupied_alpha_indices) * len(virtual_alpha_indices))]

    nEle = 0
    for p in range(n_occupied - 1):
        for q in range(p + 1, n_occupied):
            for r in range(n_virtual - 1):
                for s in range(r + 1, n_virtual):
                    nEle = nEle + 1

    doubles_a = [[0, 0, 0, 0] for k in range(nEle)]

    nEle = 0
    for p in range(n_occupied - 2):
        for q in range(p + 1, n_occupied - 2):
            for r in range(n_virtual):
                for s in range(r + 1, n_virtual + 1):
                    nEle = nEle + 1

    doubles_b = [[0, 0, 0, 0] for k in range(nEle)]
    genExcitations(occupied_alpha_indices, virtual_alpha_indices,
                    occupied_beta_indices, virtual_beta_indices, singles_a,
                    singles_b, doubles_m, doubles_a, doubles_b)
    

@cudaq.kernel 
def uccsd_even_electrons(qubits: cudaq.qview, thetas: List[float], n_electrons: int,
          n_qubits: int):
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

    singles_a = [[0, 0] for k in range(lenOccA*lenVirtA)]
    counter = 0
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_a[counter] = [p,q]
            counter = counter + 1
    
    counter = 0
    singles_b = [[0, 0] for k in range(lenOccB * lenVirtB)]
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_b[counter] = [p,q]
            counter = counter + 1
    
    counter = 0
    doubles_m = [[0, 0, 0, 0] for k in range(lenOccB * lenVirtB * lenOccA *lenVirtA)]
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_m[counter] = [p,q,r,s]
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
        single_excitation(qubits, singles_a[i][0],
                               singles_a[i][1], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_singles):
        single_excitation(qubits, singles_b[i][0],
                               singles_b[i][1], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_mixed_doubles):
        double_excitation_opt(qubits, doubles_m[i][0],
                                   doubles_m[i][1], doubles_m[i][2],
                                   doubles_m[i][3], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_alpha_doubles):
        double_excitation_opt(qubits, doubles_a[i][0],
                                   doubles_a[i][1], doubles_a[i][2],
                                   doubles_a[i][3], thetas[thetaCounter])
        thetaCounter += 1

    for i in range(n_beta_doubles):
        double_excitation_opt(qubits, doubles_b[i][0],
                                   doubles_b[i][1], doubles_b[i][2],
                                   doubles_b[i][3], thetas[thetaCounter])
        thetaCounter += 1


@cudaq.kernel
def __mlir__cudaq__uccsd(qubits: cudaq.qview, thetas: list[float], n_electrons: int,
          n_qubits: int):
    
    if n_electrons % 2 == 0:
        uccsd_even_electrons(qubits, thetas, n_electrons, n_qubits)
    else:
        uccsd_odd_electrons(qubits, thetas, n_electrons, n_qubits)



def __builder__cudaq__uccsd(kernel: cudaq.PyKernel, qubits: cudaq.qview, thetas: list[float], n_electrons: int,
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

def uccsd(*args):
    if isinstance(args[0], cudaq.PyKernel):
        __builder__cudaq__uccsd(*args)
        return 
    raise RuntimeError("")