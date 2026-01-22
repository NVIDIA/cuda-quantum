# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


# Use a snapshot of the uccsd.py to make sure we can compile
# complex code. Importing uccsd from cudaq.kernels fails due
# clearing the caches in the tests.
# Issue: https://github.com/NVIDIA/cuda-quantum/issues/1954
def test_cudaq_uccsd1():

    @cudaq.kernel
    def single_excitation1(qubits: cudaq.qview, p_occ: int, q_virt: int,
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
    def double_excitation_opt1(qubits: cudaq.qview, p_occ: int, q_occ: int,
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
    def uccsd1_odd_electrons(qubits: cudaq.qview, thetas: list[float],
                             n_electrons: int, n_qubits: int):
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
        for i in range(n_virtual):
            virtual_beta_indices[i + 1] = i * 2 + 1 + n_electrons

        lenOccA = len(occupied_alpha_indices)
        lenOccB = len(occupied_beta_indices)
        lenVirtA = len(virtual_alpha_indices)
        lenVirtB = len(virtual_beta_indices)

        singles_a0 = [0 for k in range(lenOccA * lenVirtA)]
        singles_a1 = [0 for k in range(lenOccA * lenVirtA)]
        counter = 0
        for p in occupied_alpha_indices:
            for q in virtual_alpha_indices:
                singles_a0[counter] = p
                singles_a1[counter] = q
                counter = counter + 1

        counter = 0
        singles_b0 = [0 for k in range(lenOccB * lenVirtB)]
        singles_b1 = [0 for k in range(lenOccB * lenVirtB)]
        for p in occupied_beta_indices:
            for q in virtual_beta_indices:
                singles_b0[counter] = p
                singles_b1[counter] = q
                counter = counter + 1

        counter = 0
        doubles_m0 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m1 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m2 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m3 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        for p in occupied_alpha_indices:
            for q in occupied_beta_indices:
                for r in virtual_beta_indices:
                    for s in virtual_alpha_indices:
                        doubles_m0[counter] = p
                        doubles_m1[counter] = q
                        doubles_m2[counter] = r
                        doubles_m3[counter] = s
                        counter = counter + 1

        counter = 0
        nEle = 0
        for p in range(lenOccA - 1):
            for q in range(p + 1, lenOccA):
                for r in range(lenVirtA - 1):
                    for s in range(r + 1, lenVirtA):
                        nEle = nEle + 1

        counter = 0
        doubles_a0 = [0 for k in range(nEle)]
        doubles_a1 = [0 for k in range(nEle)]
        doubles_a2 = [0 for k in range(nEle)]
        doubles_a3 = [0 for k in range(nEle)]
        for p in range(lenOccA - 1):
            for q in range(p + 1, lenOccA):
                for r in range(lenVirtA - 1):
                    for s in range(r + 1, lenVirtA):
                        doubles_a0[counter] = occupied_alpha_indices[p]
                        doubles_a1[counter] = occupied_alpha_indices[q]
                        doubles_a2[counter] = virtual_alpha_indices[r]
                        doubles_a3[counter] = virtual_alpha_indices[s]
                        counter = counter + 1

        counter = 0
        nEle = 0
        for p in range(lenOccB - 1):
            for q in range(p + 1, lenOccB):
                for r in range(lenVirtB - 1):
                    for s in range(r + 1, lenVirtB):
                        nEle = nEle + 1

        doubles_b0 = [0 for k in range(nEle)]
        doubles_b1 = [0 for k in range(nEle)]
        doubles_b2 = [0 for k in range(nEle)]
        doubles_b3 = [0 for k in range(nEle)]
        for p in range(lenOccB - 1):
            for q in range(p + 1, lenOccB):
                for r in range(lenVirtB - 1):
                    for s in range(r + 1, lenVirtB):
                        doubles_b0[counter] = occupied_beta_indices[p]
                        doubles_b1[counter] = occupied_beta_indices[q]
                        doubles_b2[counter] = virtual_beta_indices[r]
                        doubles_b3[counter] = virtual_beta_indices[s]
                        counter = counter + 1

        n_alpha_singles = len(singles_a0)
        n_beta_singles = len(singles_b0)
        n_mixed_doubles = len(doubles_m0)
        n_alpha_doubles = len(doubles_a0)
        n_beta_doubles = len(doubles_b0)

        thetaCounter = 0
        for i in range(n_alpha_singles):
            single_excitation1(qubits, singles_a0[i], singles_a1[i],
                               thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_beta_singles):
            single_excitation1(qubits, singles_b0[i], singles_b1[i],
                               thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_mixed_doubles):
            double_excitation_opt1(qubits, doubles_m0[i], doubles_m1[i],
                                   doubles_m2[i], doubles_m3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_alpha_doubles):
            double_excitation_opt1(qubits, doubles_a0[i], doubles_a1[i],
                                   doubles_a2[i], doubles_a3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_beta_doubles):
            double_excitation_opt1(qubits, doubles_b0[i], doubles_b1[i],
                                   doubles_b2[i], doubles_b3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

    @cudaq.kernel
    def uccsd1_even_electrons(qubits: cudaq.qview, thetas: list[float],
                              n_electrons: int, n_qubits: int):
        n_spatial_orbitals = n_qubits // 2
        n_occupied = int(np.ceil(n_electrons / 2))
        n_virtual = n_spatial_orbitals - n_occupied

        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [i * 2 + n_electrons for i in range(n_virtual)]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
        virtual_beta_indices = [
            i * 2 + 1 + n_electrons for i in range(n_virtual)
        ]

        lenOccA = len(occupied_alpha_indices)
        lenOccB = len(occupied_beta_indices)
        lenVirtA = len(virtual_alpha_indices)
        lenVirtB = len(virtual_beta_indices)

        singles_a0 = [0 for k in range(lenOccA * lenVirtA)]
        singles_a1 = [0 for k in range(lenOccA * lenVirtA)]
        counter = 0
        for p in occupied_alpha_indices:
            for q in virtual_alpha_indices:
                singles_a0[counter] = p
                singles_a1[counter] = q
                counter = counter + 1

        counter = 0
        singles_b0 = [0 for k in range(lenOccB * lenVirtB)]
        singles_b1 = [0 for k in range(lenOccB * lenVirtB)]
        for p in occupied_beta_indices:
            for q in virtual_beta_indices:
                singles_b0[counter] = p
                singles_b1[counter] = q
                counter = counter + 1

        counter = 0
        doubles_m0 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m1 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m2 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        doubles_m3 = [0 for k in range(lenOccB * lenVirtB * lenOccA * lenVirtA)]
        for p in occupied_alpha_indices:
            for q in occupied_beta_indices:
                for r in virtual_beta_indices:
                    for s in virtual_alpha_indices:
                        doubles_m0[counter] = p
                        doubles_m1[counter] = q
                        doubles_m2[counter] = r
                        doubles_m3[counter] = s
                        counter = counter + 1

        counter = 0
        nEle = 0
        for p in range(lenOccA - 1):
            for q in range(p + 1, lenOccA):
                for r in range(lenVirtA - 1):
                    for s in range(r + 1, lenVirtA):
                        nEle = nEle + 1

        counter = 0
        doubles_a0 = [0 for k in range(nEle)]
        doubles_a1 = [0 for k in range(nEle)]
        doubles_a2 = [0 for k in range(nEle)]
        doubles_a3 = [0 for k in range(nEle)]
        for p in range(lenOccA - 1):
            for q in range(p + 1, lenOccA):
                for r in range(lenVirtA - 1):
                    for s in range(r + 1, lenVirtA):
                        doubles_a0[counter] = occupied_alpha_indices[p]
                        doubles_a1[counter] = occupied_alpha_indices[q]
                        doubles_a2[counter] = virtual_alpha_indices[r]
                        doubles_a3[counter] = virtual_alpha_indices[s]
                        counter = counter + 1

        counter = 0
        nEle = 0
        for p in range(lenOccB - 1):
            for q in range(p + 1, lenOccB):
                for r in range(lenVirtB - 1):
                    for s in range(r + 1, lenVirtB):
                        nEle = nEle + 1

        doubles_b0 = [0 for k in range(nEle)]
        doubles_b1 = [0 for k in range(nEle)]
        doubles_b2 = [0 for k in range(nEle)]
        doubles_b3 = [0 for k in range(nEle)]
        for p in range(lenOccB - 1):
            for q in range(p + 1, lenOccB):
                for r in range(lenVirtB - 1):
                    for s in range(r + 1, lenVirtB):
                        doubles_b0[counter] = occupied_beta_indices[p]
                        doubles_b1[counter] = occupied_beta_indices[q]
                        doubles_b2[counter] = virtual_beta_indices[r]
                        doubles_b3[counter] = virtual_beta_indices[s]
                        counter = counter + 1

        n_alpha_singles = len(singles_a0)
        n_beta_singles = len(singles_b0)
        n_mixed_doubles = len(doubles_m0)
        n_alpha_doubles = len(doubles_a0)
        n_beta_doubles = len(doubles_b0)

        thetaCounter = 0
        for i in range(n_alpha_singles):
            single_excitation1(qubits, singles_a0[i], singles_a1[i],
                               thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_beta_singles):
            single_excitation1(qubits, singles_b0[i], singles_b1[i],
                               thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_mixed_doubles):
            double_excitation_opt1(qubits, doubles_m0[i], doubles_m1[i],
                                   doubles_m2[i], doubles_m3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_alpha_doubles):
            double_excitation_opt1(qubits, doubles_a0[i], doubles_a1[i],
                                   doubles_a2[i], doubles_a3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

        for i in range(n_beta_doubles):
            double_excitation_opt1(qubits, doubles_b0[i], doubles_b1[i],
                                   doubles_b2[i], doubles_b3[i],
                                   thetas[thetaCounter])
            thetaCounter += 1

    @cudaq.kernel
    def uccsd1(qubits: cudaq.qview, thetas: list[float], n_electrons: int,
               n_qubits: int):
        """
        Generate the unitary coupled cluster singlet doublet CUDA-Q kernel.

        Args:
            qubits (:class:`qview`): Pre-allocated qubits
            thetas (list[float]): List of parameters
            n_electrons (int): Number of electrons
            n_qubits (int): Number of qubits
        """

        if n_electrons % 2 == 0:
            uccsd1_even_electrons(qubits, thetas, n_electrons, n_qubits)
        else:
            uccsd1_odd_electrons(qubits, thetas, n_electrons, n_qubits)

    num_electrons = 2
    num_qubits = 8

    thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(num_qubits)
        for i in range(num_electrons):
            x(qubits[i])
        uccsd1(qubits, thetas, num_electrons, num_qubits)

    counts = cudaq.sample(kernel, shots_count=1000)

    assert len(counts) == 6
    assert '00000011' in counts
    assert '00000110' in counts
    assert '00010010' in counts
    assert '01000010' in counts
    assert '10000001' in counts
    assert '11000000' in counts
