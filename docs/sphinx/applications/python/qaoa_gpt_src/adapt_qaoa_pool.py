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

import cudaq
from cudaq import spin


def qaoa_mixer(n):

    term = spin.x(0)

    for i in range(1, n):
        term += spin.x(i)

    pool = [term]
    return pool


def qaoa_single_x(n):

    pool = []

    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))

    return pool


def qaoa_double(n):

    pool = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            pool.append(
                cudaq.SpinOperator(spin.x(i)) * cudaq.SpinOperator(spin.x(j)))
            pool.append(
                cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.y(j)))
            pool.append(
                cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.z(j)))
            pool.append(
                cudaq.SpinOperator(spin.z(i)) * cudaq.SpinOperator(spin.y(j)))

    return pool


def all_pool(qubits_num):
    return (qaoa_single_x(qubits_num) + qaoa_mixer(qubits_num) +
            qaoa_double(qubits_num))
