# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import scipy as sp


def zero_matrix(size):
    return np.zeros((size, size), dtype=np.complex128)


def identity_matrix(size):
    return np.eye(size, dtype=np.complex128)


def paulix_matrix():
    mat = np.zeros((2, 2), dtype=np.complex128)
    mat[0, 1] = 1
    mat[1, 0] = 1
    return mat


def pauliy_matrix():
    mat = np.zeros((2, 2), dtype=np.complex128)
    mat[0, 1] = -1j
    mat[1, 0] = 1j
    return mat


def pauliz_matrix():
    return parity_matrix(2)


def sigmap_matrix():
    return annihilate_matrix(2)


def sigmam_matrix():
    return create_matrix(2)


def create_matrix(size):
    diag = np.sqrt(np.arange(1, size, dtype=np.complex128))
    return np.diag(diag, -1)


def annihilate_matrix(size):
    diag = np.sqrt(np.arange(1, size, dtype=np.complex128))
    return np.diag(diag, 1)


def number_matrix(size):
    return np.diag(np.arange(size, dtype=np.complex128))


def parity_matrix(size):
    diag = np.ones(size, dtype=np.complex128)
    diag[1::2] = -1
    return np.diag(diag)


def position_matrix(size):
    return 0.5 * (create_matrix(size) + annihilate_matrix(size))


def momentum_matrix(size):
    return 0.5j * (create_matrix(size) - annihilate_matrix(size))


def squeeze_matrix(size, ampl):
    term1 = np.conjugate(ampl) * np.linalg.matrix_power(annihilate_matrix(size),
                                                        2)
    term2 = ampl * np.linalg.matrix_power(create_matrix(size), 2)
    return sp.linalg.expm(0.5 * (term1 - term2))


def displace_matrix(size, ampl):
    term1 = ampl * create_matrix(size)
    term2 = np.conjugate(ampl) * annihilate_matrix(size)
    return sp.linalg.expm(term1 - term2)
