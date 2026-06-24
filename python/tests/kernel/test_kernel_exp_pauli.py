# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_param():

    @cudaq.kernel
    def test_param(w: cudaq.pauli_word):
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, w)

    counts = cudaq.sample(test_param, cudaq.pauli_word("XX"))
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts
