# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.mark.parametrize("target", ['qpp-cpu', 'quantinuum'])
def test_exp_pauli(target: str):

    if target == 'quantinuum':
        cudaq.set_target(target, emulate=True)
    else:
        cudaq.set_target(target)

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts

    cudaq.reset_target()


@pytest.mark.parametrize("target", ['qpp-cpu', 'quantinuum'])
def test_exp_pauli_param(target: str):

    if target == 'quantinuum':
        cudaq.set_target(target, emulate=True)
    else:
        cudaq.set_target(target)

    @cudaq.kernel
    def test_param(w: cudaq.pauli_word):
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, w)

    counts = cudaq.sample(test_param, cudaq.pauli_word("XX"))
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts

    cudaq.reset_target()
