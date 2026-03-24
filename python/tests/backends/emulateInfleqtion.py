# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
from cudaq import spin
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def do_something():
    cudaq.set_target("infleqtion", emulate=True)
    yield "Running the tests."
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_deep_exp_pauli():

    @cudaq.kernel
    def example(n: int, words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n)
        for qi in q:
            h(qi)
        for word in words:
            exp_pauli(0.5, q, word)
        mz(q)

    words = ["XXI", "YYI", "IXX", "IYY"]
    counts = cudaq.sample(example, 3, words)

    counts.dump()
    assert '000' in counts
    assert '001' in counts
    assert '010' in counts
    assert '011' in counts
    assert '100' in counts
    assert '101' in counts
    assert '110' in counts
    assert '111' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
