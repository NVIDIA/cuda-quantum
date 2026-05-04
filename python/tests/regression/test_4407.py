# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    yield
    cudaq.__clearKernelRegistries()


# Qubit scoping test from
# https://github.com/NVIDIA/cuda-quantum/issues/4407
def test_issue_4407():

    @cudaq.kernel
    def test_kernel() -> int:
        q = cudaq.qvector(5)
        h(q[0])
        return mz(q[0])

    @cudaq.kernel
    def main_kernel() -> int:

        a = test_kernel()
        b = test_kernel()
        c = test_kernel()
        return a + b + c

    cudaq.run(main_kernel, shots_count=10)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
