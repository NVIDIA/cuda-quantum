# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Compile and run with:

import cudaq

def test_resource_counter():
    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        p = cudaq.qubit()

        # Alias them
        h(q)

        m1 = mz(q)
        if m1:
            x(p)
            m2 = mz(p)
        else:
            m3 = mz(p)

    def choice():
        return True

    cudaq.set_target("quantinuum", emulate=True)

    counts1 = cudaq.sample(mykernel, shots_count=5)
    counts2 = cudaq.count_resources(choice, mykernel)
    counts3 = cudaq.sample(mykernel, shots_count=10)

    assert counts1.count("00") + counts1.count("11") == 5
    assert counts2.count("h") == 1
    assert counts2.count("x") == 1
    assert counts3.count("00") + counts3.count("11") == 10

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
