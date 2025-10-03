# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.fixture(scope="session", autouse=True)
def clean():
    cudaq.reset_target()

    yield "Running the tests."

    cudaq.reset_target()


def test_basic():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()

        rx(.0, q)
        ry(.0, q)
        rz(.0, q)
        h(q)
        x(q)
        y(q)
        z(q)
        s(q)
        t(q)

    counts = cudaq.estimate_resources(mykernel)
    assert counts.count("rx") == 1
    assert counts.count("ry") == 1
    assert counts.count("rz") == 1
    assert counts.count("h") == 1
    assert counts.count("x") == 1
    assert counts.count("y") == 1
    assert counts.count("z") == 1
    assert counts.count("s") == 1
    assert counts.count("t") == 1


def test_choice_function():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        p = cudaq.qubit()

        h(q)

        m1 = mz(q)
        if m1:
            x(p)
            m2 = mz(p)
        else:
            m3 = mz(p)

    def choice():
        return True

    counts1 = cudaq.estimate_resources(mykernel)
    counts2 = cudaq.estimate_resources(mykernel, choice=choice)

    assert counts1.count("h") == 1
    assert counts2.count("h") == 1
    assert counts2.count("x") == 1

    cudaq.set_target("quantinuum", emulate=True)
    counts1 = cudaq.estimate_resources(mykernel)
    counts2 = cudaq.estimate_resources(mykernel, choice=choice)

    assert counts1.count("h") == 1
    assert counts2.count("h") == 1
    assert counts2.count("x") == 1


def test_choice_function():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        p = cudaq.qubit()

        h(q)

        m1 = mz(q)
        if m1:
            x(p)
            m2 = mz(p)
        else:
            m3 = mz(p)

    counts1 = cudaq.sample(mykernel, shots_count=5)
    counts2 = cudaq.estimate_resources(mykernel)
    counts3 = cudaq.sample(mykernel, shots_count=10)

    assert counts1.count("00") + counts1.count("11") == 5
    assert counts2.count("h") == 1
    assert counts3.count("00") + counts3.count("11") == 10

    cudaq.set_target("quantinuum", emulate=True)
    counts1 = cudaq.sample(mykernel, shots_count=5)
    counts2 = cudaq.estimate_resources(mykernel)
    counts3 = cudaq.sample(mykernel, shots_count=10)

    assert counts1.count("00") + counts1.count("11") == 5
    assert counts2.count("h") == 1
    assert counts3.count("00") + counts3.count("11") == 10


def test_sample_in_choice():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        p = cudaq.qubit()

        h(q)

        m1 = mz(q)
        if m1:
            x(p)
            m2 = mz(p)
        else:
            m3 = mz(p)

    def choice():
        cudaq.sample(mykernel, shots_count=10)
        return True

    with pytest.raises(RuntimeError):
        cudaq.estimate_resources(mykernel, choice)

    with pytest.raises(RuntimeError):
        cudaq.set_target("quantinuum", emulate=True)
        cudaq.estimate_resources(mykernel, choice)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
