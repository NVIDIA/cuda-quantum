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

    d = counts.to_dict()
    assert d["rx"] == 1
    assert d["ry"] == 1
    assert d["rz"] == 1
    assert d["h"] == 1
    assert d["x"] == 1
    assert d["y"] == 1
    assert d["z"] == 1
    assert d["s"] == 1
    assert d["t"] == 1


def test_control_gates_resources():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qvector(3)
        x(q[0])
        x.ctrl(q[0], q[1])
        x.ctrl([q[0], q[1]], q[2])
        h(q[0])
        h.ctrl(q[0], q[1])

    counts = cudaq.estimate_resources(mykernel)

    assert counts.count_controls("x", 0) == 1
    assert counts.count_controls("x", 1) == 1
    assert counts.count_controls("x", 2) == 1

    assert counts.count_controls("h", 0) == 1
    assert counts.count_controls("h", 1) == 1

    assert counts.count() == 5

    d = counts.to_dict()
    assert isinstance(d, dict)

    assert d["x"] == 1
    assert d["h"] == 1

    assert d["cx"] == 1
    assert d["ccx"] == 1
    assert d["ch"] == 1


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

    d1 = counts1.to_dict()
    d2 = counts2.to_dict()
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert d1["h"] == 1
    assert d2["h"] == 1
    assert d2["x"] == 1

    cudaq.set_target("quantinuum", emulate=True)
    counts1 = cudaq.estimate_resources(mykernel)
    counts2 = cudaq.estimate_resources(mykernel, choice=choice)

    assert counts1.count("h") == 1
    assert counts2.count("h") == 1
    assert counts2.count("x") == 1

    d1 = counts1.to_dict()
    d2 = counts2.to_dict()
    assert d1["h"] == 1
    assert d2["h"] == 1
    assert d2["x"] == 1


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


def test_loop_with_args():

    @cudaq.kernel
    def callee(q: cudaq.qview, other_args: list[float]):
        h(q[0])
        for i, arg in enumerate(other_args):
            rx(arg, q[i])

    @cudaq.kernel
    def caller(n: int, angles: list[float]):
        q = cudaq.qvector(n)
        callee(q, angles)
        mz(q)

    counts = cudaq.estimate_resources(caller, 3, [1.0, 2.0, 3.0])
    assert counts.count("rx") == 3
    assert counts.count("h") == 1

    d = counts.to_dict()
    assert isinstance(d, dict)
    assert d["rx"] == 3
    assert d["h"] == 1

    cudaq.set_target("qci", emulate=True)
    counts = cudaq.estimate_resources(caller, 3, [4.0, 5.0, 6.0])
    assert counts.count("rx") == 3
    assert counts.count("h") == 1

    d = counts.to_dict()
    assert isinstance(d, dict)
    assert d["rx"] == 3
    assert d["h"] == 1


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
