# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np
import os
import pytest


@cudaq.kernel
def bar(qvec: cudaq.qview):
    rx(np.e, qvec[0])
    ry(np.pi, qvec[1])
    cudaq.adjoint(rz, np.pi, qvec[2])


@cudaq.kernel
def zaz(qub: cudaq.qubit):
    sdg(qub)


@cudaq.kernel
def kernel():
    q = cudaq.qvector(4)
    h(q)
    x.ctrl(q[0], q[1])
    y.ctrl(q[0], q[1], q[2])
    y.ctrl(q[2], q[0], q[1])
    y.ctrl(q[1], q[2], q[0])
    z(q[2])
    r1(3.14159, q[0])
    tdg(q[1])
    s(q[2])
    swap.ctrl(q[0], q[2])
    swap.ctrl(q[1], q[2])
    swap.ctrl(q[0], q[1])
    swap.ctrl(q[0], q[2])
    swap.ctrl(q[1], q[2])
    swap.ctrl(q[3], q[0], q[1])
    swap.ctrl(q[0], q[3], q[1], q[2])
    swap.ctrl(q[1], q[0], q[3])
    swap.ctrl(q[1], q[2], q[0], q[3])
    bar(q)
    cudaq.control(zaz, q[1], q[0])
    cudaq.adjoint(bar, q)


def test_draw():
    """Test draw function, mainly copied from draw_tester.cpp"""
    # fmt: off
    expected_str = R"""
     ╭───╮               ╭───╮╭───────────╮                          ╭───────╮»
q0 : ┤ h ├──●────●────●──┤ y ├┤ r1(3.142) ├──────╳─────╳──╳─────╳──●─┤>      ├»
     ├───┤╭─┴─╮  │  ╭─┴─╮╰─┬─╯╰──┬─────┬──╯      │     │  │     │  │ │       │»
q1 : ┤ h ├┤ x ├──●──┤ y ├──●─────┤ tdg ├─────────┼──╳──╳──┼──╳──╳──╳─┤●      ├»
     ├───┤╰───╯╭─┴─╮╰─┬─╯  │     ╰┬───┬╯   ╭───╮ │  │     │  │  │  │ │  swap │»
q2 : ┤ h ├─────┤ y ├──●────●──────┤ z ├────┤ s ├─╳──╳─────╳──╳──┼──╳─│       │»
     ├───┤     ╰───╯              ╰───╯    ╰───╯                │  │ │       │»
q3 : ┤ h ├──────────────────────────────────────────────────────●──●─┤>      ├»
     ╰───╯                                                           ╰───────╯»

################################################################################

╭───────╮╭───────────╮    ╭─────╮   ╭────────────╮
┤>      ├┤ rx(2.718) ├────┤ sdg ├───┤ rx(-2.718) ├
│       │├───────────┤    ╰──┬──╯   ├────────────┤
┤●      ├┤ ry(3.142) ├───────●──────┤ ry(-3.142) ├
│  swap │├───────────┴╮╭───────────╮╰────────────╯
┤●      ├┤ rz(-3.142) ├┤ rz(3.142) ├──────────────
│       │╰────────────╯╰───────────╯              
┤>      ├─────────────────────────────────────────
╰───────╯                                         
"""
    # fmt: on
    expected_str = expected_str[1:]
    produced_string = cudaq.draw(kernel)
    print()
    print(produced_string)
    assert expected_str == produced_string


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
