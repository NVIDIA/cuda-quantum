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
    # FIXME https://github.com/NVIDIA/cuda-quantum/issues/1734
    # rx(np.e, qvec[0])
    rx(2.71828182845904523536028, qvec[0])
    ry(np.pi, qvec[1])
    # FIXME https://github.com/NVIDIA/cuda-quantum/issues/1734
    # cudaq.adjoint(rz, np.pi, qvec[2])
    rz(-np.pi, qvec[2])


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
    swap(q[0], q[2])
    swap(q[1], q[2])
    swap(q[0], q[1])
    swap(q[0], q[2])
    swap(q[1], q[2])
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
    assert expected_str == produced_string


def test_draw_latex():
    """Test draw function, mainly copied from draw_tester.cpp"""
    # fmt: off
    expected_str = R"""
\documentclass{minimal}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
  \lstick{$q_0$} & \gate{H} & \ctrl{1} & \ctrl{2} & \ctrl{1} & \gate{Y} & \gate{R_1(3.142)} & \qw & \swap{2} & \qw & \swap{1} & \swap{2} & \qw & \swap{1} & \ctrl{2} & \swap{3} & \swap{3} & \gate{R_x(2.718)} & \gate{S^\dag} & \gate{R_x(-2.718)} & \qw \\
  \lstick{$q_1$} & \gate{H} & \gate{X} & \ctrl{1} & \gate{Y} & \ctrl{-1} & \gate{T^\dag} & \qw & \qw & \swap{1} & \targX{} & \qw & \swap{1} & \targX{} & \swap{1} & \ctrl{2} & \ctrl{2} & \gate{R_y(3.142)} & \ctrl{-1} & \gate{R_y(-3.142)} & \qw \\
  \lstick{$q_2$} & \gate{H} & \qw & \gate{Y} & \ctrl{-1} & \ctrl{-2} & \gate{Z} & \gate{S} & \targX{} & \targX{} & \qw & \targX{} & \targX{} & \qw & \targX{} & \qw & \ctrl{-2} & \gate{R_z(-3.142)} & \gate{R_z(3.142)} & \qw & \qw \\
  \lstick{$q_3$} & \gate{H} & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \qw & \ctrl{-3} & \ctrl{-2} & \targX{} & \targX{} & \qw & \qw & \qw & \qw \\
\end{quantikz}
\end{document}
"""
    # fmt: on
    expected_str = expected_str[1:]
    produced_string = cudaq.draw("latex", kernel)
    assert expected_str == produced_string


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
