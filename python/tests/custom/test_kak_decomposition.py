# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq

## NOTE: The random operations in this file are generated using
#        `scipy.stats.unitary_group.rvs(4)` with `seed=13`.


def check_state(matrix, state):
    # state must match the first column of the custom unitary matrix
    assert np.isclose(matrix[:, 0], np.array(state), atol=1e-8).all()


def test_random_unitary_1():
    # yapf: disable
    matrix1 = np.array([
        [-0.25534142 + 0.04562918j, 0.11619328 + 0.7978548j,  0.19980911 - 0.24754117j,  0.05245516 + 0.42272181j],
        [ 0.48212336 - 0.35275169j, 0.47307302 + 0.204771j,   0.38804407 + 0.34346751j, -0.30236462 - 0.13199084j],
        [ 0.53000373 - 0.05204794j,-0.05546452 + 0.04480838j,-0.39853872 - 0.60358143j, -0.40979785 + 0.1422147j],
        [ 0.20174057 + 0.50152752j, 0.04256283 - 0.2780322j,  0.14896845 + 0.29140402j, -0.16938781 + 0.70203793j]])
    # yapf: enable

    cudaq.register_operation("op1", matrix1)

    @cudaq.kernel
    def kernel1():
        q = cudaq.qvector(2)
        op1(q[1], q[0])

    # print(kernel1)
    cudaq.get_state(kernel1).dump()
    check_state(matrix1, cudaq.get_state(kernel1))

    @cudaq.kernel
    def synth_kernel1():
        q = cudaq.qvector(2)
        
        rz(3.9582625248746566, q[0])
        ry(0.93802610748277016, q[0])
        rz(2.2568237856512323, q[0])
        
        rz(-1.9066099708330588, q[1])
        ry(2.783651792391125, q[1])
        rz(0.7280736766746525, q[1])
        
        exp_pauli(-0.57180470583457921, q, "XX")
        # h(q[0])
        # h(q[1])
        # x.ctrl(q[0], q[1])
        # rz(-0.57180470583457921, q[1])
        # x.ctrl(q[0], q[1])
        # h(q[1])
        # h(q[0])
        
        exp_pauli(-0.06673487344215917, q, "YY")
        # rx(1.5707963267948966, q[1])
        # rx(1.5707963267948966, q[0])
        # x.ctrl(q[1], q[0])
        # rz(-0.06673487344215917, q[0])
        # x.ctrl(q[1], q[0])
        # rx(-1.5707963267948966, q[0])
        # rx(-1.5707963267948966, q[1])
        
        exp_pauli(0.21810769954508441, q, "ZZ")
        # x.ctrl(q[1], q[0])
        # rz(0.21810769954508441, q[0])
        # x.ctrl(q[1], q[0])
        
        rz(1.6888584582114208, q[0])
        ry(2.2872369478030228, q[0])
        rz(-3.1401730467170035, q[0])
        
        rz(2.088853123967366, q[1])
        ry(2.0186522227162649, q[1])
        rz(-0.20630121734301887, q[1])
        
        r1(-1.2996367006005645, q[0])
        rz(1.2996367006005645, q[0])

    cudaq.get_state(synth_kernel1).dump()
    check_state(matrix1, cudaq.get_state(synth_kernel1))


def test_random_unitary_2():
    # yapf: disable
    matrix2 = np.array([
        [-0.29131252-0.10903507j, -0.2101072 +0.18218874j, -0.08729306-0.28146386j,  0.21049735-0.83352233j],
        [ 0.1863529 -0.18035587j, -0.64602307-0.18970824j, -0.27983504+0.61560257j, -0.08699033-0.12069728j],
        [-0.14883506-0.02698946j,  0.57821686+0.20023664j, -0.2694992 +0.61062667j,  0.37462953-0.12980067j],
        [ 0.86768584-0.24542555j,  0.21850017+0.21715302j,  0.07738398-0.06633863j, -0.05929165-0.27943735j]])
    # yapf: enable

    cudaq.register_operation("op2", matrix2)

    @cudaq.kernel
    def kernel2():
        q = cudaq.qvector(2)
        op2(q[1], q[0])

    # print(kernel2)
    # cudaq.get_state(kernel2).dump()
    check_state(matrix2, cudaq.get_state(kernel2))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
