# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq

## NOTE: The random operations in this file are generated using
#        `scipy.stats.unitary_group.rvs(4)` with `seed=13`. The synthesized
#        kernels are generated by running transformation passes on the original
#        kernels which use the custom operation. These conversions are covered
#        in the `test/Transforms/UnitarySynthesis/random_unitary_*` tests.


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

        h(q[1])
        h(q[0])
        x.ctrl(q[0], q[1])
        rz(1.1436094116691584, q[1])
        x.ctrl(q[0], q[1])
        h(q[0])
        h(q[1])

        rx(1.5707963267948966, q[1])
        rx(1.5707963267948966, q[0])
        x.ctrl(q[0], q[1])
        rz(0.13346974688431834, q[1])
        x.ctrl(q[0], q[1])
        rx(-1.5707963267948966, q[0])
        rx(-1.5707963267948966, q[1])

        x.ctrl(q[0], q[1])
        rz(-0.43621539909016882, q[1])
        x.ctrl(q[0], q[1])

        rz(1.6888584582114208, q[0])
        ry(2.2872369478030228, q[0])
        rz(-3.1401730467170035, q[0])

        rz(2.088853123967366, q[1])
        ry(2.0186522227162649, q[1])
        rz(-0.20630121734301887, q[1])

        r1(-1.2996367006005645, q[1])
        rz(1.2996367006005645, q[1])

    check_state(matrix1, cudaq.get_state(synth_kernel1))


def test_random_unitary_2():
    # yapf: disable
    matrix2 = np.array([[ 0.18897759+0.33963024j,  0.12335642-0.48243451j, 0.42873799-0.22386284j, -0.38231687-0.46998072j],
                        [ 0.26665664+0.31917547j,  0.66539471+0.25221666j,-0.47503402-0.12900718j, -0.26305423+0.09570885j],
                        [-0.1821702 +0.14533363j,  0.18060332-0.34169107j, 0.00131404-0.64370213j,  0.54215898+0.29670066j],
                        [-0.30045971+0.72895551j, -0.26715636-0.15790473j,-0.06966553+0.32335977j, -0.13738248+0.39211303j]])
    # yapf: enable

    cudaq.register_operation("op2", matrix2)

    @cudaq.kernel
    def kernel2():
        q = cudaq.qvector(2)
        op2(q[1], q[0])

    check_state(matrix2, cudaq.get_state(kernel2))

    @cudaq.kernel
    def synth_kernel2():
        q = cudaq.qvector(2)

        rz(3.3597983877882998, q[0])
        ry(1.1124416939078243, q[0])
        rz(-1.5227607222807453, q[0])

        rz(1.0022361850018475, q[1])
        ry(2.3499858725474598, q[1])
        rz(0.70669321414482034, q[1])

        h(q[1])
        h(q[0])
        x.ctrl(q[0], q[1])
        rz(0.41098890378696051, q[1])
        x.ctrl(q[0], q[1])
        h(q[0])
        h(q[1])

        rx(1.5707963267948966, q[1])
        rx(1.5707963267948966, q[0])
        x.ctrl(q[0], q[1])
        rz(-4.0833361355387012, q[1])
        x.ctrl(q[0], q[1])
        rx(-1.5707963267948966, q[0])
        rx(-1.5707963267948966, q[1])

        x.ctrl(q[0], q[1])
        rz(1.2323317339216211, q[1])
        x.ctrl(q[0], q[1])

        rz(-0.57588264019689317, q[0])
        ry(0.45370093726152877, q[0])
        rz(0.63586258232390358, q[0])

        rz(0.44527705872095541, q[1])
        ry(1.7688004823405488, q[1])
        rz(1.0308660415707038, q[1])

        r1(0.89327181859161264, q[1])
        rz(-0.89327181859161264, q[1])

    check_state(matrix2, cudaq.get_state(synth_kernel2))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
