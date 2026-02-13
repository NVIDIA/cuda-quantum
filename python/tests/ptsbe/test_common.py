# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def rotation_kernel(angle: float):
    q = cudaq.qvector(1)
    ry(angle, q[0])
    mz(q)


@cudaq.kernel
def x_op():
    q = cudaq.qvector(1)
    x(q[0])
    mz(q)


@cudaq.kernel
def phase_flip_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    z(q[0])
    h(q[0])
    mz(q)


@cudaq.kernel
def cnot_echo():
    q = cudaq.qvector(2)
    x.ctrl(q[0], q[1])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def mcm_kernel():
    q = cudaq.qvector(2)
    h(q[0])
    b = mz(q[0])
    if b:
        x(q[1])
    mz(q)


def ptsbe_target_setup():
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(42)


def ptsbe_target_teardown():
    cudaq.reset_target()


def make_depol_noise():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.1), num_controls=1)
    return noise
