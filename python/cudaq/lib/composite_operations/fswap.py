# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
import numpy as np


@cudaq.kernel
def fermionic_swap(phi: float, q0: cudaq.qubit, q1: cudaq.qubit):
    h(q0)
    h(q1)

    x.ctrl(q0, q1)
    rz(phi / 2.0, q1)
    x.ctrl(q0, q1)

    h(q0)
    h(q1)

    rx(np.pi / 2., q0)
    rx(np.pi / 2., q1)

    x.ctrl(q0, q1)
    rz(phi / 2.0, q1)
    x.ctrl(q0, q1)

    rx(-np.pi / 2., q0)
    rx(-np.pi / 2., q1)
    rz(phi / 2.0, q0)
    rz(phi / 2.0, q1)

    # Global phase correction
    r1(phi, q0)
    rz(-phi, q0)
