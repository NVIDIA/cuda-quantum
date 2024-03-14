# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np


def fermionic_swap_builder(kernel, phi, q0, q1):
    kernel.h(q0)
    kernel.h(q1)

    kernel.cx(q0, q1)
    kernel.rz(phi / 2.0, q1)
    kernel.cx(q0, q1)

    kernel.h(q0)
    kernel.h(q1)

    kernel.rx(np.pi / 2., q0)
    kernel.rx(np.pi / 2., q1)

    kernel.cx(q0, q1)
    kernel.rz(phi / 2.0, q1)
    kernel.cx(q0, q1)

    kernel.rx(-np.pi / 2., q0)
    kernel.rx(-np.pi / 2., q1)
    kernel.rz(phi / 2.0, q0)
    kernel.rz(phi / 2.0, q1)

    # Global phase correction
    kernel.r1(phi, q0)
    kernel.rz(-phi, q0)
