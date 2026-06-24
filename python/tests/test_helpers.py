# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from cudaq import spin


def h2_hamiltonian_4q():
    """
    H2 molecule Hamiltonian (4 qubits).
    """
    h = cudaq.SpinOperator.empty()
    h += 0.0454063 * (spin.y(0) * spin.x(1) * spin.x(2) * spin.y(3))
    h += 0.17028 * (spin.z(0))
    h += -0.220041 * (spin.z(2))
    h += 0.0454063 * (spin.x(0) * spin.y(1) * spin.y(2) * spin.x(3))
    h += -0.106477 * (1.0)
    h += 0.17028 * (spin.z(1))
    h += -0.220041 * (spin.z(3))
    h += -0.0454063 * (spin.y(0) * spin.y(1) * spin.x(2) * spin.x(3))
    h += 0.168336 * (spin.z(0) * spin.z(1))
    h += 0.1202 * (spin.z(0) * spin.z(2))
    h += 0.1202 * (spin.z(1) * spin.z(3))
    h += 0.165607 * (spin.z(0) * spin.z(3))
    h += 0.165607 * (spin.z(1) * spin.z(2))
    h += 0.174073 * (spin.z(2) * spin.z(3))
    h += -0.0454063 * (spin.x(0) * spin.x(1) * spin.y(2) * spin.y(3))
    return h
