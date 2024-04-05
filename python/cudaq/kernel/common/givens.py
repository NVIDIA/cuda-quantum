# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def givens_builder(builder, theta, qubitA, qubitB):
    builder.exp_pauli(-.5 * theta, 'YX', qubitA, qubitB)
    builder.exp_pauli(.5 * theta, 'XY', qubitA, qubitB)
