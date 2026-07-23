# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


@cudaq.kernel
def leaf(q: cudaq.qview):
    for i in range(q.size()):
        x(q[i])


@cudaq.kernel
def composite(q: cudaq.qview):
    leaf(q)


@cudaq.kernel
def caller():
    c = cudaq.qubit()
    q = cudaq.qvector(2)
    x(c)
    cudaq.control(leaf, c, q)
    cudaq.control(composite, c, q)


cudaq.sample(caller)
