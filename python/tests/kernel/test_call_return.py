# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


def test():

    @cudaq.kernel()
    def callMe(q: cudaq.qubit) -> bool:
        h(q)
        m = mz(q)
        return m

    @cudaq.kernel()
    def IWillCallYou() -> bool:
        q = cudaq.qubit()
        t = callMe(q)
        return t

    print(IWillCallYou())
