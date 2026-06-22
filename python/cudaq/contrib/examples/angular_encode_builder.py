# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

kernel = cudaq.make_kernel()
q = kernel.qalloc(3)
cudaq.contrib.angular_encode(kernel, q, [0.1, 0.2, 0.3], rotation='Y')

print(cudaq.draw(kernel))
