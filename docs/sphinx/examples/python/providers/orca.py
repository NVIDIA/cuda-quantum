# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np
import os

orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost:8080/sample")

cudaq.set_target("orca", url=orca_url)

# [Documentation TODO]: Explanation of the following terms and APIs

bs_angles = [np.pi / 3, np.pi / 6]
ps_angles = [np.pi / 4, np.pi / 5]

input_state = [1, 1, 1]
loop_lengths = [1]

counts = cudaq.orca.sample(bs_angles, ps_angles, input_state, loop_lengths)

print(counts)
