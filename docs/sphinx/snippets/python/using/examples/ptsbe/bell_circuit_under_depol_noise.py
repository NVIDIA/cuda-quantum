# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin PTSBE_Bell]
import cudaq
from cudaq import ptsbe
from utils import bell, noise

result = ptsbe.sample(bell, shots_count=10_000, noise_model=noise)
print(result)
#[End PTSBE_Bell]