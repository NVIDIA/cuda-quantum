# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

# For data = [0.5, 0.5, 0.5] and pad = 0, padding gives a 4-amplitude state
# normalized to (|0⟩ + |1⟩ + |2⟩) / √3.
state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
print(state)
