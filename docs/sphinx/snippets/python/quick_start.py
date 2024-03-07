# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq

print(f"Simulation Target = {cudaq.get_target().name}")

kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.h(qubit)
kernel.mz(qubit)

result = cudaq.sample(kernel)
print(result)  # Example: { 1:500 0:500 }
