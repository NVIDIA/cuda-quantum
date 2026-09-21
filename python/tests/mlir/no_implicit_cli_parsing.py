# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that importing CUDA-Q does not consume host application arguments.
# RUN: PYTHONPATH=../../ python3 %s --target host-application-target | FileCheck %s

import cudaq

print("CUDA-Q import ignored host arguments")

# CHECK: CUDA-Q import ignored host arguments
