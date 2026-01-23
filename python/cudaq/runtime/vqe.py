# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def vqe(*args,
        kernel=None,
        gradient_strategy=None,
        spin_operator=None,
        optimizer=None,
        parameter_count=None,
        argument_wrapper=None,
        shots=None):
    raise RuntimeError("This CUDA-Q function has been removed. Please use the "
                       "VQE implementation from CUDA-QX.")
