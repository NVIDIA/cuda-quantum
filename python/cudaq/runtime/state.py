# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def to_cupy(state, dtype=None):
    """
    A CUDA Quantum state is composed of a list of tensors (e.g. state-vector 
    state is composed of a single rank-1 tensor). Map all tensors 
    """
    try:
        import cupy as cp
    except ImportError:
        print('to_cupy not supported, CuPy not available. Please install CuPy.')

    if dtype == None:
        dtype = cp.complex64

    if not state.is_on_gpu():
        raise RuntimeError(
            "cudaq.to_cupy invoked but the state is not on the GPU.")

    arrays = []
    for tensor in state.getTensors():
        mem = cp.cuda.UnownedMemory(tensor.data(), 1024, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, offset=0)
        arrays.append(cp.ndarray(tensor.extents, dtype=dtype, memptr=memptr))
    return arrays
