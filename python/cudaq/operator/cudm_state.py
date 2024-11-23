# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cuquantum.densitymat import DenseMixedState, DensePureState, WorkStream
import numpy, cupy, atexit
from typing import Sequence
from cupy.cuda.memory import MemoryPointer, UnownedMemory
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


# Wrap state data (on device memory) as a `cupy` array.
# Note: the `cupy` array only holds a reference to the GPU memory buffer, no copy.
def to_cupy_array(state):
    tensor = state.getTensor()
    pDevice = tensor.data()
    dtype = cupy.complex128
    sizeByte = tensor.get_num_elements() * tensor.get_element_size()
    # Use `UnownedMemory` to wrap the device pointer
    mem = UnownedMemory(pDevice, sizeByte, owner=state)
    memptr = MemoryPointer(mem, 0)
    cupy_array = cupy.ndarray(tensor.get_num_elements(),
                              dtype=dtype,
                              memptr=memptr)
    return cupy_array


# A Python wrapper of `CuDensityMatState` state.
class CuDensityMatState(object):
    __ctx = None

    def __init__(self, data):
        if self.__ctx is None:
            self.__ctx = WorkStream()

        self.hilbert_space_dims = None
        if isinstance(data, DenseMixedState) or isinstance(
                data, DensePureState):
            self.state = data
            self.raw_data = self.state.storage
        else:
            self.raw_data = data
            self.state = None

    def init_state(self, hilbert_space_dims: Sequence[int]):
        if self.state is None:
            self.hilbert_space_dims = hilbert_space_dims
            dm_shape = hilbert_space_dims * 2
            sv_shape = hilbert_space_dims
            try:
                self.raw_data = cupy.asfortranarray(
                    self.raw_data.reshape(dm_shape))
                self.state = DenseMixedState(self.__ctx,
                                             self.hilbert_space_dims,
                                             batch_size=1,
                                             dtype="complex128")
                self.state.attach_storage(self.raw_data)
            except:
                try:
                    self.raw_data = cupy.asfortranarray(
                        self.raw_data.reshape(sv_shape))
                    self.state = DensePureState(self.__ctx,
                                                self.hilbert_space_dims,
                                                batch_size=1,
                                                dtype="complex128")
                    self.state.attach_storage(self.raw_data)
                except:
                    raise ValueError(
                        f"Invalid state data: state data must be either a state vector (equivalent to {sv_shape} shape) or a density matrix (equivalent to {dm_shape} shape)."
                    )

    def is_initialized(self) -> bool:
        return self.state is not None

    def is_density_matrix(self) -> bool:
        return self.is_initialized() and isinstance(self.state, DenseMixedState)

    @staticmethod
    def from_data(data):
        return CuDensityMatState(data)

    def get_impl(self):
        return self.state

    def dump(self):
        if self.state is None:
            return cupy.array_str(self.raw_data)
        return cupy.array_str(self.state.storage)

    def to_dm(self):
        if self.is_density_matrix():
            raise ValueError("CuDensityMatState is already a density matrix")
        dm = cupy.outer(self.state.storage, cupy.conj(self.state.storage))
        if self.hilbert_space_dims is not None:
            dm = dm.reshape(self.hilbert_space_dims * 2)
        dm = cupy.asfortranarray(dm)
        dm_state = DenseMixedState(self.__ctx,
                                   self.hilbert_space_dims,
                                   batch_size=1,
                                   dtype="complex128")
        dm_state.attach_storage(dm)
        return CuDensityMatState(dm_state)


# Wrap a CUDA-Q state as a `CuDensityMatState`
def as_cudm_state(state):
    return CuDensityMatState(to_cupy_array(state))
