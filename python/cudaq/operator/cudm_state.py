# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
from cuquantum.bindings import cudensitymat as cudm
from .helpers import InitialState


def is_multi_processes():
    return cudaq_runtime.mpi.is_initialized() and cudaq_runtime.mpi.num_ranks(
    ) > 1


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
    __is_multi_process = False

    def __init__(self, data):
        if self.__ctx is None:
            if (is_multi_processes()):
                NUM_DEVICES = cupy.cuda.runtime.getDeviceCount()
                rank = cudaq_runtime.mpi.rank()
                dev = cupy.cuda.Device(rank % NUM_DEVICES)
                dev.use()
                self.__ctx = WorkStream(device_id=cupy.cuda.runtime.getDevice())
                # FIXME: use the below once `cudensitymat` supports raw MPI Comm pointer.
                # `ctx.set_communicator(comm=cudaq_runtime.mpi.comm_dup(), provider="MPI")`
                # At the moment, only `mpi4py` communicator objects are supported, thus we use the underlying `reset_distributed_configuration` API.
                _comm_ptr, _size = cudaq_runtime.mpi.comm_dup()
                cudm.reset_distributed_configuration(
                    self.__ctx._handle._validated_ptr,
                    cudm.DistributedProvider.MPI, _comm_ptr, _size)
                CuDensityMatState.__is_multi_process = True
            else:
                self.__ctx = WorkStream()

        self.hilbert_space_dims = None
        if isinstance(data, DenseMixedState) or isinstance(
                data, DensePureState):
            self.state = data
            self.raw_data = self.state.storage
        else:
            self.raw_data = data
            self.state = None

    @staticmethod
    def is_multi_process():
        # Returns true if MPI distribution is activated
        return CuDensityMatState.__is_multi_process

    def try_init_state(self, shape):
        """Try initialize state according to a shape, e.g., density matrix or state vector."""
        slice_shape, slice_offsets = self.state.local_info
        state_size = numpy.prod(shape)
        if state_size == self.raw_data.size:
            slice_obj = tuple(
                slice(offset, offset + size) for offset, size in zip(
                    slice_offsets, slice_shape))[:len(shape)]
            self.raw_data = cupy.asfortranarray(self.raw_data.reshape(shape))
            self.raw_data = cupy.asfortranarray(self.raw_data[slice_obj].copy())
            self.state.attach_storage(self.raw_data)
        else:
            self.raw_data = cupy.asfortranarray(
                self.raw_data.reshape(slice_shape))
            self.state.attach_storage(self.raw_data)

    def init_state(self, hilbert_space_dims: Sequence[int]):
        if self.state is None:
            self.hilbert_space_dims = hilbert_space_dims
            dm_shape = hilbert_space_dims * 2
            sv_shape = hilbert_space_dims
            try:
                self.state = DenseMixedState(self.__ctx,
                                             self.hilbert_space_dims,
                                             batch_size=1,
                                             dtype="complex128")
                self.try_init_state(dm_shape)
            except:
                try:
                    self.state = DensePureState(self.__ctx,
                                                self.hilbert_space_dims,
                                                batch_size=1,
                                                dtype="complex128")
                    self.try_init_state(sv_shape)
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

    @staticmethod
    def create_initial_state(type: InitialState,
                             hilbert_space_dims: Sequence[int],
                             mix_state: bool):
        new_state = CuDensityMatState(None)
        new_state.hilbert_space_dims = hilbert_space_dims

        if mix_state:
            new_state.state = DenseMixedState(new_state.__ctx,
                                              new_state.hilbert_space_dims,
                                              batch_size=1,
                                              dtype="complex128")
        else:
            new_state.state = DensePureState(new_state.__ctx,
                                             new_state.hilbert_space_dims,
                                             batch_size=1,
                                             dtype="complex128")
        required_buffer_size = new_state.state.storage_size
        slice_shape, slice_offsets = new_state.state.local_info

        if type == InitialState.ZERO:
            buffer = cupy.asfortranarray(
                cupy.zeros((required_buffer_size,),
                           dtype="complex128",
                           order="F"))
            bitstring_is_local = False
            # Follow `cudensitymat` example to set the amplitude based on `local_info`
            for slice_dim, slice_offset in zip(slice_shape, slice_offsets):
                bitstring_is_local = 0 in range(slice_offset,
                                                slice_offset + slice_dim)
                if not bitstring_is_local:
                    break
            if bitstring_is_local:
                buffer[0] = 1.0
            new_state.state.raw_data = cupy.asfortranarray(buffer)
            new_state.state.attach_storage(new_state.state.raw_data)
        elif type == InitialState.UNIFORM:
            buffer = cupy.asfortranarray(
                cupy.zeros((required_buffer_size,),
                           dtype="complex128",
                           order="F"))
            hilberg_space_size = numpy.cumprod(hilbert_space_dims)[-1]
            if mix_state:
                dm_shape = hilbert_space_dims * 2
                # FIXME: currently, we use host-device data transfer, hence a host allocation is required.
                # A custom GPU memory initialization can be also used so that no host allocation is needed.
                host_array = (1. / (hilberg_space_size)) * numpy.identity(
                    hilberg_space_size, dtype="complex128")
                host_array = host_array.reshape(dm_shape)
                slice_obj = []
                for i in range(len(slice_offsets) - 1):
                    slice_obj.append(
                        slice(slice_offsets[i],
                              slice_offsets[i] + slice_shape[i]))
                slice_obj = tuple(slice_obj)
                sliced_host_array = numpy.ravel(host_array[slice_obj].copy())
                buffer = cupy.array(sliced_host_array)
            else:
                buffer[:] = 1. / numpy.sqrt(hilberg_space_size)

            new_state.state.raw_data = cupy.asfortranarray(buffer)
            new_state.state.attach_storage(new_state.state.raw_data)
        else:
            raise ValueError("Unsupported initial state type")

        return new_state

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
