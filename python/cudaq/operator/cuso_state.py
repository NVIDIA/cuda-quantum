from cusuperop import DenseMixedState, DensePureState, WorkStream
import cupy, atexit
from typing import Sequence

class CuSuperOpState(object):
    __ctx = WorkStream()
    
    def __init__(self, data):
        if isinstance(data, DenseMixedState) or isinstance(data, DensePureState):
            self.state = data
            self.raw_data = self.state.storage
        else:
            self.raw_data = data
            self.state = None

    def init_state(self, hilbert_space_dims: Sequence[int]):
        if self.state is None:
            dm_shape = hilbert_space_dims * 2
            sv_shape = hilbert_space_dims
            try:
                self.raw_data = cupy.asfortranarray(self.raw_data.reshape(dm_shape))
                self.state = DenseMixedState(self.__ctx, self.raw_data)
            except:
                try:
                    self.raw_data = cupy.asfortranarray(self.raw_data.reshape(sv_shape))
                    self.state = DensePureState(self.__ctx, self.raw_data)
                except:
                    raise ValueError(f"Invalid state data: state data must be either a state vector (equivalent to {sv_shape} shape) or a density matrix (equivalent to {dm_shape} shape).")
    
    def is_initialized(self) -> bool:
        return self.state is not None

    def is_density_matrix(self) -> bool:
        return self.is_initialized() and isinstance(self.state, DenseMixedState)

    @staticmethod
    def from_data(data):
        return CuSuperOpState(data)  
     
    def get_impl(self):
        return self.state

    def dump(self):
        if self.state is None:
            return cupy.array_str(self.raw_data)
        return cupy.array_str(self.state.storage)

    @staticmethod
    def tear_down():
        CuSuperOpState.__ctx.free()

atexit.register(CuSuperOpState.tear_down)