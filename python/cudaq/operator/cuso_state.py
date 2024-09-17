from cusuperop import DenseMixedState, WorkStream
import cupy, atexit

class CuSuperOpState(object):
    __ctx = WorkStream()
    
    def __init__(self, data):
        if isinstance(data, DenseMixedState):
            self.density_matrix = data
        else:
            rho0_ = cupy.asfortranarray(data)
            self.density_matrix = DenseMixedState(self.__ctx, rho0_)

    @staticmethod
    def from_data(data):
        return CuSuperOpState(data)  
     
    def get_impl(self):
        return self.density_matrix

    def dump(self):
        return cupy.array_str(self.density_matrix.storage)

    @staticmethod
    def tear_down():
        CuSuperOpState.__ctx.free()

atexit.register(CuSuperOpState.tear_down)