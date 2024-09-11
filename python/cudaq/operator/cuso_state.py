from cusuperop import DenseDensityMatrix, WorkStream
import cupy, atexit

class CuSuperOpState(object):
    __ctx = WorkStream()
    
    def __init__(self, data):
        rho0_ = cupy.asfortranarray(data)
        self.density_matrix = DenseDensityMatrix(self.__ctx, rho0_)
    
    @staticmethod
    def from_data(data):
        return CuSuperOpState(data)  
     
    def get_impl(self):
        return self.density_matrix

    @staticmethod
    def tear_down():
        CuSuperOpState.__ctx.free()

atexit.register(CuSuperOpState.tear_down)