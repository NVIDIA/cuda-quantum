
from .integrator import BaseTimeStepper, BaseIntegrator
import cusuperop as cuso
import cupy 

class cuSuperOpTimeStepper(BaseTimeStepper[cuso.State]):
    def __init__(self, liouvillian: cuso.Operator, ctx: cuso.WorkStream):
        self.liouvillian = liouvillian
        self.ctx = ctx
        self.state = None
        self.liouvillian_action = None

    def compute(self, state: cuso.State, t: float):
        if self.liouvillian_action is None:
            self.liouvillian_action = cuso.OperatorAction(self.ctx, (self.liouvillian, ))

        if state != self.state:
            self.state = state 
            self.liouvillian_action.prepare(self.ctx, (self.state ,)) 
         
        action_result = cuso.DenseDensityMatrix(self.ctx, cupy.zeros_like(self.state.storage))
        self.liouvillian_action.compute(t, (), (self.state,), action_result)
        return action_result

class RungeKuttaIntegrator(BaseIntegrator[cuso.State]):
    n_steps = 100
    
    def __init__(self, stepper: BaseTimeStepper[cuso.State], **kwargs):
        super().__init__(stepper, **kwargs)
        
    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]
    
    def integrate(self, t):
        if self.state is None:
            raise ValueError("Initial state is not set")
        self.ctx = self.state._ctx

        if t <= self.t:
            raise ValueError("Integration time must be greater than current time")
        dt = (t - self.t)/self.n_steps
        for i in range(self.n_steps):
            current_t = self.t + i * dt
            k1 = self.stepper.compute(self.state, current_t)
            
            
            rho_temp = cupy.copy(self.state.storage)
            rho_temp += ((dt/2) * k1.storage)
            k2 = self.stepper.compute(cuso.DenseDensityMatrix(self.ctx, rho_temp), current_t + dt/2)
            
            rho_temp = cupy.copy(self.state.storage)
            rho_temp += ((dt/2) * k2.storage)
            k3 = self.stepper.compute(cuso.DenseDensityMatrix(self.ctx, rho_temp), current_t + dt/2)
           
            rho_temp = cupy.copy(self.state.storage)
            rho_temp += ((dt) * k3.storage)
            k4 = self.stepper.compute(cuso.DenseDensityMatrix(self.ctx, rho_temp), current_t + dt)
            
            # Scale      
            k1.inplace_scale(dt/6)
            k2.inplace_scale(dt/3)
            k3.inplace_scale(dt/3)
            k4.inplace_scale(dt/6)
            
            self.state.inplace_add(k1)
            self.state.inplace_add(k2)
            self.state.inplace_add(k3)
            self.state.inplace_add(k4)
        self.t = t
