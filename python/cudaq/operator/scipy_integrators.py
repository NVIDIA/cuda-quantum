from .integrator import BaseTimeStepper, BaseIntegrator
import cusuperop as cuso
import cupy
from scipy.integrate import ode
from scipy.integrate._ode import zvode


class cuSuperOpTimeStepper(BaseTimeStepper[cuso.State]):

    def __init__(self, liouvillian: cuso.Operator, ctx: cuso.WorkStream):
        self.liouvillian = liouvillian
        self.ctx = ctx
        self.state = None
        self.liouvillian_action = None

    def compute(self, state: cuso.State, t: float):
        if self.liouvillian_action is None:
            self.liouvillian_action = cuso.OperatorAction(
                self.ctx, (self.liouvillian,))

        if state != self.state:
            self.state = state
            self.liouvillian_action.prepare(self.ctx, (self.state,))

        action_result = cuso.DenseDensityMatrix(
            self.ctx, cupy.zeros_like(self.state.storage))
        self.liouvillian_action.compute(t, (), (self.state,), action_result)
        return action_result


class ScipyZvodeIntegrator(BaseIntegrator[cuso.State]):
    n_steps = 2500
    atol = 1e-8
    rtol = 1e-6
    order = 12

    def __init__(self, stepper: BaseTimeStepper[cuso.State], **kwargs):
        super().__init__(stepper, **kwargs)
        self.dm_shape = None

    def compute_rhs(self, t, vec):
        rho_data = cupy.asfortranarray(cupy.array(vec).reshape(self.dm_shape))
        temp_state = cuso.DenseDensityMatrix(self.state._ctx, rho_data)
        result = self.stepper.compute(temp_state, t)
        as_array = result.storage.ravel().get()
        return as_array

    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]

        if "atol" in self.integrator_options:
            self.atol = self.integrator_options["atol"]

        if "rtol" in self.integrator_options:
            self.rtol = self.integrator_options["rtol"]

        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]
        self.solver = ode(self.compute_rhs)
        self.solver.set_integrator("zvode")
        self.solver._integrator = zvode(method="adams",
                                        atol=self.atol,
                                        rtol=self.rtol,
                                        order=self.order,
                                        nsteps=self.n_steps)

    def integrate(self, t):
        if t <= self.t:
            raise ValueError(
                "Integration time must be greater than current time")
        new_state = self.solver.integrate(t)
        rho_data = cupy.asfortranarray(
            cupy.array(new_state).reshape(self.dm_shape))
        self.state.inplace_scale(0.0)
        self.state.inplace_add(
            cuso.DenseDensityMatrix(self.state._ctx, rho_data))
        self.t = t

    def set_state(self, state: cuso.State, t: float = 0.0):
        super().set_state(state, t)
        if self.dm_shape is None:
            self.dm_shape = self.state.storage.shape
        else:
            assert self.dm_shape == self.state.storage.shape, "State shape must remain constant"
        as_array = self.state.storage.ravel().get()
        self.solver.set_initial_value(as_array, t)
