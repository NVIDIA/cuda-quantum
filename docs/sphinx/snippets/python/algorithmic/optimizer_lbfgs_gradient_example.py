import cudaq
from cudaq.optimizers import LBFGS
from cudaq.gradients import CentralDifference

# [Begin LBFGS Example Python]
@cudaq.kernel
def deuteron_n3_ansatz_py(x0: float, x1: float):
    q = cudaq.qvector(3)
    x(q[0])
    ry(x0, q[1])
    ry(x1, q[2])
    x.ctrl(q[2], q[0])
    x.ctrl(q[0], q[1]) # Assuming vctrl was a typo for ctrl
    ry(-x0, q[1])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[0])

h_deut_lbfgs_py = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + \
                 0.21829 * spin.z(0) - 6.125 * spin.z(1)
h3_deut_lbfgs_py = h_deut_lbfgs_py + 9.625 - 9.625 * spin.z(2) - \
                  3.913119 * spin.x(1) * spin.x(2) - 3.913119 * spin.y(1) * spin.y(2)

gradient_py = CentralDifference(deuteron_n3_ansatz_py)
optimizer_py = LBFGS()
optimizer_py.initial_parameters = [0.5, 0.5]

def objective_lbfgs_py(params: list[float]):
    cost_func = lambda p_lambda: cudaq.observe(deuteron_n3_ansatz_py, h3_deut_lbfgs_py, p_lambda[0], p_lambda[1]).expectation()
    current_cost = cost_func(params)
    grad_vector = gradient_py.compute(params, cost_func, current_cost)
    return current_cost, grad_vector

result_lbfgs = optimizer_py.optimize(dimensions=2, function=objective_lbfgs_py)

min_val_py = result_lbfgs.optimal_value
opt_params_py = result_lbfgs.optimal_parameters

print(f"L-BFGS Optimizer found {min_val_py} at {opt_params_py}")
# [End LBFGS Example Python]

if __name__ == "__main__":
    pass