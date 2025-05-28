import cudaq
from cudaq import spin, x, y, z, ry, rx, cx
from cudaq.optimizers import COBYLA

# [Begin COBYLA Example Python]
@cudaq.kernel
def ansatz_cobyla(theta: float, phi: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    rx(phi, q[0])
    cx(q[1], q[0])

H_cobyla = spin.z(0) * spin.x(1) + 0.5 * spin.y(0)

optimizer = COBYLA()
optimizer.initial_parameters = [0.1, 0.1]
optimizer.max_iterations = 100

def objective_cobyla(params: list[float]) -> float:
    return cudaq::observe(ansatz_cobyla, H_cobyla, params[0], params[1]).expectation()

result_cobyla = optimizer.optimize(dimensions=2, function=objective_cobyla)

print(f"COBYLA Optimal value: {result_cobyla.optimal_value}")
print(f"COBYLA Optimal parameters: {result_cobyla.optimal_parameters}")
# [End COBYLA Example Python]

if __name__ == "__main__":
    pass