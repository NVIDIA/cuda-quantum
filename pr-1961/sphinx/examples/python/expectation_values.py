# The example here shows a simple use case for the `cudaq::observe``
# function in computing expected values of provided spin operators.

import cudaq
from cudaq import spin


@cudaq.kernel
def kernel(theta: float):
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(theta, qvector[1])
    x.ctrl(qvector[1], qvector[0])


spin_operator = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

# Pre-computed angle that minimizes the energy expectation of the `spin_operator`.
angle = 0.59

energy = cudaq.observe(kernel, spin_operator, angle).expectation()
print(f"Energy is {energy}")
