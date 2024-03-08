# The example here shows a simple use case for the `cudaq::observe``
# function in computing expected values of provided spin operators.

import cudaq
from cudaq import spin

kernel, theta = cudaq.make_kernel(float)
qvector = kernel.qalloc(2)
kernel.x(qvector[0])
kernel.ry(theta, qvector[1])
kernel.cx(qvector[1], qvector[0])

h = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(
    1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

energy = cudaq.observe(kernel, h, 0.59).expectation()
print(f"Energy is {energy}")
