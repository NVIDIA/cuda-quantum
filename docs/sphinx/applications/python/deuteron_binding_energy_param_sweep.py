import cudaq
from cudaq import spin
import numpy as np


@cudaq.kernel
def ansatz(angle: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(angle, q[1])
    x.ctrl(q[1], q[0])


hamiltonian = (
    5.907
    - 2.1433 * spin.x(0) * spin.x(1)
    - 2.1433 * spin.y(0) * spin.y(1)
    + 0.21829 * spin.z(0)
    - 6.125 * spin.z(1)
)

# Perform parameter sweep for deuteron N=2 Hamiltonian
for angle in np.linspace(-np.pi, np.pi, 25):
    # KERNEL::observe(...) <==>
    # E(params...) = <psi(params...) | H | psi(params...)>
    energyAtParam = cudaq.observe(ansatz, hamiltonian, 0.59)
    print("<H>({}) = {}".format(angle, energyAtParam))
