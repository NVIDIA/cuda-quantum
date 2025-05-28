import cudaq
from cudaq import spin, x, ry 
import numpy as np 

# [Begin Deuteron Sweep Python]
@cudaq.kernel
def ansatz(angle:float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(angle, q[1])
    x.ctrl(q[1], q[0])

hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

# Perform parameter sweep for deuteron N=2 Hamiltonian
print("Param Sweep <H>(angle) = energy")
for angle in np.linspace(-np.pi, np.pi, 25):
     # KERNEL::observe(...) <==>
     # E(params...) = <psi(params...) | H | psi(params...)>
     # Corrected: use 'angle' not fixed '.59'
     energyAtParam = cudaq.observe(ansatz, hamiltonian, angle).expectation()
     print(f'<H>({angle}) = {energyAtParam}')
# [End Deuteron Sweep Python]

if __name__ == "__main__":
    pass # Logic is at top level