import cudaq
from cudaq.operator import *

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# In this example, we simulate a two-level system that decays into a squeezed vacuum state.
# The master equation is given in "The theory of open quantum systems", by `Francesco Petruccione and Heinz-Peter Breuer`, section 3.4.3 - 3.4.4.


def n_thermal(w: float, w_th: float):
    """
    Return the number of average photons in thermal equilibrium for a
        an oscillator with the given frequency and temperature.
    """
    if (w_th > 0) and np.exp(w / w_th) != 1.0:
        return 1.0 / (np.exp(w / w_th) - 1.0)
    else:
        return 0.0


# Problem parameters
w0 = 1.0 * 2 * np.pi
gamma0 = 0.05
# the temperature of the environment
w_th = 0.0 * 2 * np.pi
# the number of average excitations in the environment mode `w0` at temperature `w_th`
Nth = n_thermal(w0, w_th)
# squeezing parameter for the environment
r = 1.0
theta = 0.1 * np.pi
N = Nth * (np.cosh(r)**2 + np.sinh(r)**2) + np.sinh(r)**2
sz_ss_analytical = -1 / (2 * N + 1)
print(f"Analytical squeezing parameter: {sz_ss_analytical}")

# System Hamiltonian
hamiltonian = -0.5 * w0 * pauli.z(0)
# Collapse operators
c_ops = [
    np.sqrt(gamma0) * (pauli.minus(0) * np.cosh(r) +
                       pauli.plus(0) * np.sinh(r) * np.exp(1j * theta))
]

# System dimension
dimensions = {0: 2}
# Start in an arbitrary superposition state
psi0_ = cp.array([2j, 1.0], dtype=cp.complex128)
psi0_ = psi0_ / cp.linalg.norm(psi0_)
psi0 = cudaq.State.from_data(psi0_)
# Simulation time points
steps = np.linspace(0, 50, 1001)
schedule = Schedule(steps, ["time"])

# Run the simulation
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          psi0,
                          observables=[pauli.x(0),
                                       pauli.y(0),
                                       pauli.z(0)],
                          collapse_operators=c_ops,
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator())

exp_val_x = [
    exp_vals[0].expectation()
    for exp_vals in evolution_result.expectation_values()
]
exp_val_y = [
    exp_vals[1].expectation()
    for exp_vals in evolution_result.expectation_values()
]
exp_val_z = [
    exp_vals[2].expectation()
    for exp_vals in evolution_result.expectation_values()
]

# Plot the results
fig = plt.figure(figsize=(12, 6))
plt.plot(steps, exp_val_x)
plt.plot(steps, exp_val_y)
plt.plot(steps, exp_val_z)

plt.plot(steps, sz_ss_analytical * np.ones(np.shape(steps)), 'k--')
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z", "Analytical"))
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig("squeezing.png", dpi=fig.dpi)
