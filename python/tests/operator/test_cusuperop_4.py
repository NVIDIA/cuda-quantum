import cudaq
from cudaq.operator import *

import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")

# Decay into a squeezed vacuum field
# https://github.com/jrjohansson/qutip-lectures/blob/master/Lecture-12-Decay-into-a-squeezed-vacuum-field.ipynb
#


def n_thermal(w, w_th):
    """
    Return the number of photons in thermal equilibrium for an harmonic
    oscillator mode with frequency 'w', at the temperature described by
    'w_th' where :math:`\\omega_{\\rm th} = k_BT/\\hbar`.

    Parameters
    ----------

    w : *float* or *array*
        Frequency of the oscillator.

    w_th : *float*
        The temperature in units of frequency (or the same units as `w`).


    Returns
    -------

    n_avg : *float* or *array*

        Return the number of average photons in thermal equilibrium for a
        an oscillator with the given frequency and temperature.


    """

    if type(w) is np.ndarray:
        return 1.0 / (np.exp(w / w_th) - 1.0)

    else:
        if (w_th > 0) and np.exp(w / w_th) != 1.0:
            return 1.0 / (np.exp(w / w_th) - 1.0)
        else:
            return 0.0


# Problem parameter
w0 = 1.0 * 2 * np.pi
gamma0 = 0.05

# the temperature of the environment in frequency units
w_th = 0.0 * 2 * np.pi
# the number of average excitations in the environment mode w0 at temperature w_th
Nth = n_thermal(w0, w_th)
# squeezing parameter for the environment
r = 1.0
theta = 0.1 * np.pi
N = Nth * (np.cosh(r)**2 + np.sinh(r)**2) + np.sinh(r)**2
print(f"N = {N}")
# H = - 0.5 * w0 * sigmaz()
hamiltonian = -0.5 * w0 * pauli.z(0)
sm = operators.annihilate(0)
sp = operators.create(0)
c_ops = [
    np.sqrt(gamma0) * (sm * np.cosh(r) + sp * np.sinh(r) * np.exp(1j * theta))
]
num_qubits = 1
dimensions = {0: 2}
# start in the qubit superposition state
psi0 = cp.array([2j, 1.0], dtype=cp.complex128)
psi0 = psi0 / cp.linalg.norm(psi0)
rho0_ = cp.outer(psi0, cp.conj(psi0).T)
rho0 = cudaq.State.from_data(cp.outer(psi0, cp.conj(psi0).T))
steps = np.linspace(0, 50, 1001)
schedule = Schedule(steps, ["time"])

evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.x(0),
                                       pauli.y(0),
                                       pauli.z(0)],
                          collapse_operators=c_ops,
                          store_intermediate_results=False,
                          integrator=ScipyZvodeIntegrator(nsteps=10))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))
plt.plot(steps, evolution_result.expect[0])
plt.plot(steps, evolution_result.expect[1])
plt.plot(steps, evolution_result.expect[2])
sz_ss_analytical = 1 / (2 * N + 1)
print(f"Squeezing parameter: {sz_ss_analytical}")
plt.plot(steps, sz_ss_analytical * np.ones(np.shape(steps)), 'k--')
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z", "Analytical"))
fig.savefig('example_4.png', dpi=fig.dpi)

# print(evolution_result.expect[2])
