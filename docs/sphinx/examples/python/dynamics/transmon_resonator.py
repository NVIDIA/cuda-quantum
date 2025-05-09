import cudaq
from cudaq import operators, spin, operators, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrates a simulation of a superconducting transmon qubit coupled to a resonator (i.e., cavity).
# References:
# - "Charge-insensitive qubit design derived from the Cooper pair box", PRA 76, 042319
# - QuTiP lecture: https://github.com/jrjohansson/qutip-lectures/blob/master/Lecture-10-cQED-dispersive-regime.ipynb

# Number of cavity photons
N = 20

# System dimensions: transmon + cavity
dimensions = {0: 2, 1: N}

# See III.B of PRA 76, 042319
# System parameters
# Unit: GHz
omega_01 = 3.0 * 2 * np.pi  # transmon qubit frequency
omega_r = 2.0 * 2 * np.pi  # resonator frequency
# Dispersive shift
chi_01 = 0.025 * 2 * np.pi
chi_12 = 0.0

omega_01_prime = omega_01 + chi_01
omega_r_prime = omega_r - chi_12 / 2.0
chi = chi_01 - chi_12 / 2.0

# System Hamiltonian
hamiltonian = 0.5 * omega_01_prime * spin.z(0) + (
    omega_r_prime + chi * spin.z(0)) * operators.number(1)

# Initial state of the system
# Transmon in a superposition state
transmon_state = cp.array([1. / np.sqrt(2.), 1. / np.sqrt(2.)],
                          dtype=cp.complex128)


# Helper to create a coherent state in Fock basis truncated at `num_levels`.
# Note: There are a couple of ways of generating a coherent state,
# e.g., see https://qutip.readthedocs.io/en/v5.0.3/apidoc/functions.html#qutip.core.states.coherent
# or https://en.wikipedia.org/wiki/Coherent_state
# Here, in this example, we use a the formula: `|alpha> = D(alpha)|0>`,
# i.e., apply the displacement operator on a zero (or vacuum) state to compute the corresponding coherent state.
def coherent_state(num_levels, amplitude):
    displace_mat = operators.displace(0).to_matrix({0: num_levels},
                                                   displacement=amplitude)
    # `D(alpha)|0>` is the first column of `D(alpha)` matrix
    return cp.array(np.transpose(displace_mat)[0])


# Cavity in a coherent state
cavity_state = coherent_state(N, 2.0)
psi0 = cudaq.State.from_data(cp.kron(transmon_state, cavity_state))

steps = np.linspace(0, 250, 1000)
schedule = Schedule(steps, ["time"])

# Evolve the system
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[
                                    operators.number(1),
                                    operators.number(0),
                                    operators.position(1),
                                    operators.position(0)
                                ],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
count_results = [
    get_result(0, evolution_result),
    get_result(1, evolution_result)
]

quadrature_results = [
    get_result(2, evolution_result),
    get_result(3, evolution_result)
]

fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, count_results[0])
plt.plot(steps, count_results[1])
plt.ylabel("n")
plt.xlabel("Time [ns]")
plt.legend(("Cavity Photon Number", "Transmon Excitation Probability"))
plt.title("Excitation Numbers")

plt.subplot(1, 2, 2)
plt.plot(steps, quadrature_results[0])
plt.ylabel("x")
plt.xlabel("Time [ns]")
plt.legend(("cavity"))
plt.title("Resonator Quadrature")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('transmon_resonator.png', dpi=fig.dpi)
