import cudaq
from cudaq import spin, boson, Schedule, ScalarOperator, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrates simulation of an electrically-driven silicon spin qubit.
# The system dynamics is taken from https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.19.044078

dimensions = {0: 2}
resonance_frequency = 2 * np.pi * 10  # 10 Ghz

# Run the simulation:
evolution_results = []
get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
# Sweep the amplitude
amplitudes = np.linspace(0.0, 0.5, 20)
for amplitude in amplitudes:
    rho0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
    t_final = 100
    dt = 0.005
    n_steps = int(np.ceil(t_final / dt)) + 1
    steps = np.linspace(0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])
    # Electric dipole spin resonance (`EDSR`) Hamiltonian
    H = 0.5 * resonance_frequency * spin.z(0) + ScalarOperator(
        lambda t: 0.5 * np.sin(resonance_frequency * t) * amplitude) * spin.x(0)
    evolution_result = cudaq.evolve(H,
                                    dimensions,
                                    schedule,
                                    rho0,
                                    observables=[boson.number(0)],
                                    collapse_operators=[],
                                    store_intermediate_results=True,
                                    integrator=ScipyZvodeIntegrator())
    evolution_results.append(get_result(0, evolution_result))

fig, ax = plt.subplots()
im = ax.contourf(steps, amplitudes, evolution_results)
ax.set_xlabel("Time (ns)")
ax.set_ylabel(f"Amplitude (a.u.)")
fig.suptitle(f"Excited state probability")
fig.colorbar(im)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# For reference, see figure 5 in https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.19.044078
fig.savefig("spin_qubit_edsr.png", dpi=fig.dpi)
