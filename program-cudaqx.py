# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq, cudaq_solvers as solvers

from scipy.optimize import minimize


def test_solvers_uccsd():

    # fail
    cudaq.set_target('ionq', emulate="true")
    # cudaq.set_target('quantinuum', emulate="true")

    # pass
    # cudaq.set_target('nvidia')
    # cudaq.set_target('nvidia-fp64')
    # cudaq.set_target('qpp-cpu')
    # cudaq.set_target("remote-mqpu", auto_launch="1")

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons
    spin = 0
    print(numQubits)
    print(numElectrons)

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            rx(np.pi / 4, q[i])
        #solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()

    energy, params, all_data = solvers.vqe(ansatz,
                                           molecule.hamiltonian,
                                           [-.2, -.2, -.2],
                                           optimizer=minimize,
                                           method='L-BFGS-B',
                                           jac='3-point',
                                           tol=1e-4,
                                           options={'disp': True},
                                           shots=1000,
                                           verbose=False)
    print(energy)
    
    assert np.isclose(energy, 0.35, 1e-1)

# test_solvers_uccsd()

def test_uccsd():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    repro_num_qubits = molecule.n_orbitals * 2
    repro_num_electrons = molecule.n_electrons
    spin = 0


    repro_num_electrons = 2
    repro_num_qubits = 8

    print(repro_num_qubits)
    print(repro_num_electrons)

    # # should be 3 thetas
    repro_thetas = [-0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558]
    # #repro_thetas = [-.2, -.2, -.2]

    @cudaq.kernel
    def repro_trial_state(qubits: cudaq.qvector, num_electrons:int, thetas:list[float]):
        for i in range(num_electrons):
            x(qubits[i])
        solvers.stateprep.uccsd(qubits, thetas, num_electrons, 0)

    @cudaq.kernel
    def repro():
        repro_qubits = cudaq.qvector(repro_num_qubits)
        repro_trial_state(repro_qubits, repro_num_electrons, repro_thetas)

    counts = cudaq.sample(repro, shots_count=1000)
    print(counts)


cudaq.set_target('nvidia', option='fp64')
print('nvidia')
test_uccsd()

print('ionq')
cudaq.set_target('ionq', emulate="true")
test_uccsd()
