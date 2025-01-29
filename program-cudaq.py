# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq, cudaq_solvers as solvers

from scipy.optimize import minimize


def test_solvers_uccsd():
    # fail
    # how to find the breaking opt
    # BaseRestRemoteClient.h
    # currently reads the pipeline, performs opts, then converts to base qir profile
    # then converts to llvm (from qir?) to run on a simulator
    
    cudaq.set_target('ionq', emulate="true")
    #cudaq.set_target('quantinuum', emulate="true")

    # pass
    #cudaq.set_target('nvidia')
    #cudaq.set_target('nvidia-fp64')
    #cudaq.set_target('qpp-cpu')

    # pass
    # this one is doing synthesis just like quantum devices
    #cudaq.set_target("remote-mqpu", auto_launch="1")
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
            x(q[i])
        #solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()
    #print(ansatz)

    energy, params, all_data = solvers.vqe(ansatz,
                                           molecule.hamiltonian,
                                           [-.2, -.2, -.2],
                                           optimizer=minimize,
                                           method='L-BFGS-B',
                                           jac='3-point',
                                           tol=1e-4,
                                           options={'disp': True},
                                           shots=1000,
                                           verbose=True)
    print(energy)
    #print(params)
    #print(all_data[0].parameters)
    
    assert np.isclose(energy, -1.04, 1e-2)

#test_solvers_uccsd()

def test_cudaq_uccsd():
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
                    -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558]

    @cudaq.kernel
    def repro_trial_state(qubits: cudaq.qvector, num_electrons:int, num_qubits:int, thetas:list[float]):
        for i in range(num_electrons):
            x(qubits[i])
        uccsd(qubits, thetas, num_electrons, num_qubits)

    @cudaq.kernel
    def repro():
        repro_qubits = cudaq.qvector(repro_num_qubits)
        repro_trial_state(repro_qubits, repro_num_electrons, repro_num_qubits, repro_thetas)

    counts = cudaq.sample(repro, shots_count=1000)
    print(counts)


cudaq.set_target('nvidia', option='fp64')
print('nvidia')
test_cudaq_uccsd()

print('ionq')
cudaq.set_target('ionq', emulate="true")
test_cudaq_uccsd()
