# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

cudaq.set_target("quantinuum", emulate=True)

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)

qubit_count = data.n_orbitals * 2

@cudaq.kernel
def kernel(thetas: list[float]):
    qubits = cudaq.qvector(qubit_count)

result = cudaq.observe(kernel, molecule, [.0,.0,.0,.0], shots_count = 1000)

expectation = result.expectation()
print(expectation)
