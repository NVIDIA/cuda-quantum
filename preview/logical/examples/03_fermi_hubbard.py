# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate a Trotterized Fermi--Hubbard kernel with two target stacks."""

# %%
# Import CUDA-Q and the CUDA-Q Logical target and result APIs.
import cudaq
import cudaq.logical as cql


# %%
# Define helpers that construct the Pauli terms for one Trotter step.
def _pauli_word(num_qubits: int, operators: list[tuple[int, str]]) -> str:
    word = ["I"] * num_qubits
    for qubit, operator in operators:
        word[qubit] = operator
    return "".join(word)


def fermi_hubbard_step(
    sites: int,
    hopping: float,
    interaction: float,
    evolution_time: float,
    trotter_steps: int,
) -> tuple[list[float], list[str]]:
    num_qubits = 2 * sites
    dt = evolution_time / trotter_steps
    angles: list[float] = []
    paulis: list[str] = []

    for site in range(sites - 1):
        for spin in range(2):
            left = 2 * site + spin
            right = 2 * (site + 1) + spin
            for endpoint in ("X", "Y"):
                angles.append(hopping * dt / 2)
                paulis.append(
                    _pauli_word(num_qubits, [
                        (left, endpoint),
                        (left + 1, "Z"),
                        (right, endpoint),
                    ]))

    for site in range(sites):
        up = 2 * site
        down = up + 1
        angles.extend(
            (interaction * dt / 4, interaction * dt / 4, -interaction * dt / 4))
        paulis.extend((
            _pauli_word(num_qubits, [(up, "Z")]),
            _pauli_word(num_qubits, [(down, "Z")]),
            _pauli_word(num_qubits, [(up, "Z"), (down, "Z")]),
        ))
    return angles, paulis


# %%
# Author the Trotter evolution as an ordinary CUDA-Q kernel.
@cudaq.kernel
def fermi_hubbard_trotter(
    num_qubits: int,
    electron_count: int,
    angles: list[float],
    paulis: list[cudaq.pauli_word],
    trotter_steps: int,
):
    qubits = cudaq.qvector(num_qubits)
    for orbital in range(electron_count):
        x(qubits[orbital])
    for _ in range(trotter_steps):
        for term in range(len(angles)):
            exp_pauli(angles[term], qubits, paulis[term])
    mz(qubits)


# %%
# Prepare a modest four-site resource-estimation workload.
sites = 4
trotter_steps = 8
angles, paulis = fermi_hubbard_step(sites, 1.0, 4.0, 1.0, trotter_steps)
kernel_arguments = (2 * sites, 4, angles, paulis, trotter_steps)

# %%
# Estimate the workload directly as portable logical actions.
cudaq.set_target(cql.targets.estimator)
logical_estimate = cudaq.estimate(fermi_hubbard_trotter, *kernel_arguments)
logical_resources = cql.estimate.LogicalEstimate.from_annotations(
    logical_estimate.annotations)

# %%
# Estimate the same workload after synthesis with the Clifford+T target.
clifford_t_target = cql.targets.clifford_t_target(precision=1.0e-6)
cudaq.set_target(clifford_t_target)
clifford_t_estimate = cudaq.estimate(fermi_hubbard_trotter, *kernel_arguments)
clifford_t_resources = cql.estimate.LogicalEstimate.from_annotations(
    clifford_t_estimate.annotations)

# %%
# Print the two independently calculated resource summaries.
print("Fermi--Hubbard logical resources:")
print(f"  logical: depth={logical_resources.action_depth_upper_bound}, "
      f"actions={dict(logical_resources.actions)}")
print(f"  clifford_t: depth={clifford_t_resources.action_depth_upper_bound}, "
      f"actions={dict(clifford_t_resources.actions)}")
