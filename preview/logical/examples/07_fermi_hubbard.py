# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#%%
"""Estimate a Trotterized Fermi-Hubbard evolution with CUDA-Q Logical."""

# Build the Fermi-Hubbard Trotter kernel and its runtime parameters.
import cudaq
import cudaq.logical
from cudaq.logical.estimate import LogicalEstimate


def print_logical_resources(label: str, estimate):
    """Print a compact, human-readable logical resource summary."""
    resources = LogicalEstimate.from_annotations(estimate.annotations)
    print(f"\n{label}")
    print("-" * len(label))
    print("Action-depth upper bound: "
          f"{resources.action_depth_upper_bound} logical layers")
    print("Logical actions:")
    for action, count in sorted(resources.actions.items()):
        print(f"  {action:<32} {count:>12,}")


def _pauli_word(num_qubits: int, operators: list[tuple[int, str]]) -> str:
    """Build a full-width CUDA-Q Pauli word in qubit-index order."""
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
    """Return one Trotter step for ``H = -hopping*T + interaction*V``.

    CUDA-Q applies ``exp_pauli(angle, ..., P) = exp(i * angle * P)`` and
    converts the returned strings to the kernel's ``cudaq.pauli_word`` inputs.
    """
    num_qubits = 2 * sites
    dt = evolution_time / trotter_steps
    angles: list[float] = []
    paulis: list[str] = []

    # Interleaved Jordan-Wigner orbitals make nearest-site hopping XZX + YZY.
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

    # U n_up n_down contributes Z_up, Z_down, and ZZ; its identity is a phase.
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


SITES = 4
HOPPING = 1.0
INTERACTION = 4.0
EVOLUTION_TIME = 1.0
ELECTRONS = 4
TROTTER_STEPS = 8

angles, paulis = fermi_hubbard_step(SITES, HOPPING, INTERACTION, EVOLUTION_TIME,
                                    TROTTER_STEPS)
kernel_args = (2 * SITES, ELECTRONS, angles, paulis, TROTTER_STEPS)

#%%
# Select the logical estimator target.
from cudaq.logical.targets import estimator

cudaq.set_target(estimator)

#%%
# Estimate the workload in the logical action model.
estimate = cudaq.estimate(
    fermi_hubbard_trotter,
    *kernel_args,
)
print_logical_resources("Logical action estimate", estimate)

#%%
# Select the Clifford+T synthesis target.
from cudaq.logical.targets import clifford_t

cudaq.set_target(clifford_t)

#%%
# Estimate the workload after Clifford+T decomposition.
estimate = cudaq.estimate(
    fermi_hubbard_trotter,
    *kernel_args,
)
print_logical_resources("Clifford+T action estimate", estimate)
