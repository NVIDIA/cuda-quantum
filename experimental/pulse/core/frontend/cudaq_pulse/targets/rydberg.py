# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neutral atom / Rydberg QPU target definitions.

Models the Rydberg many-body Hamiltonian:

    H/hbar = sum_j Omega_j(t)/2 (e^{i phi_j} |g><r| + h.c.)
           - sum_j Delta_j(t) n_j
           + sum_{j<k} C6 / |x_j - x_k|^6  n_j n_k

Default C6 coefficient corresponds to Rb-87 |70S_{1/2}> Rydberg state:
    C6 = 862690 * 2pi MHz um^6

Parameters are based on standard Rb-87 Rydberg physics from the published
neutral-atom literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Coupling, Qubit, Target

# Rb-87, |70S_1/2> Rydberg state
_DEFAULT_C6_MHZ_UM6 = 862690.0  # 2pi * MHz * um^6


@dataclass(frozen=True)
class RydbergAtom:
    """A single atom in a Rydberg array."""

    index: int
    position: Tuple[float, float]  # (x, y) in micrometers


class RydbergTarget:
    """Neutral atom target using Rydberg interactions.

    Parameters
    ----------
    atoms : list[RydbergAtom]
        Atom positions in um.
    c6 : float
        Van der Waals coefficient in MHz * um^6 (includes 2pi).
    global_rabi_mhz : float
        Global Rabi frequency in MHz.
    global_detuning_mhz : float
        Global detuning in MHz.
    """

    def __init__(
        self,
        atoms: List[RydbergAtom],
        c6: float = _DEFAULT_C6_MHZ_UM6,
        global_rabi_mhz: float = 4.0,
        global_detuning_mhz: float = 0.0,
    ):
        if not atoms:
            raise ValueError("Must provide at least one atom.")
        self.atoms = sorted(atoms, key=lambda a: a.index)
        self.c6 = c6
        self.global_rabi_mhz = global_rabi_mhz
        self.global_detuning_mhz = global_detuning_mhz

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def blockade_radius(self) -> float:
        """Rydberg blockade radius in um: R_b = (C6 / Omega)^{1/6}."""
        if self.global_rabi_mhz <= 0:
            raise ValueError(
                "global_rabi_mhz must be > 0 to compute blockade radius.")
        return (self.c6 / self.global_rabi_mhz)**(1.0 / 6.0)

    def _distance(self, a: RydbergAtom, b: RydbergAtom) -> float:
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def interaction_strength(self, a: RydbergAtom, b: RydbergAtom) -> float:
        """V_{jk} = C6 / |x_j - x_k|^6 in MHz."""
        r = self._distance(a, b)
        if r < 1e-12:
            raise ValueError(
                f"Atoms {a.index} and {b.index} are at the same position.")
        return self.c6 / (r**6)

    def to_target(self) -> Target:
        """Convert to a generic Target for the compilation pipeline."""
        qubits: Dict[int, Qubit] = {}
        for atom in self.atoms:
            qubits[atom.index] = Qubit(
                index=atom.index,
                frequency_hz=self.global_rabi_mhz * 1e6,
                anharmonicity_hz=0.0,
                t1_us=1000.0,  # Rydberg T1 ~ ms scale
                t2_star_us=100.0,
                label=f"atom_{atom.index}",
            )

        couplings: List[Coupling] = []
        for i, ai in enumerate(self.atoms):
            for aj in self.atoms[i + 1:]:
                v = self.interaction_strength(ai, aj)
                couplings.append(
                    Coupling(
                        qubit_a=ai.index,
                        qubit_b=aj.index,
                        coupling_strength_hz=v * 1e6,
                        gate_type="rydberg_blockade",
                        gate_duration_ns=0.0,
                        gate_fidelity=1.0,
                    ))

        return Target(
            name="rydberg_array",
            qubits=qubits,
            couplings=couplings,
            architecture="neutral_atom",
            attribution=
            ("Rydberg Hamiltonian parameterization based on Rb-87 |70S_1/2> "
             "state. C6 coefficient from the published neutral-atom literature."
            ),
        )

    def hamiltonian_terms(self) -> List[Dict[str, Any]]:
        """Generate Rydberg Hamiltonian terms.

        Terms:
          - Rabi drive: Omega_j/2 * sigma_x_j (time-dependent)
          - Detuning: -Delta_j * n_j
          - Interaction: C6/r^6 * n_j * n_k
        """
        terms: List[Dict[str, Any]] = []

        for atom in self.atoms:
            rabi_rad = self.global_rabi_mhz * 1e6 * 2.0 * math.pi
            terms.append({
                "kind": "rabi_drive",
                "qubit_indices": (atom.index,),
                "coefficient": complex(rabi_rad / 2.0, 0),
                "time_dependent": True,
            })

            if self.global_detuning_mhz != 0:
                delta_rad = self.global_detuning_mhz * 1e6 * 2.0 * math.pi
                terms.append({
                    "kind": "detuning",
                    "qubit_indices": (atom.index,),
                    "coefficient": complex(-delta_rad, 0),
                    "time_dependent": True,
                })

        for i, ai in enumerate(self.atoms):
            for aj in self.atoms[i + 1:]:
                v_mhz = self.interaction_strength(ai, aj)
                v_rad = v_mhz * 1e6 * 2.0 * math.pi
                terms.append({
                    "kind": "rydberg_interaction",
                    "qubit_indices": (ai.index, aj.index),
                    "coefficient": complex(v_rad, 0),
                    "time_dependent": False,
                })

        return terms

    def dissipator_terms(self) -> List[Dict[str, Any]]:
        """Rydberg dissipators: spontaneous emission from |r> to |g>."""
        terms: List[Dict[str, Any]] = []
        for atom in self.atoms:
            gamma = 1.0 / (1000.0 * 1e-6)  # ~1/ms
            terms.append({
                "kind": "dissipator_spontaneous",
                "qubit_indices": (atom.index,),
                "coefficient": complex(math.sqrt(gamma), 0),
            })
        return terms


def rydberg_chain(
    n: int,
    spacing_um: float = 6.0,
    c6: float = _DEFAULT_C6_MHZ_UM6,
    global_rabi_mhz: float = 4.0,
    global_detuning_mhz: float = 0.0,
) -> RydbergTarget:
    """1D chain of n atoms with uniform spacing.

    Parameters
    ----------
    n : int
        Number of atoms.
    spacing_um : float
        Inter-atom spacing in micrometers.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    atoms = [
        RydbergAtom(index=i, position=(i * spacing_um, 0.0)) for i in range(n)
    ]
    return RydbergTarget(atoms,
                         c6=c6,
                         global_rabi_mhz=global_rabi_mhz,
                         global_detuning_mhz=global_detuning_mhz)


def rydberg_square(
    rows: int,
    cols: int,
    spacing_um: float = 6.0,
    c6: float = _DEFAULT_C6_MHZ_UM6,
    global_rabi_mhz: float = 4.0,
    global_detuning_mhz: float = 0.0,
) -> RydbergTarget:
    """2D square lattice of atoms.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    spacing_um : float
        Lattice spacing in micrometers.
    """
    if rows < 1 or cols < 1:
        raise ValueError(
            f"rows and cols must be >= 1, got rows={rows}, cols={cols}")
    atoms = []
    for r in range(rows):
        for c in range(cols):
            atoms.append(
                RydbergAtom(
                    index=r * cols + c,
                    position=(c * spacing_um, r * spacing_um),
                ))
    return RydbergTarget(atoms,
                         c6=c6,
                         global_rabi_mhz=global_rabi_mhz,
                         global_detuning_mhz=global_detuning_mhz)
