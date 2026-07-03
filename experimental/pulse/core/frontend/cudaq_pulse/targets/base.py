# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture-neutral target API for QPU Hamiltonians and decoherence models.

A ``Target`` fully describes a quantum device: qubit frequencies,
anharmonicities, coupling topology, decoherence parameters, and readout.
It can generate Hamiltonian and Lindblad dissipator terms for the
``pulse_to_operator`` lowering pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Qubit:
    """Single qubit in a target device."""

    index: int
    frequency_hz: float
    anharmonicity_hz: float
    t1_us: float
    t2_star_us: float
    label: str = ""
    drive_params: Dict[str, float] = field(default_factory=dict)
    readout_params: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Coupling:
    """Coupling edge between two qubits."""

    qubit_a: int
    qubit_b: int
    coupling_strength_hz: float
    gate_type: str = "cz"
    gate_duration_ns: float = 98.0
    gate_buffer_ns: float = 15.0
    gate_fidelity: float = 0.985
    gate_params: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CrosstalkEntry:
    """Residual ZZ or other parasitic coupling between qubit pairs."""

    qubit_a: int
    qubit_b: int
    zz_coupling: float
    static_zz_hz: float
    freq_delta_hz: float


@dataclass()
class Target:
    """Full description of a quantum processing unit."""

    name: str
    qubits: Dict[int, Qubit] = field(default_factory=dict)
    couplings: List[Coupling] = field(default_factory=list)
    crosstalk: List[CrosstalkEntry] = field(default_factory=list)
    architecture: str = "transmon"
    attribution: str = ""

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def qubit_indices(self) -> List[int]:
        return sorted(self.qubits.keys())

    @property
    def frequencies(self) -> Dict[int, float]:
        return {q.index: q.frequency_hz for q in self.qubits.values()}

    @property
    def anharmonicities(self) -> Dict[int, float]:
        return {q.index: q.anharmonicity_hz for q in self.qubits.values()}

    @property
    def t1_times(self) -> Dict[int, float]:
        """T1 in microseconds, keyed by qubit index."""
        return {q.index: q.t1_us for q in self.qubits.values()}

    @property
    def t2_times(self) -> Dict[int, float]:
        """T2* in microseconds, keyed by qubit index."""
        return {q.index: q.t2_star_us for q in self.qubits.values()}

    @property
    def coupling_map(self) -> List[Tuple[int, int]]:
        return [(c.qubit_a, c.qubit_b) for c in self.couplings]

    def connectivity_graph(self) -> Dict[int, List[int]]:
        """Adjacency list representation of qubit connectivity."""
        g: Dict[int, List[int]] = {idx: [] for idx in self.qubits}
        for c in self.couplings:
            g.setdefault(c.qubit_a, []).append(c.qubit_b)
            g.setdefault(c.qubit_b, []).append(c.qubit_a)
        return g

    def get_drive_params(self, qubit_index: int) -> Dict[str, float]:
        """Per-qubit drive parameters (amp, sigma, beta, etc.)."""
        if qubit_index not in self.qubits:
            raise KeyError(f"Qubit {qubit_index} not in target {self.name!r}")
        return dict(self.qubits[qubit_index].drive_params)

    def hamiltonian_terms(self) -> List[Dict[str, Any]]:
        """Generate static + coupling Hamiltonian terms.

        Returns a list of term dicts compatible with ``OperatorTerm``.
        Each dict has keys: kind, qubit_indices, coefficient, time_dependent.
        """
        terms: List[Dict[str, Any]] = []

        for q in self.qubits.values():
            omega = q.frequency_hz * 2.0 * math.pi
            terms.append({
                "kind": "static_z",
                "qubit_indices": (q.index,),
                "coefficient": complex(omega, 0),
                "time_dependent": False,
            })
            if q.anharmonicity_hz != 0:
                alpha = q.anharmonicity_hz * 2.0 * math.pi
                terms.append({
                    "kind": "anharmonicity",
                    "qubit_indices": (q.index,),
                    "coefficient": complex(alpha / 2.0, 0),
                    "time_dependent": False,
                })

        for c in self.couplings:
            g = c.coupling_strength_hz * 2.0 * math.pi
            terms.append({
                "kind": "coupling_xx",
                "qubit_indices": (c.qubit_a, c.qubit_b),
                "coefficient": complex(g, 0),
                "time_dependent": False,
            })

        for xt in self.crosstalk:
            zz = xt.static_zz_hz * 2.0 * math.pi
            terms.append({
                "kind": "crosstalk_zz",
                "qubit_indices": (xt.qubit_a, xt.qubit_b),
                "coefficient": complex(zz, 0),
                "time_dependent": False,
            })

        return terms

    def dissipator_terms(self) -> List[Dict[str, Any]]:
        """Generate T1 / T2 Lindblad dissipator terms.

        Returns a list of term dicts, each with kind, qubit_indices,
        and coefficient (the collapse operator rate).
        """
        terms: List[Dict[str, Any]] = []

        for q in self.qubits.values():
            if q.t1_us > 0:
                gamma1 = 1.0 / (q.t1_us * 1e-6)
                terms.append({
                    "kind": "dissipator_t1",
                    "qubit_indices": (q.index,),
                    "coefficient": complex(math.sqrt(gamma1), 0),
                })
            if q.t2_star_us > 0:
                gamma_phi = (1.0 / (q.t2_star_us * 1e-6) - 1.0 /
                             (2.0 * q.t1_us * 1e-6))
                if gamma_phi > 0:
                    terms.append({
                        "kind": "dissipator_t2",
                        "qubit_indices": (q.index,),
                        "coefficient": complex(math.sqrt(gamma_phi), 0),
                    })

        return terms
