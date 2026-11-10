# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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

_SECONDS_PER_NANOSECOND = 1.0e-9
_NANOSECONDS_PER_MICROSECOND = 1.0e3


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

    def drive_amplitude_scale(self, qubit_index: int) -> float:
        """Return the conversion from pulse amplitude to radians/ns.

        Targets may provide ``amplitude_scale_rad_per_ns`` explicitly. For
        transmon calibration records containing a Gaussian/DRAG pi pulse
        (``x_amp``, ``x_dur``, and ``x_sigma``), the scale is inferred from
        its truncated-Gaussian area. Targets without either representation use
        1.0, meaning pulse amplitudes are already angular rates in rad/ns.
        """
        params = self.get_drive_params(qubit_index)
        explicit = params.get("amplitude_scale_rad_per_ns")
        if explicit is not None:
            if explicit <= 0:
                raise ValueError(
                    f"amplitude_scale_rad_per_ns must be positive for qubit {qubit_index}"
                )
            return float(explicit)

        amplitude = float(params.get("x_amp", 0.0))
        duration = float(params.get("x_dur", 0.0))
        sigma = float(params.get("x_sigma", 0.0))
        if amplitude > 0.0 and duration > 0.0 and sigma > 0.0:
            area = (sigma * math.sqrt(2.0 * math.pi) *
                    math.erf(duration / (2.0 * math.sqrt(2.0) * sigma)))
            return math.pi / (amplitude * area)
        return 1.0

    def hamiltonian_terms(self) -> List[Dict[str, Any]]:
        """Generate two-level static and coupling Hamiltonian terms.

        Returns a list of term dicts compatible with ``OperatorTerm``.
        Each dict has keys: kind, qubit_indices, coefficient, time_dependent.
        Coefficients are angular frequencies in radians per nanosecond, matching
        the pulse IR time unit.

        ``anharmonicity_hz`` is calibration metadata for leakage models and is
        intentionally not emitted into this two-level spin Hamiltonian. A
        faithful anharmonicity term requires a three-or-more-level mode.
        """
        terms: List[Dict[str, Any]] = []

        for q in self.qubits.values():
            omega = q.frequency_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append({
                "kind": "static_z",
                "qubit_indices": (q.index,),
                "coefficient": complex(omega / 2.0, 0),
                "time_dependent": False,
            })
        for c in self.couplings:
            g = c.coupling_strength_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append({
                "kind": "coupling_xx",
                "qubit_indices": (c.qubit_a, c.qubit_b),
                "coefficient": complex(g, 0),
                "time_dependent": False,
            })

        for xt in self.crosstalk:
            zz = xt.static_zz_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
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
        and coefficient. Collapse-operator coefficients have units of inverse
        square-root nanoseconds, matching evolution time in nanoseconds.
        """
        terms: List[Dict[str, Any]] = []

        for q in self.qubits.values():
            if q.t1_us > 0:
                gamma1 = 1.0 / (q.t1_us * _NANOSECONDS_PER_MICROSECOND)
                terms.append({
                    "kind": "dissipator_t1",
                    "qubit_indices": (q.index,),
                    "coefficient": complex(math.sqrt(gamma1), 0),
                })
            if q.t2_star_us > 0:
                gamma_phi = 1.0 / (q.t2_star_us * _NANOSECONDS_PER_MICROSECOND)
                if q.t1_us > 0:
                    gamma_phi -= 1.0 / (2.0 * q.t1_us *
                                        _NANOSECONDS_PER_MICROSECOND)
                if gamma_phi > 0:
                    terms.append({
                        "kind": "dissipator_t2",
                        "qubit_indices": (q.index,),
                        "coefficient": complex(math.sqrt(gamma_phi / 2.0), 0),
                    })

        return terms
