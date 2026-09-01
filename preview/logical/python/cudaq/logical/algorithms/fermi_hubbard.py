# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical Jordan--Wigner workloads for square-lattice Fermi--Hubbard models.

This module owns only P0 algorithm facts.  It exposes the logical program and
the canonical signed rotation classes implied by its first-order Trotterization;
synthesis policy, QEC architecture, and factories belong to later compiler
stages and to the selected device.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..algebra.pauli import X, Y, Z
from ..ops._impl import allocate, discard, h, rotate, x
from ..programs.decorators import program
from ..programs.definition import ProgramDefinition
from ..types.semantic import zero


@dataclass(frozen=True, slots=True)
class FermiHubbardStats:
    """Closed-form logical structure of one lattice instance."""

    lattice_size: int
    periodic: bool
    mode_qubits: int
    phase_ancillas: int
    logical_qubits: int
    prepared_particles: int
    nearest_neighbor_edges: int
    hopping_pauli_terms_per_step: int
    onsite_pauli_terms_per_step: int
    controlled_rotations_per_step: int
    controlled_global_phases_per_step: int
    logical_rotations_per_step: int


@dataclass(frozen=True, slots=True)
class FermiHubbardRotationClass:
    """One canonical signed rotation angle and its exact multiplicity."""

    name: str
    angle: float
    multiplicity: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("rotation-class name must be nonempty")
        if isinstance(self.angle,
                      bool) or not isinstance(self.angle, (int, float)):
            raise TypeError("rotation-class angle must be a finite real number")
        if not math.isfinite(float(self.angle)):
            raise ValueError(
                "rotation-class angle must be a finite real number")
        if (not isinstance(self.multiplicity, int) or
                isinstance(self.multiplicity, bool)):
            raise TypeError("rotation-class multiplicity must be an integer")
        if self.multiplicity < 0:
            raise ValueError("rotation-class multiplicity must be nonnegative")
        object.__setattr__(self, "angle", float(self.angle))


@dataclass(frozen=True, slots=True, init=False)
class FermiHubbardWorkload:
    """A factory-controlled P0 program plus compiler-independent facts.

    Instances can only be returned by :func:`fermi_hubbard_jordan_wigner`.
    Keeping the bundle factory-controlled guarantees that its rotation classes
    and statistics describe the same specialized :class:`ProgramDefinition`;
    use a fresh factory call instead of ``dataclasses.replace`` to change it.
    """

    definition: ProgramDefinition
    stats: FermiHubbardStats
    trotter_steps: int
    evolution_time: float
    hopping: float
    interaction: float
    rotation_classes: tuple[FermiHubbardRotationClass, ...]

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise TypeError("FermiHubbardWorkload is factory-controlled; use "
                        "fermi_hubbard_jordan_wigner()")

    @property
    def logical_qubits(self) -> int:
        return self.stats.logical_qubits

    @property
    def logical_rotations(self) -> int:
        return sum(item.multiplicity for item in self.rotation_classes)

    def materialize(self):
        """Compile the specialized logical program through the P0 pipeline."""

        from ..compiler import compile

        return compile(self.definition)


def _positive_int(name: str, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive int")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _finite_real(name: str, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _rotation_classes(
    stats: FermiHubbardStats,
    *,
    trotter_steps: int,
    evolution_time: float,
    hopping: float,
    interaction: float,
) -> tuple[FermiHubbardRotationClass, ...]:
    """Derive the only valid class summary for one specialized workload."""

    dt = evolution_time / trotter_steps
    sites = stats.lattice_size * stats.lattice_size
    return (
        FermiHubbardRotationClass(
            "hopping_positive",
            0.5 * hopping * dt,
            stats.hopping_pauli_terms_per_step * trotter_steps,
        ),
        FermiHubbardRotationClass(
            "hopping_negative",
            -0.5 * hopping * dt,
            stats.hopping_pauli_terms_per_step * trotter_steps,
        ),
        FermiHubbardRotationClass(
            "interaction_positive",
            0.25 * interaction * dt,
            3 * sites * trotter_steps,
        ),
        FermiHubbardRotationClass(
            "interaction_negative",
            -0.25 * interaction * dt,
            4 * sites * trotter_steps,
        ),
    )


def _workload_bundle(
    *,
    definition: ProgramDefinition,
    stats: FermiHubbardStats,
    trotter_steps: int,
    evolution_time: float,
    hopping: float,
    interaction: float,
) -> FermiHubbardWorkload:
    """Create a provenance-safe bundle from facts derived in this module."""

    if not isinstance(definition, ProgramDefinition):
        raise TypeError("definition must be a ProgramDefinition")
    canonical_stats = fermi_hubbard_stats(
        stats.lattice_size,
        periodic=stats.periodic,
    )
    if stats != canonical_stats:
        raise ValueError("Fermi--Hubbard statistics are not canonical")
    trotter_steps = _positive_int("trotter_steps", trotter_steps)
    evolution_time = _finite_real("evolution_time", evolution_time)
    hopping = _finite_real("hopping", hopping)
    interaction = _finite_real("interaction", interaction)
    rotation_classes = _rotation_classes(
        canonical_stats,
        trotter_steps=trotter_steps,
        evolution_time=evolution_time,
        hopping=hopping,
        interaction=interaction,
    )
    if sum(item.multiplicity for item in rotation_classes) != (
            canonical_stats.logical_rotations_per_step * trotter_steps):
        raise RuntimeError("internal Fermi--Hubbard rotation count mismatch")

    workload = object.__new__(FermiHubbardWorkload)
    for name, value in (
        ("definition", definition),
        ("stats", canonical_stats),
        ("trotter_steps", trotter_steps),
        ("evolution_time", evolution_time),
        ("hopping", hopping),
        ("interaction", interaction),
        ("rotation_classes", rotation_classes),
    ):
        object.__setattr__(workload, name, value)
    return workload


def _spin_orbital(lattice_size: int, x: int, y: int, spin: int) -> int:
    return spin * lattice_size * lattice_size + y * lattice_size + x


def _onsite_pair(lattice_size: int, x: int, y: int) -> tuple[int, int]:
    return (
        _spin_orbital(lattice_size, x, y, 0),
        _spin_orbital(lattice_size, x, y, 1),
    )


def _nearest_neighbor_edges(
    lattice_size: int,
    *,
    periodic: bool,
) -> tuple[tuple[int, int], ...]:
    # Keep a bond list rather than a simple-edge set.  On a periodic L=2
    # lattice the +x and -x (likewise +y and -y) bonds connect the same pair of
    # sites but are distinct Hamiltonian terms.
    edges: list[tuple[int, int]] = []
    for y_coordinate in range(lattice_size):
        for x_coordinate in range(lattice_size):
            site = y_coordinate * lattice_size + x_coordinate
            if x_coordinate + 1 < lattice_size:
                edges.append(tuple(sorted((site, site + 1))))
            elif periodic and lattice_size > 1:
                edges.append(tuple(sorted((site, y_coordinate * lattice_size))))
            if y_coordinate + 1 < lattice_size:
                edges.append(tuple(sorted((site, site + lattice_size))))
            elif periodic and lattice_size > 1:
                edges.append(tuple(sorted((site, x_coordinate))))
    return tuple(sorted(edges))


def fermi_hubbard_stats(
    lattice_size: int,
    *,
    periodic: bool = True,
) -> FermiHubbardStats:
    """Return exact logical counts for the square-lattice encoding."""

    lattice_size = _positive_int("lattice_size", lattice_size)
    if not isinstance(periodic, bool):
        raise TypeError("periodic must be a bool")
    sites = lattice_size * lattice_size
    edges = _nearest_neighbor_edges(lattice_size, periodic=periodic)
    hopping_terms = 4 * len(edges)
    onsite_terms = 3 * sites
    return FermiHubbardStats(
        lattice_size=lattice_size,
        periodic=periodic,
        mode_qubits=2 * sites,
        phase_ancillas=1,
        logical_qubits=2 * sites + 1,
        prepared_particles=sites,
        nearest_neighbor_edges=len(edges),
        hopping_pauli_terms_per_step=hopping_terms,
        onsite_pauli_terms_per_step=onsite_terms,
        controlled_rotations_per_step=2 * (hopping_terms + onsite_terms),
        controlled_global_phases_per_step=sites,
        logical_rotations_per_step=2 * (hopping_terms + onsite_terms) + sites,
    )


def _rotated(values, paulis: str, angle: float):
    values = tuple(values)
    product = None
    constructors = {"X": X, "Y": Y, "Z": Z}
    for value, pauli in zip(values, paulis):
        factor = constructors[pauli](value)
        product = factor if product is None else product @ factor
    if product is None:
        raise ValueError("rotation product requires at least one operand")
    successors = rotate(product, angle=angle)
    by_input = {id(old): new for old, new in zip(product.operands, successors)}
    return tuple(by_input[id(value)] for value in values)


def _rotate_modes(modes, indices, paulis: str, angle: float) -> None:
    values = _rotated((modes[index] for index in indices), paulis, angle)
    for index, value in zip(indices, values):
        modes[index] = value


def _controlled_rotate_modes(
    modes,
    phase,
    indices,
    paulis: str,
    angle: float,
):
    # rotate(theta) = exp(-i theta P / 2).  A controlled instance is the
    # product exp(-i theta P / 4) exp(+i theta Zc P / 4).
    _rotate_modes(modes, indices, paulis, 0.5 * angle)
    combined = _rotated(
        (phase, *(modes[index] for index in indices)),
        "Z" + paulis,
        -0.5 * angle,
    )
    phase = combined[0]
    for index, value in zip(indices, combined[1:]):
        modes[index] = value
    return phase


def _jw_string(p: int, q: int, endpoint: str):
    low, high = sorted((p, q))
    return (
        tuple(range(low, high + 1)),
        endpoint + "Z" * (high - low - 1) + endpoint,
    )


def _trotter_step(
    modes,
    phase,
    *,
    lattice_size: int,
    hopping: float,
    interaction: float,
    dt: float,
    periodic: bool,
):
    for left_site, right_site in _nearest_neighbor_edges(lattice_size,
                                                         periodic=periodic):
        left_x, left_y = left_site % lattice_size, left_site // lattice_size
        right_x, right_y = right_site % lattice_size, right_site // lattice_size
        for spin in (0, 1):
            left = _spin_orbital(lattice_size, left_x, left_y, spin)
            right = _spin_orbital(lattice_size, right_x, right_y, spin)
            for endpoint in ("X", "Y"):
                indices, paulis = _jw_string(left, right, endpoint)
                phase = _controlled_rotate_modes(
                    modes,
                    phase,
                    indices,
                    paulis,
                    hopping * dt,
                )

    # n_up n_down = (I - Z_up - Z_down + Z_up Z_down) / 4.  The identity
    # contribution remains a relative phase because evolution is probe-led.
    for y_coordinate in range(lattice_size):
        for x_coordinate in range(lattice_size):
            up, down = _onsite_pair(lattice_size, x_coordinate, y_coordinate)
            (phase,) = _rotated((phase,), "Z", -0.25 * interaction * dt)
            phase = _controlled_rotate_modes(modes, phase, (up,), "Z",
                                             -0.5 * interaction * dt)
            phase = _controlled_rotate_modes(modes, phase, (down,), "Z",
                                             -0.5 * interaction * dt)
            phase = _controlled_rotate_modes(modes, phase, (up, down), "ZZ",
                                             0.5 * interaction * dt)
    return phase


def fermi_hubbard_jordan_wigner(
    lattice_size: int,
    *,
    trotter_steps: int = 1,
    evolution_time: float = 1.0,
    hopping: float = 1.0,
    interaction: float = 4.0,
    periodic: bool = True,
) -> FermiHubbardWorkload:
    """Construct a specialized checkerboard-state controlled evolution.

    The returned rotation classes are derived from the same decomposition as
    the program.  They let compiler and resource examples compile each unique
    signed angle once and scale by exact P0 multiplicity without restating the
    Hamiltonian decomposition.
    """

    stats = fermi_hubbard_stats(lattice_size, periodic=periodic)
    trotter_steps = _positive_int("trotter_steps", trotter_steps)
    evolution_time = _finite_real("evolution_time", evolution_time)
    hopping = _finite_real("hopping", hopping)
    interaction = _finite_real("interaction", interaction)
    dt = evolution_time / trotter_steps

    def evolution() -> None:
        all_qubits = allocate(stats.logical_qubits,
                              state=zero,
                              name="fermi_hubbard")
        modes = list(all_qubits[:stats.mode_qubits])
        phase = h(all_qubits[stats.mode_qubits])

        for y_coordinate in range(lattice_size):
            for x_coordinate in range(lattice_size):
                up, down = _onsite_pair(lattice_size, x_coordinate,
                                        y_coordinate)
                occupied = up if (x_coordinate +
                                  y_coordinate) % 2 == 0 else down
                modes[occupied] = x(modes[occupied])

        # Straight-line specialization keeps every P0 rotation visible to
        # GridSynth and PBC instead of asking those passes to reinterpret a
        # folded loop.
        for _ in range(trotter_steps):
            phase = _trotter_step(
                modes,
                phase,
                lattice_size=lattice_size,
                hopping=hopping,
                interaction=interaction,
                dt=dt,
                periodic=periodic,
            )
        discard((*modes, phase))

    definition = program(
        evolution,
        name=f"fermi_hubbard_jw_l{lattice_size}_r{trotter_steps}",
    )
    return _workload_bundle(
        definition=definition,
        stats=stats,
        trotter_steps=trotter_steps,
        evolution_time=evolution_time,
        hopping=hopping,
        interaction=interaction,
    )


__all__ = [
    "FermiHubbardRotationClass",
    "FermiHubbardStats",
    "FermiHubbardWorkload",
    "fermi_hubbard_jordan_wigner",
    "fermi_hubbard_stats",
]
