# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Folded logical workload for Gidney--Ekerå windowed RSA factoring.

The library keeps billion-operation multiplicities in nested ``qlx.repeat``
regions. It is intentionally a logical workload and does not materialize a
factory device.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from ..programs.decorators import program
from ..ops._impl import (
    allocate,
    ccz,
    discard,
    repeat,
)
from ..programs.definition import ProgramDefinition

if TYPE_CHECKING:
    from ..estimate import LogicalProfile


def exponent_length(n_bits: int) -> int:
    """Ekerå--Håstad short discrete-logarithm exponent length ``ceil(3n/2)``."""

    return (3 * n_bits + 1) // 2


def lookup_additions_per_modmul(n_bits: int,
                                c_exp: int,
                                c_mul: int,
                                c_sep: int = 1024) -> int:
    """Equation (1) lookup-addition multiplicity per modular multiply."""

    numerator = 2 * n_bits * (c_sep + 1)
    denominator = c_exp * c_mul * c_sep
    return max(1, -(-numerator // denominator))


def c_pad_for(n_bits: int, n_e: int, delta_off: int = 4) -> int:
    """Coset-register padding from Section 2.9 of arXiv:1905.09749."""

    return math.ceil(2 * math.log2(max(n_bits, 2)) +
                     math.log2(max(n_e, 2))) + delta_off


def toffolis_per_lookup_addition(width: int, c_exp: int, c_mul: int) -> int:
    """Exact CCZ/Toffoli count of one folded lookup-addition body."""

    address = c_exp + c_mul
    return (2 * width + (1 << address) - 1 + (1 << ((address + 1) // 2)) - 1)


@dataclass(frozen=True, slots=True)
class GidneyEkeraProgram:
    definition: ProgramDefinition
    n_bits: int
    c_exp: int
    c_mul: int
    c_sep: int
    n_e: int
    c_pad: int
    work_qubits: int
    lookup_additions: int
    toffolis_per_lookup_addition: int

    @property
    def total_toffolis(self) -> int:
        return (self.n_e * self.lookup_additions *
                self.toffolis_per_lookup_addition)

    def materialize(self):
        from ..compiler import compile

        return compile(self.definition)


@dataclass(frozen=True, slots=True)
class GidneyEkeraResourceModel:
    """Analytical proxy for the paper's Section-3.1 operating point.

    This is a calibrated reporting model, not a physical compilation target.
    It consumes verified P0 logical facts and makes its layout, timing, and
    factory assumptions explicit.
    """

    compute_distance: int = 25
    factory_distance: int = 27
    factory_count: int = 28
    factory_tiles: int = 8
    c_sep: int = 1024
    piece_width: int = 113
    fixed_piece_height: int = 33
    cycle_time_ns: float = 1_000.0
    reaction_time_ns: float = 10_000.0
    rearrange_time_ns: float = 1.0e6
    p_phys: float = 1.0e-3
    scaling_prefactor: float = 0.03
    threshold: float = 0.01
    factory_output_error: float = 5.5e-11


@dataclass(frozen=True, slots=True)
class GidneyEkeraEstimate:
    """Paper-model projections derived from a verified logical profile."""

    program: GidneyEkeraProgram
    model: GidneyEkeraResourceModel
    logical_profile: object
    toffolis: int
    lookup_cycles: int
    total_cycles: int
    compute_tiles: int
    physical_qubits: int
    runtime_hours: float
    p_idle: float
    p_factory: float
    retry_risk: float
    build_root: str
    build_sha256: str
    workload_variant: str = "compact_resource_envelope"
    bottleneck: str = "reaction_limited"


gidney_ekera_2019 = GidneyEkeraResourceModel()


def estimate_gidney_ekera(
    program: GidneyEkeraProgram,
    *,
    model: GidneyEkeraResourceModel = gidney_ekera_2019,
    logical_profile: LogicalProfile | None = None,
) -> GidneyEkeraEstimate:
    """Project a verified P0 profile onto the Gidney--Ekerå paper model."""

    if not isinstance(program, GidneyEkeraProgram):
        raise TypeError("estimate_gidney_ekera expects a GidneyEkeraProgram")
    if not isinstance(model, GidneyEkeraResourceModel):
        raise TypeError("model must be GidneyEkeraResourceModel")
    if program.c_sep != model.c_sep:
        raise ValueError(
            "program c_sep must match the Gidney--Ekerå resource model")

    from ..estimate import LogicalProfile, logical_counts

    if logical_profile is None:
        logical_profile = logical_counts(program.materialize())
    if not isinstance(logical_profile, LogicalProfile):
        raise TypeError("logical_profile must be a LogicalProfile")

    if logical_profile.actions.get("qlx_standard_ccz",
                                   0) != program.total_toffolis:
        raise ValueError("P0 actions disagree with the Gidney--Ekerå workload")
    if logical_profile.logical_qubits_peak != program.work_qubits:
        raise ValueError(
            "P0 logical_qubits_peak disagrees with the Gidney--Ekerå workload")
    if logical_profile.logical_qubits_peak != program.work_qubits:
        raise ValueError("folded P0 live-qubit count disagrees with workload")
    if logical_profile.synthesis_demand.get(
            'qlx_standard_ccz') != program.total_toffolis:
        raise ValueError(
            "P0 synthesis demand disagrees with the Gidney--Ekerå workload")

    window = program.c_exp + program.c_mul
    lookup_cycles = max(
        1,
        int(
            round((model.factory_distance / 2.0) * (1 << window) + 2 *
                  (model.c_sep + program.c_pad) *
                  max(1.0, model.reaction_time_ns / model.cycle_time_ns) +
                  model.rearrange_time_ns / model.cycle_time_ns)),
    )
    total_cycles = program.n_e * program.lookup_additions * lookup_cycles
    pieces = math.ceil(program.n_bits / model.c_sep)
    compute_tiles = (model.piece_width * pieces *
                     (model.fixed_piece_height + program.c_pad))
    physical_qubits = (compute_tiles * 2 * model.compute_distance**2 +
                       model.factory_count * model.factory_tiles * 2 *
                       model.factory_distance**2)
    runtime_hours = total_cycles * model.cycle_time_ns / 1.0e9 / 3600.0
    p_logical = min(
        1.0,
        model.scaling_prefactor *
        (model.p_phys / model.threshold)**((model.compute_distance + 1) / 2),
    )

    def repeated_failure(probability, trials):
        if probability <= 0.0 or trials <= 0:
            return 0.0
        if probability >= 1.0:
            return 1.0
        return -math.expm1(trials * math.log1p(-probability))

    p_idle = repeated_failure(
        p_logical, total_cycles * logical_profile.logical_qubits_peak)
    p_factory = repeated_failure(model.factory_output_error,
                                 program.total_toffolis)
    return GidneyEkeraEstimate(
        program=program,
        model=model,
        logical_profile=logical_profile,
        toffolis=program.total_toffolis,
        lookup_cycles=lookup_cycles,
        total_cycles=total_cycles,
        compute_tiles=compute_tiles,
        physical_qubits=physical_qubits,
        runtime_hours=runtime_hours,
        p_idle=p_idle,
        p_factory=p_factory,
        retry_risk=1.0 - (1.0 - p_idle) * (1.0 - p_factory),
        build_root=logical_profile.build_root,
        build_sha256=logical_profile.build_sha256,
    )


def gidney_ekera_factor(
    n_bits: int,
    *,
    c_exp: int = 5,
    c_mul: int = 5,
    c_sep: int = 1024,
    delta_off: int = 4,
) -> GidneyEkeraProgram:
    """Construct the calibrated two-piece folded workload for ``n_bits``.

    The checked-in aggregate model transcribes the historical example's two
    carry-runway pieces.  Other ``c_sep`` geometries require a different
    padding/register model and are rejected instead of silently reusing the
    coincidentally equal two-piece Toffoli formula.
    """

    for name, value in {
            "n_bits": n_bits,
            "c_exp": c_exp,
            "c_mul": c_mul,
            "c_sep": c_sep,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive Python int")
    if not isinstance(delta_off, int) or isinstance(delta_off, bool):
        raise ValueError("delta_off must be a Python int")
    if delta_off != 4:
        raise ValueError(
            "the bounded Gidney--Ekera proxy is calibrated only for "
            "delta_off == 4")
    if n_bits != 2 * c_sep:
        raise ValueError(
            "the Gidney--Ekera workload proxy is calibrated only for the "
            "two-piece geometry n_bits == 2 * c_sep")
    n_e = exponent_length(n_bits)
    c_pad = c_pad_for(n_bits, n_e, delta_off)
    lookup_additions = lookup_additions_per_modmul(n_bits, c_exp, c_mul, c_sep)
    per_lookup = toffolis_per_lookup_addition(n_bits + c_pad, c_exp, c_mul)
    # Exponent, coset accumulator, and arithmetic workspace.  The exact
    # allocation remains visible in P0 instead of being cost metadata.
    width = n_bits + c_pad
    address = c_exp + c_mul
    runway_carries = math.ceil(width / c_sep)
    work_qubits = 2 * width + address + runway_carries + 2

    def rsa_workload() -> None:
        work = allocate(work_qubits, name="work")

        def lookup_body(_lookup, left, middle, right):
            return repeat(
                per_lookup,
                carries=(left, middle, right),
                body=lambda _toffoli, a, b, c: ccz(a, b, c),
            )

        def exponent_body(_exponent, left, middle, right):
            return repeat(
                lookup_additions,
                carries=(left, middle, right),
                body=lookup_body,
            )

        work[0], work[1], work[2] = repeat(
            n_e,
            carries=(work[0], work[1], work[2]),
            body=exponent_body,
        )
        discard(work)

    definition = program(
        rsa_workload,
        name=f"gidney_ekera_rsa{n_bits}_w{c_exp}x{c_mul}",
    )
    return GidneyEkeraProgram(
        definition=definition,
        n_bits=n_bits,
        c_exp=c_exp,
        c_mul=c_mul,
        c_sep=c_sep,
        n_e=n_e,
        c_pad=c_pad,
        work_qubits=work_qubits,
        lookup_additions=lookup_additions,
        toffolis_per_lookup_addition=per_lookup,
    )


__all__ = [
    "GidneyEkeraProgram",
    "GidneyEkeraResourceModel",
    "GidneyEkeraEstimate",
    "c_pad_for",
    "exponent_length",
    "gidney_ekera_factor",
    "gidney_ekera_2019",
    "estimate_gidney_ekera",
    "lookup_additions_per_modmul",
    "toffolis_per_lookup_addition",
]
