# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Folded logical workload for Gidney--Ekerå windowed RSA factoring.

The library keeps billion-operation multiplicities in nested ``cflow.repeat``
regions.  It is intentionally a logical workload: device-specific factories,
reaction timing, and failure models enter through later P1--P3 compilation and
estimation products.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import TYPE_CHECKING

from cudaq.logical.programs.decorators import program
from cudaq.logical.ops._impl import (
    allocate,
    ccx,
    discard,
    repeat,
)
from cudaq.logical.programs.definition import ProgramDefinition

if TYPE_CHECKING:
    from ..compiler import Build
    from ..devices import Device


def exponent_length(n_bits: int) -> int:
    """Ekerå--Håstad short-DLP exponent length ``ceil(3n/2)``."""

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
class GidneyEkeraArchitecture:
    """Repository analytical proxy for the paper's Section-3.1 operating point.

    The proxy preserves the checked-in QLX calibration. It is not the paper's
    complete ancillary estimator: compute and factory distances are explicit,
    and the failure model includes logical-idle and accepted-factory-output
    buckets rather than every error bucket in the publication.
    """

    compute_distance: int = 25
    level_1_factory_distance: int = 15
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
    # Effective accepted |CCZ> output error of the two-level catalyzed
    # factory at this operating point. It is a device-model input, separate
    # from the algorithm's exactly counted CCZ demand.
    factory_output_error: float = 5.5e-11

    def device(
        self,
        program: GidneyEkeraProgram,
        *,
        name: str = "GidneyEkeraLogicalArchitecture",
    ) -> Device:
        """Materialize the proxy's typed compute-plus-CCZ-factory device.

        The program supplies only the derived problem-size capacity. Code
        distances, factory provisioning, carrier pools, timing, and calibration stay
        architecture facts. The resulting device can place either the direct
        QLX P0 or an imported CUDA-Q P0 with the accepted aggregate resource
        profile.
        """

        if not isinstance(program, GidneyEkeraProgram):
            raise TypeError("GidneyEkeraArchitecture.device expects a "
                            "GidneyEkeraProgram")
        if program.c_sep != self.c_sep:
            raise ValueError(
                "program c_sep must match the Gidney--Ekerå architecture")
        from .. import codes, protocols, standard
        from ..devices import DeviceBuilder

        pieces = math.ceil(program.n_bits / self.c_sep)
        compute_tiles = (self.piece_width * pieces *
                         (self.fixed_piece_height + program.c_pad))
        compute_qubits = compute_tiles * 2 * self.compute_distance**2
        factory_qubits = (self.factory_count * self.factory_tiles * 2 *
                          self.factory_distance**2)

        builder = DeviceBuilder(
            name,
            metadata={
                "paper": "arXiv:1905.09749",
                "compute_distance": self.compute_distance,
                "level_1_factory_distance": self.level_1_factory_distance,
                "factory_distance": self.factory_distance,
            },
        )
        builder.physical.set_operating_point(
            timing={
                "surface_cycle_ns": self.cycle_time_ns,
                "reaction_time_ns": self.reaction_time_ns,
                "rearrange_time_ns": self.rearrange_time_ns,
            },
            calibration={
                "physical_error": self.p_phys,
                "surface_scaling_prefactor": self.scaling_prefactor,
                "surface_threshold": self.threshold,
                "accepted_ccz_error": self.factory_output_error,
            },
        )
        compute = builder.logical.add_compute(capacity=program.work_qubits)
        factories = builder.logical.add_factory(
            produces=standard.CCZ_STATE,
            via=protocols.ccz_gidney_fowler,
            capacity=self.factory_count,
            buffer_size=1,
            name="ccz_factories",
            stream_name="ccz_states",
        )
        compute_qec = builder.qec.bind(compute, encoding=codes.BareQubit)
        factory_qec = builder.qec.bind(factories, encoding=codes.BareQubit)
        compute_carriers = builder.physical.add_qubits(compute_qubits,
                                                       name="compute_qubits")
        factory_carriers = builder.physical.add_qubits(factory_qubits,
                                                       name="factory_qubits")
        builder.physical.bind(compute_qec, to=compute_carriers)
        builder.physical.bind(factory_qec, to=factory_carriers)
        return builder.build()


@dataclass(frozen=True, slots=True)
class GidneyEkeraEstimate:
    program: GidneyEkeraProgram
    architecture: GidneyEkeraArchitecture
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


gidney_ekera_2019 = GidneyEkeraArchitecture()


def estimate_gidney_ekera(
    program: GidneyEkeraProgram,
    *,
    architecture: GidneyEkeraArchitecture = gidney_ekera_2019,
    build: Build | None = None,
) -> GidneyEkeraEstimate:
    """Estimate the analytical operating-point proxy from verified P0 facts.

    The complete estimator-relevant logical fingerprint is read back from
    ``build`` when supplied, otherwise from the program's direct QLX P0. This
    lets an independently imported CUDA-Q P0 use the exact same architecture
    equations without transcribing its counts while rejecting builds with a
    different aggregate resource profile. This check does not authenticate
    circuit ordering, dataflow, or algorithmic equivalence. The result commits
    to the accepted build bundle. Layout, code distance, timing, factory
    multiplicity, and output error are explicit proxy inputs.
    """

    if not isinstance(program, GidneyEkeraProgram):
        raise TypeError("estimate_gidney_ekera expects a GidneyEkeraProgram")
    if not isinstance(architecture, GidneyEkeraArchitecture):
        raise TypeError("architecture must be GidneyEkeraArchitecture")
    if program.c_sep != architecture.c_sep:
        raise ValueError(
            "program c_sep must match the Gidney--Ekerå architecture")
    from ..estimate import logical_counts
    from ..compiler import Build
    from ..stages import P0

    if build is None:
        build = program.materialize()
    if not isinstance(build, Build):
        raise TypeError("build must be a QLX Build")
    if build.stage != P0:
        raise ValueError("Gidney--Ekerå estimation requires a P0 Build")
    logical_profile = logical_counts(build)
    normalized_actions = dict(logical_profile.actions)
    normalized_instruments = dict(logical_profile.instruments)
    plus_preparations = normalized_instruments.pop("qlx_standard_prepare_plus",
                                                   0)
    if plus_preparations:
        # Standard CUDA-Q spells |+> allocation as |0> preparation followed by
        # H. Normalize the direct QLX spelling to that common semantic profile.
        normalized_instruments["qlx_standard_prepare_zero"] = (
            normalized_instruments.get("qlx_standard_prepare_zero", 0) +
            plus_preparations)
        normalized_actions["qlx_standard_h"] = (
            normalized_actions.get("qlx_standard_h", 0) + plus_preparations)
    normalized_depth = (logical_profile.action_depth_upper_bound +
                        plus_preparations)

    compact_actions = {"qlx_standard_ccx": program.total_toffolis}
    compact_instruments = {"qlx_standard_prepare_zero": program.work_qubits}
    historical_actions = {
        "qlx_standard_h": 5_038_080,
        "qlx_standard_ccx": 2_632_900_608,
        "qlx_standard_cx": 4_719_169_536,
    }
    historical_instruments = {
        "qlx_standard_prepare_zero": 1_058_502_694,
        "qlx_standard_measure_x": 1_055_981_569,
        "qlx_standard_measure_z": 2_519_040,
    }
    is_compact = (normalized_actions == compact_actions and
                  normalized_instruments == compact_instruments and
                  normalized_depth
                  == program.work_qubits + program.total_toffolis and
                  logical_profile.discards in (1, program.work_qubits))
    historical_parameters = (program.n_bits == 2048 and program.c_exp == 5 and
                             program.c_mul == 5 and program.c_sep == 1024)
    is_historical_profile = (historical_parameters and
                             normalized_actions == historical_actions and
                             normalized_instruments == historical_instruments
                             and normalized_depth == 9_474_111_527 and
                             logical_profile.discards in (1, 2085))
    if not (is_compact or is_historical_profile):
        raise ValueError(
            "P0 action profile disagrees with the Gidney--Ekerå workload")
    workload_variant = ("historical_aggregate_resource_profile" if
                        is_historical_profile else "compact_resource_envelope")
    toffolis = normalized_actions.get("qlx_standard_ccx", 0)
    if logical_profile.idle_sites != 0:
        raise ValueError("Gidney--Ekerå P0 must not contain explicit idle work")
    if logical_profile.logical_qubits_peak != program.work_qubits:
        raise ValueError("folded P0 live-qubit count disagrees with workload")
    if logical_profile.synthesis_demand != {
            "qlx_standard_ccx": program.total_toffolis
    }:
        raise ValueError(
            "P0 synthesis demand disagrees with the Gidney--Ekerå workload")
    if logical_profile.resource_requests or logical_profile.resource_consumptions:
        raise ValueError(
            "standard-action Gidney--Ekerå P0 must not contain resource flow")

    build_sha256 = hashlib.sha256(build.serialize()).hexdigest()

    window = program.c_exp + program.c_mul
    lookup_cycles = max(
        1,
        int(
            round(
                (architecture.factory_distance / 2.0) * (1 << window) + 2 *
                (architecture.c_sep + program.c_pad) * max(
                    1.0,
                    architecture.reaction_time_ns / architecture.cycle_time_ns,
                ) +
                architecture.rearrange_time_ns / architecture.cycle_time_ns)),
    )
    total_cycles = program.n_e * program.lookup_additions * lookup_cycles
    pieces = math.ceil(program.n_bits / architecture.c_sep)
    compute_tiles = (architecture.piece_width * pieces *
                     (architecture.fixed_piece_height + program.c_pad))
    physical_qubits = (compute_tiles * 2 * architecture.compute_distance**2 +
                       architecture.factory_count * architecture.factory_tiles *
                       2 * architecture.factory_distance**2)
    runtime_hours = (total_cycles * architecture.cycle_time_ns / 1.0e9 / 3600.0)
    p_logical = min(
        1.0,
        architecture.scaling_prefactor *
        (architecture.p_phys / architecture.threshold)**(
            (architecture.compute_distance + 1) / 2),
    )

    def repeated_failure(probability, trials):
        if probability <= 0.0 or trials <= 0:
            return 0.0
        if probability >= 1.0:
            return 1.0
        return -math.expm1(trials * math.log1p(-probability))

    p_idle = repeated_failure(
        p_logical, total_cycles * logical_profile.logical_qubits_peak)
    p_factory = repeated_failure(architecture.factory_output_error, toffolis)
    retry_risk = 1.0 - (1.0 - p_idle) * (1.0 - p_factory)
    return GidneyEkeraEstimate(
        program=program,
        architecture=architecture,
        logical_profile=logical_profile,
        toffolis=toffolis,
        lookup_cycles=lookup_cycles,
        total_cycles=total_cycles,
        compute_tiles=compute_tiles,
        physical_qubits=physical_qubits,
        runtime_hours=runtime_hours,
        p_idle=p_idle,
        p_factory=p_factory,
        retry_risk=retry_risk,
        build_root=build.root.symbol,
        build_sha256=build_sha256,
        workload_variant=workload_variant,
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
    if (not isinstance(delta_off, int) or isinstance(delta_off, bool) or
            delta_off != 4):
        raise ValueError(
            "the calibrated Gidney--Ekerå workload requires delta_off == 4")
    if n_bits != 2 * c_sep:
        raise ValueError(
            "the Gidney--Ekerå workload proxy is calibrated only for the "
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
                body=lambda _toffoli, a, b, c: ccx(a, b, c),
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
        estimate_only=True,
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
    "GidneyEkeraArchitecture",
    "GidneyEkeraEstimate",
    "c_pad_for",
    "exponent_length",
    "gidney_ekera_factor",
    "gidney_ekera_2019",
    "estimate_gidney_ekera",
    "lookup_additions_per_modmul",
    "toffolis_per_lookup_addition",
]
