# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Detailed surface-code factory and device used by the Gidney--Ekerå example.

This companion keeps the complete 15-to-1, CCZ, AutoCCZ, injection, timing,
and physical-device definitions inspectable without crowding the CUDA-Q
arithmetic kernel and its two estimation methods.

Every definition is parameterized by an `OperatingPoint` (one row of Table 3
in arXiv:1905.09749).  The arithmetic and layout quantities derive from that
row with the formulas of the paper's ancillary `estimate_costs.py`. The factory
protocols are built for the row's two surface-code distances.
"""

# %%
# Import the APIs used to define the factory protocols and physical device.
from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import cudaq.logical as cql

# %%
# Define the paper-selected arithmetic and machine operating points.
CARRY_RUNWAY_SEPARATION = 1_024
FACTORY_ROWS_PER_PIECE = 2
AUTOCZZ_ROUTING_PATCHES = 6
DETAILED_AUTOCZZ_WORKSPACES = 2
DETAILED_LEVEL2_PATCHES = 11 + DETAILED_AUTOCZZ_WORKSPACES * 9
RAW_T_INJECTION_LANES = 8 * 15
CCZ_FACTORY_DEPTH = 5
REACTION_TIME_CYCLES = 10
PHYSICAL_ERROR_RATE = 1.0e-3
FACTORY_CHARACTERIZATION_FAILURE_BUDGET = 0.8
DEFAULT_SCALING = cql.estimate.Scaling(prefactor=0.03, threshold=0.01)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One Table-3 row of arXiv:1905.09749 and its characterized factory lane.

    The first block is the paper's selected parameters.  The second block is
    the reference result of compiling one detailed AutoCCZ lane at the row's
    code distances. `characterize_surface_autoccz_factory` checks that the
    compiler still reproduces it.  Every other quantity is derived.
    """

    modulus_bits: int
    level_1_code_distance: int
    level_2_code_distance: int
    padding_offset: int
    exponent_window: int
    multiplication_window: int

    factory_schedule_events: int
    factory_startup_cycles: float
    factory_output_interval_cycles: float
    factory_lane_physical_qubits: int

    # Arithmetic: `Ekera--Hastad` exponent length, coset padding/runways, and
    # the folded lookup-addition recurrence.
    @property
    def exponent_qubits(self) -> int:
        m = math.ceil(self.modulus_bits / 2) - 1
        return m + 2 * (m - 20)

    @property
    def padding(self) -> int:
        return (
            math.ceil(math.log2(self.modulus_bits**2 * self.exponent_qubits)) +
            self.padding_offset)

    @property
    def carry_pieces(self) -> int:
        return math.ceil(self.modulus_bits / CARRY_RUNWAY_SEPARATION)

    @property
    def piece_length(self) -> int:
        return CARRY_RUNWAY_SEPARATION + self.padding

    @property
    def accumulator_width(self) -> int:
        return self.carry_pieces * self.piece_length

    @property
    def address_width(self) -> int:
        return self.exponent_window + self.multiplication_window

    @property
    def table_rows(self) -> int:
        return (1 << self.address_width) - 1

    @property
    def fixup_count(self) -> int:
        """Measurement-based unlookup, ``2 sqrt(rows)`` rounded up."""

        return math.ceil(2.0 * math.sqrt(self.table_rows))

    @property
    def lookup_count(self) -> int:
        multiplicand_bits = self.modulus_bits + self.padding + 2
        return math.ceil(2 * self.exponent_qubits * multiplicand_bits /
                         (self.exponent_window * self.multiplication_window))

    @property
    def toffolis_per_lookup(self) -> int:
        return (self.table_rows + self.fixup_count + self.carry_pieces *
                (2 * self.piece_length - 1))

    # Layout: each carry piece is two rows of CCZ factories over an operating
    # area and three interleaved register row groups.
    @property
    def level_1_lanes(self) -> int:
        t1_depth = 5.75 * self.level_1_code_distance / self.level_2_code_distance
        return math.ceil(8 * t1_depth / CCZ_FACTORY_DEPTH)

    @property
    def factory_box_width(self) -> int:
        ratio = self.level_1_code_distance / self.level_2_code_distance
        return math.ceil(8 * ratio * 2 + 3 + 2 * ratio)

    @property
    def factory_box_height(self) -> int:
        ratio = self.level_1_code_distance / self.level_2_code_distance
        return math.ceil(max(6, 4 * ratio * math.ceil(self.level_1_lanes / 2)))

    @property
    def factory_columns_per_piece(self) -> int:
        """Factory pairs needed to supply one CCZ per reaction time."""

        t1_depth = 5.75 * self.level_1_code_distance / self.level_2_code_distance
        ccz_cycles = max(CCZ_FACTORY_DEPTH,
                         t1_depth) * self.level_2_code_distance
        return math.ceil(ccz_cycles / REACTION_TIME_CYCLES / 2)

    @property
    def factory_lanes(self) -> int:
        return (self.carry_pieces * FACTORY_ROWS_PER_PIECE *
                self.factory_columns_per_piece)

    @property
    def piece_width(self) -> int:
        return ((self.factory_box_width + 1) * self.factory_columns_per_piece +
                1)

    @property
    def operating_rows(self) -> int:
        return 2 * self.factory_box_height + 2 * 3 + 3 + 6

    @property
    def piece_height(self) -> int:
        register_rows = math.ceil(self.piece_length / (self.piece_width - 2))
        return self.operating_rows + 3 * register_rows

    @property
    def board_patches(self) -> int:
        return self.carry_pieces * self.piece_width * self.piece_height

    @property
    def factory_patches(self) -> int:
        return (self.factory_lanes * self.factory_box_width *
                self.factory_box_height)

    @property
    def routing_pool_patches(self) -> int:
        return self.carry_pieces * AUTOCZZ_ROUTING_PATCHES

    @property
    def compute_patches(self) -> int:
        return (self.board_patches - self.factory_patches -
                self.routing_pool_patches)


RSA_2048 = OperatingPoint(
    modulus_bits=2_048,
    level_1_code_distance=15,
    level_2_code_distance=27,
    padding_offset=4,
    exponent_window=5,
    multiplication_window=5,
    factory_schedule_events=688,
    factory_startup_cycles=379.0,
    factory_output_interval_cycles=135.0,
    factory_lane_physical_qubits=142_808,
)
RSA_3072 = OperatingPoint(
    modulus_bits=3_072,
    level_1_code_distance=17,
    level_2_code_distance=29,
    padding_offset=6,
    exponent_window=5,
    multiplication_window=4,
    factory_schedule_events=688,
    factory_startup_cycles=417.0,
    factory_output_interval_cycles=145.0,
    factory_lane_physical_qubits=201_720,
)
RSA_4096 = OperatingPoint(
    modulus_bits=4_096,
    level_1_code_distance=17,
    level_2_code_distance=31,
    padding_offset=9,
    exponent_window=5,
    multiplication_window=4,
    factory_schedule_events=688,
    factory_startup_cycles=429.0,
    factory_output_interval_cycles=155.0,
    factory_lane_physical_qubits=186_488,
)
OPERATING_POINTS = {
    point.modulus_bits: point for point in (RSA_2048, RSA_3072, RSA_4096)
}


# %%
# Define helpers shared by the encoded factory protocols.
def _pauli_product(pauli, patches, indices):
    """Build one typed product over named logical ports."""

    selected = tuple(patches[index] for index in indices)
    product = pauli(selected[0][0])
    for patch in selected[1:]:
        product = product @ pauli(patch[0])
    return product


def _replace_product_successors(values, product, successors):
    """Advance owners in the canonical order carried by ``product``."""

    for factor, successor in zip(product.factors, successors, strict=True):
        owner = getattr(factor.operand, "patch", factor.operand)
        index = next(
            index for index, value in enumerate(values) if value is owner)
        values[index] = successor


def routing_region(context, placement):
    primary = context.qec_region_for(placement)
    binding = next(item for item in context.device.logical_to_qec
                   if item.qec_region == primary)
    if (len(binding.auxiliary_regions) != 1 or
            binding.auxiliary_regions[0].role != "scratch"):
        raise ValueError(
            "AutoCCZ lowering requires one auxiliary scratch QEC region")
    return binding.auxiliary_regions[0]


@dataclass(frozen=True, slots=True)
class SurfaceAutoCCZFactory:
    """The typed factory protocols and Toffoli lowering for two distances."""

    surface_l1: object
    surface: object
    logical_z: object
    raw_t_injection: object
    distill_15to1: object
    ccz_8to1: object
    autoccz_factory: object
    consume_autoccz_as_toffoli: object
    inject_autoccz: object

    @property
    def level_1_code(self):
        return self.surface_l1.code

    @property
    def code(self):
        return self.surface.code


# %%
# Build the complete factory for one level-1/level-2 distance pair.  The
# protocols close over the pair's codes, so each distinct pair is built once.
@functools.cache
def surface_autoccz_factory(level_1_distance: int,
                            level_2_distance: int) -> SurfaceAutoCCZFactory:
    SURFACE_L1 = cql.architectures.surface.definitions(level_1_distance)
    SURFACE = cql.architectures.surface.definitions(level_2_distance)
    LEVEL_1_CODE = SURFACE_L1.code
    CODE = SURFACE.code
    logical_z = cql.gadgets.logical_pauli(
        CODE,
        basis=cql.architecture.Basis.Z,
    )
    logical_x = cql.gadgets.logical_pauli(
        CODE,
        basis=cql.architecture.Basis.X,
    )

    def _conditional_z(bit, patch):
        return cql.cond(
            bit,
            then=lambda live: (logical_z(live),),
            else_=lambda live: (live,),
            carries=(patch,),
        )[0]

    # Raw state injection is the physical leaf of this detailed factory.  It
    # is a typed QEC producer with an explicit 120-lane physical
    # timing/resource binding. The distillation circuit therefore has no
    # unresolved external supply.
    @cql.protocol(implements=cql.logical.produce(cql.logical.RAW_T_STATE))
    def surface_raw_t_injection(
    ) -> cql.types.resource[cql.logical.RAW_T_STATE]:
        return cql.produce(cql.logical.RAW_T_STATE)

    # One explicit level-1 lane. The first four raw T states initialize the
    # four check rows of the compressed tri-orthogonal circuit. The other
    # eleven power its Z-product pi/4 rotations. Nothing here is a timing or
    # footprint model. These are ordinary QEC allocations, resource flows,
    # rotations, measurements, and post-selections that the selected native
    # projector carries into hardware.
    @cql.protocol(implements=cql.logical.produce(cql.logical.T_STATE))
    def surface_distill_15to1() -> cql.types.resource[cql.logical.T_STATE]:
        raw = cql.request_many(cql.logical.RAW_T_STATE, count=15)
        output = SURFACE_L1.prepare_plus(
            cql.allocate_patch(LEVEL_1_CODE, region="factory"))

        checks = []
        for state in raw[:4]:
            output, check = cql.unpack_resource(
                state,
                like=output,
                encoding=LEVEL_1_CODE,
            )
            checks.append(check)

        values = [*checks, output]
        for state, support in zip(
                raw[4:], cql.protocols.FIFTEEN_TO_ONE_ROTATION_SUPPORTS):
            product = _pauli_product(cql.types.Z, values, support)
            updated = cql.resource_rotate(
                state,
                product,
                angle=math.pi / 4.0,
            )
            _replace_product_successors(values, product, updated)

        # This positive-angle convention yields T-dagger on the odd row;
        # logical S converts it to the canonical T|+> resource.
        values[4] = SURFACE_L1.fold_s(values[4])
        for check in values[:4]:
            cql.postselect(SURFACE_L1.measure_x(check), expected=False)
        return cql.pack_resource(values[4], kind=cql.logical.T_STATE)

    # Gidney--Fowler Figure 5, expressed directly as typed CUDA-Q Logical. The
    # four initial X-product measurements are the four syndrome lines in the
    # figure. The eight distilled T resources drive the eight quarter rotations
    # on a..h. Their measurement results then enact the exact classical
    # correction matrix.
    @cql.protocol(implements=cql.logical.produce(cql.logical.CCZ_STATE))
    def surface_ccz_8to1() -> cql.types.resource[cql.logical.CCZ_STATE]:
        distilled_t = tuple(surface_distill_15to1() for _ in range(8))
        patches = [
            SURFACE.prepare_zero(
                cql.allocate_patch(CODE, region="factory_level2"))
            for _ in range(11)
        ]

        checks = []
        for support in cql.protocols.CCZ_8TO1_CHECK_SUPPORTS:
            product = _pauli_product(cql.types.X, patches, support)
            updated = cql.mpp(product)
            _replace_product_successors(patches, product, updated[:-1])
            checks.append(updated[-1])

        injection_bits = []
        for state, index in zip(distilled_t,
                                cql.protocols.CCZ_8TO1_INJECTION_TARGETS):
            (patches[index],) = cql.resource_rotate(
                state,
                cql.types.Z(patches[index][0]),
                angle=math.pi / 4.0,
            )
            patches[index] = SURFACE.h(patches[index])
            injection_bits.append(SURFACE.measure_z(patches[index]))

        parity = checks[1]
        for bit in injection_bits:
            parity = cql.xor(parity, bit)
        cql.postselect(parity, expected=False)

        # a..h -> output-Z masks 111, 110, 101, 100, 011, 010, 001, 000.
        for bit, mask in zip(injection_bits,
                             cql.protocols.CCZ_8TO1_OUTPUT_CORRECTION_MASKS):
            for output in range(3):
                if mask & (1 << (2 - output)):
                    patches[output] = _conditional_z(bit, patches[output])

        # The three remaining syndrome measurements correct outputs 1, 3, and
        # 2.
        for check, output in zip((checks[0], checks[2], checks[3]),
                                 cql.protocols.CCZ_8TO1_SYNDROME_OUTPUTS):
            patches[output] = _conditional_z(check, patches[output])
        patches[:3] = [logical_x(patch) for patch in patches[:3]]
        return cql.pack_resource(patches[:3], kind=cql.logical.CCZ_STATE)

    # The outer producer consumes the actual Figure-5 CCZ result, unpacks its
    # three output patches, adds six |+> routing patches, and applies every
    # encoded-CZ edge of the exact nine-patch AutoCCZ ring.
    @cql.protocol(implements=cql.logical.produce(cql.logical.AUTO_CCZ_STATE))
    def surface_autoccz_factory(
    ) -> cql.types.resource[cql.logical.AUTO_CCZ_STATE]:
        main_anchors = tuple(
            cql.allocate_patch(CODE, region="factory_level2") for _ in range(3))
        routing = tuple(
            cql.prepare_plus(cql.allocate_patch(CODE, region="factory_level2"))
            for _ in range(AUTOCZZ_ROUTING_PATCHES))
        ccz_state = surface_ccz_8to1()
        empty_anchors, main_payloads = cql.unpack_resource(
            ccz_state,
            like=main_anchors,
            logical_ports=((0,), (0,), (0,)),
        )
        cql.discard(empty_anchors)

        ring = [
            main_payloads[0],
            routing[0],
            routing[1],
            main_payloads[1],
            routing[2],
            routing[3],
            main_payloads[2],
            routing[4],
            routing[5],
        ]
        # An odd nine-cycle has edge-chromatic number three.  Authoring its
        # edges as three disjoint matchings exposes the actual encoded-CZ
        # concurrency to the generic physical scheduler without attaching a
        # duration model here.
        for matching in ((0, 3, 6), (1, 4, 7), (2, 5, 8)):
            updated = {}
            for left in matching:
                right = (left + 1) % len(ring)
                updated[left], updated[right] = SURFACE.cz(
                    ring[left], ring[right])
            for index, patch in updated.items():
                ring[index] = patch
        return cql.pack_resource(
            (
                ring[0],
                ring[3],
                ring[6],
                ring[1],
                ring[2],
                ring[4],
                ring[5],
                ring[7],
                ring[8],
            ),
            kind=cql.logical.AUTO_CCZ_STATE,
        )

    # Figure 4's adaptive consumer injects the three main payloads into the
    # two controls and target, resolves the six delayed-choice routing
    # measurements, applies the linear and quadratic Z corrections, and
    # returns the three live data patches.
    @cql.protocol
    def consume_surface_autoccz_as_toffoli(
        control_a: cql.patch[CODE],
        control_b: cql.patch[CODE],
        target: cql.patch[CODE],
        main_a: cql.patch[CODE],
        main_b: cql.patch[CODE],
        main_c: cql.patch[CODE],
        ab_a: cql.patch[CODE],
        ab_b: cql.patch[CODE],
        bc_b: cql.patch[CODE],
        bc_c: cql.patch[CODE],
        ca_c: cql.patch[CODE],
        ca_a: cql.patch[CODE],
    ) -> tuple[cql.patch[CODE], cql.patch[CODE], cql.patch[CODE]]:

        def measure_delayed_pair(choice, left, right):
            left, right = cql.cond(
                choice,
                then=lambda a, b: (SURFACE.h(a), SURFACE.h(b)),
                else_=lambda a, b: (a, b),
                carries=(left, right),
            )
            left_bit = SURFACE.measure_z(left)
            right_bit = SURFACE.measure_z(right)
            return cql.cond(
                choice,
                then=lambda a, b: (b, a),
                else_=lambda a, b: (a, b),
                carries=(left_bit, right_bit),
            )

        def apply_z_if(bit, block):
            return cql.cond(
                bit,
                then=lambda live: (logical_z(live),),
                else_=lambda live: (live,),
                carries=(block,),
            )[0]

        def z_if_both(first, second, block):
            return cql.cond(
                first,
                then=lambda live: (cql.cond(
                    second,
                    then=lambda nested: (logical_z(nested),),
                    else_=lambda nested: (nested,),
                    carries=(live,),
                )[0],),
                else_=lambda live: (live,),
                carries=(block,),
            )[0]

        target = SURFACE.h(target)
        control_a, main_a = SURFACE.transversal_cx(control_a, main_a)
        control_b, main_b = SURFACE.transversal_cx(control_b, main_b)
        target, main_c = SURFACE.transversal_cx(target, main_c)
        outcome_a = SURFACE.measure_z(main_a)
        outcome_b = SURFACE.measure_z(main_b)
        outcome_c = SURFACE.measure_z(main_c)

        bit_ab_a, bit_ab_b = measure_delayed_pair(outcome_c, ab_a, ab_b)
        bit_bc_b, bit_bc_c = measure_delayed_pair(outcome_a, bc_b, bc_c)
        bit_ca_c, bit_ca_a = measure_delayed_pair(outcome_b, ca_c, ca_a)

        control_a = apply_z_if(bit_ab_a, control_a)
        control_b = apply_z_if(bit_ab_b, control_b)
        control_b = apply_z_if(bit_bc_b, control_b)
        target = apply_z_if(bit_bc_c, target)
        target = apply_z_if(bit_ca_c, target)
        control_a = apply_z_if(bit_ca_a, control_a)
        target = z_if_both(outcome_a, outcome_b, target)
        control_a = z_if_both(outcome_b, outcome_c, control_a)
        control_b = z_if_both(outcome_a, outcome_c, control_b)
        target = SURFACE.h(target)
        return control_a, control_b, target

    # Selection is explicit and typed: every logical CCX on this code requests
    # an AutoCCZ state, unpacks the nine roles against three live data owners
    # and six fresh routing anchors, and invokes the adaptive consumer above.
    @cql.compiler.qec_lowering(
        objective=cql.logical.ccx,
        codes=(CODE,),
        dependencies=(consume_surface_autoccz_as_toffoli,),
        consumes=(cql.logical.AUTO_CCZ_STATE,),
        plugin="example.surface_autoccz",
        version="1",
    )
    def inject_surface_autoccz(site, context):
        encoding = context.encoding
        scratch = routing_region(context, site.placements[0])

        @cql.protocol(implements=cql.logical.ccx)
        def surface_autoccz_toffoli(
            control_a: cql.patch[encoding],
            control_b: cql.patch[encoding],
            target: cql.patch[encoding],
        ) -> tuple[
                cql.patch[encoding],
                cql.patch[encoding],
                cql.patch[encoding],
        ]:
            state = cql.event_await(cql.request(cql.logical.AUTO_CCZ_STATE))
            anchors = tuple(
                cql.allocate_patch(encoding, region=scratch)
                for _ in range(AUTOCZZ_ROUTING_PATCHES))
            successors, payloads = cql.unpack_resource(
                state,
                like=(control_a, control_b, target, *anchors),
                logical_ports=((0,), (0,), (0,), (), (), (), (), (), ()),
            )
            control_a, control_b, target = successors[:3]
            cql.discard(successors[3:])
            return consume_surface_autoccz_as_toffoli(
                control_a,
                control_b,
                target,
                *payloads,
            )

        return surface_autoccz_toffoli

    return SurfaceAutoCCZFactory(
        surface_l1=SURFACE_L1,
        surface=SURFACE,
        logical_z=logical_z,
        raw_t_injection=surface_raw_t_injection,
        distill_15to1=surface_distill_15to1,
        ccz_8to1=surface_ccz_8to1,
        autoccz_factory=surface_autoccz_factory,
        consume_autoccz_as_toffoli=consume_surface_autoccz_as_toffoli,
        inject_autoccz=inject_surface_autoccz,
    )


def factory_for(point: OperatingPoint) -> SurfaceAutoCCZFactory:
    return surface_autoccz_factory(point.level_1_code_distance,
                                   point.level_2_code_distance)


# These are ordinary CUDA-Q kernels. CUDA-Q Logical retains their typed
# helper-call graph through logical placement and QEC, then lowers each
# selected action normally. Neither the device nor a compiler pass assigns
# algorithm-specific roles to these helpers.


# %%
# Define the physical patch capabilities and distance-qualified timing model.
def _surface_resource_options(surface):
    return {
        "granularity":
            cql.architecture.ResourceGranularity.PATCH,
        "footprint":
            surface.square_patch_footprint,
        "native_actions":
            cql.architecture.physical_actions.clifford_set(),
        "native_instruments": (
            cql.architecture.physical_instruments.MX,
            cql.architecture.physical_instruments.MZ,
            cql.architecture.physical_instruments.MPP,
        ),
    }


def surface_timing(point: OperatingPoint = RSA_2048):
    cycle = 1 * cql.devices.us

    def encoded_patch_layer(distance):
        layer = distance * cycle
        return {
            # These are patch-level lattice-surgery macro durations. Pauli
            # corrections are frame updates; every other encoded operation is
            # one distance-round layer in this explicit operating-point model.
            "prepare_ns": layer,
            "h_ns": layer,
            "cx_ns": layer,
            "cz_ns": layer,
            "x_ns": 0 * cql.devices.ns,
            "z_ns": 0 * cql.devices.ns,
            "mpp_ns": layer,
            "rpp_ns": layer,
            "resource_rpp_ns": layer,
            "measure_z_instrument_ns": layer,
            "measure_x_instrument_ns": layer,
            "reset_ns": layer,
            # Packing changes typed ownership but performs no extra physical
            # evolution beyond the surrounding compiled operations.
            "pack_resource_ns": 0 * cql.devices.ns,
            "unpack_resource_ns": 0 * cql.devices.ns,
        }

    return cql.devices.TimingModel(
        {
            "surface_cycle_ns": cycle,
            "reaction_time_ns": REACTION_TIME_CYCLES * cycle,
            "condition_ns": REACTION_TIME_CYCLES * cycle,
            "postselect_ns": REACTION_TIME_CYCLES * cycle,
            "xor_ns": 0 * cql.devices.ns,
        },
        by_code_distance={
            distance: encoded_patch_layer(distance)
            for distance in (point.level_1_code_distance,
                             point.level_2_code_distance)
        },
        source=("surface-code patch macro model: 1us code cycle and one "
                "distance-round layer per encoded operation; informed by "
                "arXiv:1812.01238"),
    )


def _operating_calibration(p_phys, scaling):
    return {
        "physical_error": p_phys,
        "surface_scaling_prefactor": scaling.prefactor,
        "surface_threshold": scaling.threshold,
    }


# %%
# Build the device used to compile and characterize one detailed factory lane.
def build_detailed_factory_device(
    point: OperatingPoint = RSA_2048,
    *,
    p_phys=PHYSICAL_ERROR_RATE,
    scaling=DEFAULT_SCALING,
):
    """Build one explicit level-1/level-2 AutoCCZ production lane."""

    factory = factory_for(point)
    d1 = point.level_1_code_distance
    d2 = point.level_2_code_distance
    architecture = cql.devices.QECArchitecture(
        f"surface_autoccz_factory_d{d1}_d{d2}",
        factory.level_1_code.default_encoding,
        auxiliary_regions=(cql.devices.QECRegion(
            "level2",
            factory.code.default_encoding,
            DETAILED_LEVEL2_PATCHES,
            role="scratch",
        ),),
    )
    builder = cql.devices.DeviceBuilder("DetailedSurfaceAutoCCZLane")
    autoccz = builder.logical.add_factory(
        produces=cql.logical.AUTO_CCZ_STATE,
        via=factory.autoccz_factory,
        capacity=1,
        name="factory",
        stream_name="autoccz_states",
    )
    raw_factory = builder.logical.add_factory(
        produces=cql.logical.RAW_T_STATE,
        via=factory.raw_t_injection,
        capacity=RAW_T_INJECTION_LANES,
        name="raw_t_injection",
        stream_name="raw_t_states",
    )
    factory_qec = builder.qec.bind(
        autoccz,
        architecture=architecture,
        # Four check rows and one output row per level-1 15-to-1 lane.
        block_capacity=point.level_1_lanes * 5,
    )
    raw_qec = builder.qec.bind(
        raw_factory,
        architecture=cql.devices.QECArchitecture(
            f"surface_raw_t_injection_d{d1}",
            factory.level_1_code.default_encoding,
        ),
        block_capacity=RAW_T_INJECTION_LANES,
    )
    factory_layout = builder.physical.add_resources(
        "surface_code_patch",
        point.factory_box_width * point.factory_box_height,
        name="factory_layout_patches",
        **_surface_resource_options(factory.surface),
    )
    raw_injection_qubits = builder.physical.add_qubits(
        RAW_T_INJECTION_LANES,
        name="raw_t_injection_qubits",
    )
    builder.physical.bind(factory_qec, to=factory_layout)
    builder.physical.bind(
        factory_qec.auxiliary_regions[0],
        to=factory_layout,
    )
    builder.physical.bind(
        raw_qec,
        to=raw_injection_qubits,
        factory_model=cql.devices.FactoryModel(
            startup_cycles=1,
            output_interval_cycles=1,
            evidence=cql.analysis.user_assertion(
                "one physical injection qubit and one surface cycle per raw "
                "T state"),
        ),
    )
    builder.physical.set_operating_point(
        timing=surface_timing(point),
        calibration=_operating_calibration(p_phys, scaling),
    )
    return builder.build()


# %%
# Compile one detailed lane once, then derive both its resource estimate and
# the compact component model consumed by the paper-scale device.
def characterize_surface_autoccz_factory(
    point: OperatingPoint = RSA_2048,
    *,
    p_phys=PHYSICAL_ERROR_RATE,
    failure_budget=FACTORY_CHARACTERIZATION_FAILURE_BUDGET,
    scaling=DEFAULT_SCALING,
):
    detailed_device = build_detailed_factory_device(
        point,
        p_phys=p_phys,
        scaling=scaling,
    )
    detailed_schedule = cql.compiler.schedule(
        factory_for(point).autoccz_factory,
        device=detailed_device,
    )
    resource_estimate = cql.estimate(
        detailed_schedule,
        tier=cql.estimate.Tier.SCHEDULE,
        failure_budget=failure_budget,
    )
    factory_model = cql.compiler.factory_model(
        detailed_schedule,
        produces=cql.logical.AUTO_CCZ_STATE,
    )

    # Make sure the compiler is using the expected factory reference values.
    assert resource_estimate.event_count == len(
        detailed_schedule.entries) == point.factory_schedule_events
    assert math.isclose(factory_model.startup_cycles,
                        point.factory_startup_cycles)
    assert math.isclose(factory_model.output_interval_cycles,
                        point.factory_output_interval_cycles)
    assert (factory_model.characterization.physical_units ==
            point.factory_lane_physical_qubits)
    return resource_estimate, factory_model


# %%
# Calculate the detailed factory resources and use its compact model to build
# the workload-neutral paper-scale device.
def build_paper_device(
    point: OperatingPoint = RSA_2048,
    *,
    factory_lanes=None,
    p_phys=PHYSICAL_ERROR_RATE,
    scaling=DEFAULT_SCALING,
):
    """Build the level-2 board from one characterized factory lane."""

    if factory_lanes is None:
        factory_lanes = point.factory_lanes
    if (isinstance(factory_lanes, bool) or not isinstance(factory_lanes, int) or
            factory_lanes <= 0):
        raise ValueError("factory_lanes must be a positive integer")
    compute_patch_count = (
        point.board_patches -
        factory_lanes * point.factory_box_width * point.factory_box_height -
        point.routing_pool_patches)
    if compute_patch_count <= 0:
        raise ValueError("factory_lanes leave no physical compute patches")

    _, factory_model = characterize_surface_autoccz_factory(
        point,
        p_phys=p_phys,
        scaling=scaling,
    )

    factory = factory_for(point)
    architecture = cql.devices.QECArchitecture(
        f"surface_autoccz_d{point.level_2_code_distance}",
        factory.code.default_encoding,
        link_roots=(
            *factory.surface.wsc().link_roots,
            factory.logical_z,
            factory.inject_autoccz,
        ),
        auxiliary_regions=(cql.devices.QECRegion(
            "autoccz_routing",
            factory.code.default_encoding,
            AUTOCZZ_ROUTING_PATCHES,
            role="scratch",
        ),),
    )
    builder = cql.devices.DeviceBuilder("GidneyEkeraSurfaceBoard")
    compute = builder.logical.add_compute(
        capacity=compute_patch_count,
        name="compute",
    )
    autoccz = builder.logical.add_factory(
        produces=cql.logical.AUTO_CCZ_STATE,
        via=factory.autoccz_factory,
        capacity=factory_lanes,
        name="autoccz_factory",
        stream_name="autoccz_states",
    )
    builder.logical.add_stream(
        cql.logical.RAW_T_STATE,
        name="raw_t_states",
        external=True,
    )
    compute_qec = builder.qec.bind(compute, architecture=architecture)
    factory_qec = builder.qec.bind(
        autoccz,
        encoding=factory.code,
        block_capacity=9 * factory_lanes,
    )
    resource_options = _surface_resource_options(factory.surface)
    compute_patches = builder.physical.add_resources(
        "surface_code_patch",
        compute_patch_count,
        name="compute_patches",
        **resource_options,
    )
    routing_patches = builder.physical.add_resources(
        "surface_code_patch",
        point.routing_pool_patches,
        name="routing_patches",
        **resource_options,
    )
    # One schedulable member is one complete, independently compiled factory
    # lane. Its base-unit footprint comes from the characterized physical
    # schedule, including the explicit raw-injection leaf; the RSA device does
    # not recreate or round this value from paper geometry.
    factory_lane_resources = builder.physical.add_resources(
        "autoccz_factory_lane",
        factory_lanes,
        name="factory_lanes",
        granularity=cql.architecture.ResourceGranularity.PATCH,
        footprint=cql.architecture.PhysicalFootprint(
            factory_model.characterization.physical_unit_kind,
            factory_model.characterization.physical_units,
            "compiler-characterized detailed AutoCCZ factory physical schedule",
        ),
    )
    builder.physical.bind(compute_qec, to=compute_patches)
    builder.physical.bind(
        compute_qec.auxiliary_regions[0],
        to=routing_patches,
    )
    builder.physical.bind(
        factory_qec,
        to=factory_lane_resources,
        factory_model=factory_model,
    )
    builder.physical.set_operating_point(
        timing=surface_timing(point),
        calibration=_operating_calibration(p_phys, scaling),
    )
    return builder.build()
