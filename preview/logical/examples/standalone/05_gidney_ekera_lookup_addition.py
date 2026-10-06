# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lower a small Gidney--Ekerå lookup-addition circuit from P0 through P3.

The paper's modular exponentiation is built from reversible table lookups and
in-place additions. This quick example instantiates one complete four-bit
lookup-addition primitive: a two-bit address selects a classical table row,
the selected value is added modulo 16, and the lookup workspace is uncomputed.
It is a real unitary circuit, not a gate-count proxy or an RSA-2048 projection.

CUDA-Q's GridSynth-backed Clifford+T pass legalizes its Toffolis. PBC
normalization then exposes the T-state rotations selected by a distance-three
surface-code architecture before the ordinary P1, P2, and P3 lowerings.
"""

# %%
# Import the standalone logical compiler and define the lookup constants.
import cudaq.logical as cql

WIDTH = 4
LOOKUP_TABLE = (1, 3, 5, 7)


# %%
# Define the reversible lookup and ripple-carry arithmetic helpers.
def _majority(carry, target, source):
    """Cuccaro majority step, with the eventual sum retained in ``target``."""

    source, target = cql.cx(source, target)
    source, carry = cql.cx(source, carry)
    carry, target, source = cql.ccx(carry, target, source)
    return carry, target, source


def _unmajority_add(carry, target, source):
    """Restore the carry/source wires while completing the sum bit."""

    carry, target, source = cql.ccx(carry, target, source)
    source, carry = cql.cx(source, carry)
    carry, target = cql.cx(carry, target)
    return carry, target, source


def _add_mod_16(source, target, carry):
    """Reversibly apply ``target += source (mod 16)``."""

    carry, target[0], source[0] = _majority(carry, target[0], source[0])
    for bit in range(1, WIDTH):
        source[bit - 1], target[bit], source[bit] = _majority(
            source[bit - 1], target[bit], source[bit])
    for bit in range(WIDTH - 1, 0, -1):
        source[bit - 1], target[bit], source[bit] = _unmajority_add(
            source[bit - 1], target[bit], source[bit])
    carry, target[0], source[0] = _unmajority_add(carry, target[0], source[0])
    return source, target, carry


def _lookup(address, output, *, inverse=False):
    """XOR ``LOOKUP_TABLE[address]`` into ``output`` and return live owners."""

    rows = range(len(LOOKUP_TABLE) -
                 1, -1, -1) if inverse else range(len(LOOKUP_TABLE))
    for row in rows:
        zero_controls = tuple(
            bit for bit in range(len(address)) if not row & (1 << bit))
        for bit in zero_controls:
            address[bit] = cql.x(address[bit])
        for target_bit in range(WIDTH):
            if LOOKUP_TABLE[row] & (1 << target_bit):
                address[0], address[1], output[target_bit] = cql.ccx(
                    address[0], address[1], output[target_bit])
        for bit in reversed(zero_controls):
            address[bit] = cql.x(address[bit])
    return address, output


# %%
# Author one complete lookup-addition primitive as a logical program.
@cql.program
def gidney_ekera_lookup_addition() -> None:
    address = list(cql.allocate(2, state=cql.types.plus, name="address"))
    lookup = list(cql.allocate(WIDTH, state=cql.types.zero, name="lookup"))
    accumulator = list(
        cql.allocate(WIDTH, state=cql.types.zero, name="accumulator"))
    carry = cql.prepare_zero()

    # Add one first, then coherently add the selected odd lookup-table value.
    accumulator[0] = cql.x(accumulator[0])
    address, lookup = _lookup(address, lookup)
    lookup, accumulator, carry = _add_mod_16(lookup, accumulator, carry)
    address, lookup = _lookup(address, lookup, inverse=True)

    cql.discard((*address, *lookup, *accumulator, carry))


# %%
# Define the surface-code architecture and its typed T-state source.
surface = cql.architectures.surface.definitions(3)
surface_code = surface.code
surface_pbc = cql.architectures.pinnacle.for_code(surface_code, cycle_rounds=3)


@cql.protocol(implements=cql.logical.produce(cql.logical.T_STATE))
def surface_t_state_source() -> cql.types.resource[cql.logical.T_STATE]:
    """Typed boundary for the surface-code factory service used in this study."""

    return cql.produce(cql.logical.T_STATE)


# %%
# Bind compute patches, a factory lane, and timing into a physical device.
device_builder = cql.devices.DeviceBuilder("GidneyEkeraSurfaceCodeDevice")
compute = device_builder.logical.add_compute(capacity=11, name="compute")
factory = device_builder.logical.add_factory(
    produces=cql.logical.T_STATE,
    via=surface_t_state_source,
    capacity=1,
    buffer_size=1,
    name="surface_t_factory",
)
compute_qec = device_builder.qec.bind(compute, architecture=surface_pbc)
factory_qec = device_builder.qec.bind(factory, encoding=surface_code)

patch_options = {
    "granularity": cql.architecture.ResourceGranularity.PATCH,
    "footprint": surface.square_patch_footprint,
    "native_actions": cql.architecture.physical_actions.clifford_set(),
    "native_instruments": (cql.architecture.physical_instruments.MPP,),
    "capabilities": (cql.architecture.NATIVE_PAULI_PRODUCT_ROTATION,),
}
compute_patches = device_builder.physical.add_resources("surface_code_patch",
                                                        11,
                                                        name="compute_patches",
                                                        **patch_options)
scratch_patches = device_builder.physical.add_resources(
    "surface_code_patch", 1, name="pbc_scratch_patch", **patch_options)
factory_lane = device_builder.physical.add_resources(
    "surface_code_factory",
    1,
    name="surface_t_factory_lane",
    granularity=cql.architecture.ResourceGranularity.PATCH,
    footprint=cql.architecture.PhysicalFootprint(
        "qubit",
        15 * surface.square_patch_footprint.units,
        "one compact distance-three 15-to-1 factory service lane",
    ),
)
device_builder.physical.bind(compute_qec, to=compute_patches)
(scratch_qec,) = compute_qec.auxiliary_regions
device_builder.physical.bind(scratch_qec, to=scratch_patches)
device_builder.physical.bind(
    factory_qec,
    to=factory_lane,
    factory_model=cql.devices.FactoryModel(
        startup_cycles=12,
        output_interval_cycles=6,
        evidence=cql.analysis.user_assertion(
            "compact distance-three surface-code factory teaching model"),
    ),
)
device_builder.physical.set_operating_point(timing={
    "cycle_ns": 1_000.0,
    "surface_cycle_ns": 1_000.0,
    "rpp_ns": 3_000.0,
    "mpp_ns": 3_000.0,
    "condition_ns": 1_000.0,
},)
surface_device = device_builder.build()

# %%
# Compile explicitly through synthesis, placement, QEC, and physical lowering.
# CCX first enters CUDA-Q's shared GridSynth-backed Clifford+T pipeline. The
# device-free PBC pass retains the exact unitary while expressing each T gate
# as a typed pi/4 Pauli-product rotation suitable for surface-code injection.
synthesized_p0 = cql.compiler.synthesize(
    gidney_ekera_lookup_addition,
    gate_set=cql.compiler.gate_sets.clifford_t,
    precision=1.0e-10,
)
p0 = cql.compiler.to_pbc(synthesized_p0)
p1 = cql.compiler.place(p0, device=surface_device)
p2 = cql.compile(
    p1,
    pipeline=cql.compiler.pipelines.qec(),
    device=surface_device,
)
p3 = cql.compile(
    p2,
    pipeline=cql.compiler.pipelines.physical(),
    device=surface_device,
)
schedule = cql.compiler.schedule(p3)
analytical_estimate = cql.estimate(
    p2,
    tier=cql.estimate.Tier.ANALYTICAL,
    p_phys=1.0e-3,
    failure_budget=0.1,
    cycle_time=1.0e-6,
)
estimate = cql.estimate(
    schedule,
    tier=cql.estimate.Tier.SCHEDULE,
    p_phys=1.0e-3,
    failure_budget=0.1,
    cycle_time=1.0e-6,
)

# %%
# Validate and report the analytical and scheduled physical results.
assert (p0.stage, p1.stage, p2.stage, p3.stage) == (
    cql.stages.P0,
    cql.stages.P1,
    cql.stages.P2,
    cql.stages.P3,
)
assert synthesized_p0.synthesis.gate_set == "clifford_t"
assert cql.stages.PROTOCOL_NETWORK in p2.facets
assert schedule.build.module.operation.verify()
assert estimate.physical_qubits > 11 * surface.square_patch_footprint.units
assert analytical_estimate.physical_qubits_peak == estimate.physical_qubits
assert estimate.event_count == len(schedule.entries)
assert estimate.makespan_ns == schedule.makespan_ns

print("Gidney--Ekerå lookup addition, P0 -> P1 -> P2 -> P3:")
print("  code: distance-three rotated surface code")
print(f"  lookup table: {LOOKUP_TABLE}")
print(f"  physical qubits: {estimate.physical_qubits}")
print(f"  analytical budget met: {analytical_estimate.budget_met}")
print(f"  scheduled events: {estimate.event_count}")
print(f"  makespan: {estimate.makespan_ns:.1f} ns")
