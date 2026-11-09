# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile and run a CUDA-Q Bell kernel on Steane-encoded qubits.

The kernel is compiled for CUDA-Q Logical and imported as P0, and explicit
compiler calls take it to a P3 schedule. A CUDA-Q target built from the same
device then estimates the kernel, and `cudaq.run` executes the kernel and its
P3 Build.

This runnable example supplies the excerpts in cudaq-logical-in-practice.md.
"""

# %%
# Author the Bell pair as an ordinary CUDA-Q kernel.
# [START_BELL_KERNEL]
import cudaq
import cudaq.logical as cql


@cudaq.kernel
def bell_pair() -> tuple[bool, bool]:
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    return mz(qubits[0]), mz(qubits[1])


# [END_BELL_KERNEL]


# %%
# Define the Steane CSS code and materialize its compiler representation.
# [START_STEANE_CODE]
@cql.code
class Steane:
    block = cql.codes.CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = (tuple(range(7)),)
    lz = lx


code = cql.materialize(Steane)

assert (Steane.n, Steane.k, Steane.d.value) == (7, 1, 3)
assert Steane.block.size == 13
# [END_STEANE_CODE]


# %%
# Provide transversal implementations of the Bell-pair gates.
# [START_STEANE_GADGETS]
@cql.gadget(implements=cql.standard.h)
def steane_h(block: cql.patch[Steane]) -> cql.patch[Steane]:
    return cql.h(block.data)


@cql.gadget(implements=cql.standard.cx)
def steane_cx(
    control: cql.patch[Steane],
    target: cql.patch[Steane],
) -> tuple[cql.patch[Steane], cql.patch[Steane]]:
    return cql.cx(control.data, target.data)


# [END_STEANE_GADGETS]


# %%
# Prepare encoded |0> with an explicit circuit, as in standalone example 06.
# Resetting the data qubits alone gives |0000000>, which does not satisfy the
# X stabilizers.
# [START_STEANE_ENCODER]
@cql.gadget(implements=cql.standard.prepare_zero)
def steane_prepare_zero(block: cql.patch[Steane]) -> cql.patch[Steane]:
    block = cql.reset(block.data)
    for index in (3, 5, 6):
        block = cql.h(block.data[index])
    for control, targets in ((3, (0, 1, 2)), (5, (0, 1, 4)), (6, (0, 2, 4))):
        for target in targets:
            block = cql.cx(block.data[control], block.data[target])
    return block


# Check that the encoder prepares logical |0>. The contract binds the block's
# syndrome to zero, which requires every stabilizer to be +1, and the
# `prepare_zero` claim requires logical Z to be +1.
encoder_contract = cql.gadgets.GadgetContract(
    steane_prepare_zero,
    boundary={steane_prepare_zero.outputs.only().syndrome: False},
)
assert cql.gadgets.verify_gadget(encoder_contract).verified
# [END_STEANE_ENCODER]

# %%
# Combine preparation, gate, and measurement gadgets.
# [START_QEC_ARCHITECTURE]
steane_architecture = cql.devices.QECArchitecture(
    "SteaneBellArchitecture",
    Steane.default_encoding,
    link_roots=(
        steane_prepare_zero,
        steane_h,
        steane_cx,
        cql.gadgets.measure_z(Steane),
    ),
)
# [END_QEC_ARCHITECTURE]

# %%
# Connect the logical machine to the QEC architecture.
# [START_LOGICAL_DEVICE]
builder = cql.devices.DeviceBuilder("SteaneBellDevice")
compute = builder.logical.add_compute(capacity=2)
encoded_region = builder.qec.bind(compute, architecture=steane_architecture)
# [END_LOGICAL_DEVICE]

# %%
# Bind physical resources and an illustrative operating point.
# [START_PHYSICAL_DEVICE]
carriers = builder.physical.add_qubits(
    2 * Steane.block.size,
    native_actions=cql.architecture.physical_actions.clifford_set(),
    native_instruments=(
        cql.architecture.physical_instruments.MZ,
        cql.architecture.physical_instruments.MPP,
    ),
)
builder.physical.bind(encoded_region, to=carriers)
builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
device = builder.build()
# [END_PHYSICAL_DEVICE]

# %%
# Expose the device as a CUDA-Q target, with the estimate options that it and
# the explicit estimates below share.
# [START_CUDA_Q_TARGET]
estimate_options = {
    "p_phys": 1.0e-3,
    "failure_budget": 0.1,
    "cycle_time": 1.0e-9,
}
target = cql.targets.Target.from_device(
    "steane_bell",
    device,
    estimate_options=estimate_options,
)
# [END_CUDA_Q_TARGET]

# %%
# Compile the kernel for CUDA-Q Logical and import it as P0.
# [START_KERNEL_IMPORT]
logical = cql.compiler.import_cudaq(bell_pair)
# [END_KERNEL_IMPORT]

# %%
# Count the logical resources of the P0 build.
# [START_LOGICAL_ESTIMATE]
logical_resources = cql.estimate(logical, tier=cql.estimate.Tier.LOGICAL)

print(f"Peak logical qubits: {logical_resources.logical_qubits_peak}")
print(f"Hadamard actions: {logical_resources.actions['qlx_standard_h']}")
print(f"Controlled-X actions: {logical_resources.actions['qlx_standard_cx']}")
# [END_LOGICAL_ESTIMATE]

# %%
# Place the logical qubits on the device.
# [START_PLACEMENT]
placed = cql.compiler.place(logical, device=device)

print([(binding.space, binding.slot) for binding in placed.placement.bindings])
# [END_PLACEMENT]

# %%
# Compile P2 and count encoded resources.
# [START_STATIC_ESTIMATE]
encoded = cql.compile(
    placed,
    pipeline=cql.compiler.pipelines.qec(),
    device=device,
)
static_resources = cql.estimate(encoded, tier=cql.estimate.Tier.STATIC)

print(f"Peak encoded patches: {static_resources.patches_peak}")
print(f"Hadamard gadget calls: {static_resources.gadget_calls['steane_h']}")
print(
    f"Controlled-X gadget calls: {static_resources.gadget_calls['steane_cx']}")
# [END_STATIC_ESTIMATE]

# %%
# Compile and schedule the P3 physical events.
# [START_PHYSICAL_SCHEDULE]
physical = cql.compile(
    encoded,
    pipeline=cql.compiler.pipelines.physical(),
    device=device,
)
schedule = cql.compiler.schedule(physical)
# [END_PHYSICAL_SCHEDULE]

# %%
# Estimate the physical resources and timing.
# [START_SCHEDULE_ESTIMATE]
schedule_resources = cql.estimate(
    schedule,
    tier=cql.estimate.Tier.SCHEDULE,
    **estimate_options,
)

print(f"Physical qubits: {schedule_resources.physical_qubits}")
print(f"Scheduled events: {schedule_resources.event_count}")
print(f"Makespan: {schedule_resources.makespan_ns:.1f} ns")
# [END_SCHEDULE_ESTIMATE]

# %%
# Estimate the kernel through the target and verify agreement with the
# explicit workflow.
# [START_KERNEL_ESTIMATE]
cudaq.set_target(target)
estimate = cudaq.estimate(bell_pair)
kernel_logical = cql.estimate.LogicalEstimate.from_annotations(
    estimate.annotations)
kernel_static = cql.estimate.FabricCounts.from_annotations(estimate.annotations)
kernel_schedule = estimate.annotations["SCHEDULE"]

assert kernel_logical.logical_qubits_peak == logical_resources.logical_qubits_peak
assert kernel_logical.actions == logical_resources.actions
assert kernel_static.patches_peak == static_resources.patches_peak
assert kernel_static.gadget_calls == static_resources.gadget_calls
assert kernel_schedule["physical_qubits"] == schedule_resources.physical_qubits
assert kernel_schedule["event_count"] == schedule_resources.event_count
assert kernel_schedule["makespan_ns"] == schedule_resources.makespan_ns
# [END_KERNEL_ESTIMATE]

# %%
# Run the kernel and its P3 Build. Both circuits are Clifford, so Stim runs
# them.
# [START_CUDA_Q_RUN]
import collections

cudaq.set_target("stim")
for name, runnable in {
        "CUDA-Q kernel": bell_pair,
        "P3 Build": physical
}.items():
    results = cudaq.run(runnable, shots_count=100)
    counts = collections.Counter(tuple(shot) for shot in results)
    assert set(counts) == {(False, False), (True, True)}
    print(f"{name}:\toutcomes\t{dict(counts)}")
# [END_CUDA_Q_RUN]

# %%
# Optional: inspect the Quake that the P3 Build runs. The kernel runs on two
# qubits. That Quake allocates all 13 carriers of each block, including the 6
# check ancillas this program never uses, plus one ancilla for each logical-Z
# measurement. The schedule counts that measurement as one native
# Pauli-product event, so it reports only the 26 carriers.
# [START_QUAKE_INSPECTION]
kernel_resources = cudaq.estimate_resources(bell_pair)
print(f"CUDA-Q kernel: {kernel_resources.num_qubits} qubits, "
      f"{kernel_resources.count('h')} h, "
      f"{kernel_resources.count_controls('x', 1)} cx, "
      f"{kernel_resources.count('mz')} mz")
quake = physical.to_quake()
entry = next(op for op in quake.body.operations
             if "cudaq-entrypoint" in op.operation.attributes)
ops = collections.Counter((op.operation.name, len(op.operation.operands))
                          for op in entry.regions[0].blocks[0].operations)
print(f"P3 Build: {ops['quake.null_wire', 0]} qubits, "
      f"{ops['quake.h', 1]} h, {ops['quake.x', 2]} cx, "
      f"{ops['quake.reset', 1]} reset, {ops['quake.mz', 1]} mz")
# [END_QUAKE_INSPECTION]
