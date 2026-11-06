# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile a Steane Bell pair directly and through a CUDA-Q target.

This runnable example supplies the excerpts in cudaq-logical-in-practice.md.
"""

# %%
# Author the portable Bell-pair program.
# [START_LOGICAL_PROGRAM]
import cudaq.logical as cql


@cql.program
def bell() -> tuple[bool, bool]:
    qubits = cql.allocate(2, state=cql.types.zero, name="data")
    qubits[0] = cql.h(qubits[0])
    qubits[0], qubits[1] = cql.cx(qubits[0], qubits[1])
    return cql.measure_z(qubits[0]), cql.measure_z(qubits[1])


# [END_LOGICAL_PROGRAM]

# %%
# Compile P0 and count logical resources.
# [START_LOGICAL_ESTIMATE]
logical = cql.compile(bell)
logical_resources = cql.estimate(logical, tier=cql.estimate.Tier.LOGICAL)

print(f"Peak logical qubits: {logical_resources.logical_qubits_peak}")
print(f"Hadamard actions: {logical_resources.actions['qlx_standard_h']}")
print(f"Controlled-X actions: {logical_resources.actions['qlx_standard_cx']}")
# [END_LOGICAL_ESTIMATE]


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
# Combine preparation, gate, and measurement gadgets.
# [START_QEC_ARCHITECTURE]
steane_architecture = cql.devices.QECArchitecture(
    "SteaneBellArchitecture",
    Steane.default_encoding,
    link_roots=(
        cql.gadgets.prepare_zero(Steane),
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
# Place the logical allocation on the device.
# [START_PLACEMENT]
placed = cql.compiler.place(
    logical,
    device=device,
    placement=(cql.architecture.colocate(logical.values.data),),
)

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
estimate_options = {
    "p_phys": 1.0e-3,
    "failure_budget": 0.1,
    "cycle_time": 1.0e-9,
}
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
# Author the same workload as a CUDA-Q kernel.
# [START_CUDA_Q_KERNEL]
import cudaq


@cudaq.kernel
def bell_pair():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


# [END_CUDA_Q_KERNEL]

# %%
# Expose the device as a CUDA-Q estimation target.
# [START_CUDA_Q_TARGET]
target = cql.targets.Target.from_device(
    "steane_bell",
    device,
    estimate_options=estimate_options,
)
cudaq.set_target(target)
# [END_CUDA_Q_TARGET]

# %%
# Estimate the kernel and verify agreement with the direct workflow.
# [START_KERNEL_ESTIMATE]
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
# Compile the kernel for the target without estimating it, and import it as
# the P0 build that the direct workflow started from.
# [START_KERNEL_IMPORT]
module = cudaq.compile(bell_pair, target=target)
kernel_p0 = cql.compiler.import_quake(str(module.mlir_module))

assert kernel_p0.stage == cql.stages.P0
kernel_p0_resources = cql.estimate(kernel_p0, tier=cql.estimate.Tier.LOGICAL)
assert kernel_p0_resources.actions == logical_resources.actions
# [END_KERNEL_IMPORT]

# %%
# Save the builds behind the estimate, and load the physical one back.
# [START_SAVE_TEMPS]
import tempfile

with tempfile.TemporaryDirectory() as directory:
    saved = cudaq.estimate(bell_pair, save_temps=directory)
    print("Saved builds:", ", ".join(sorted(saved.annotations["BUILDS"])))
    kernel_physical = cql.load_build(saved, stage="p3")

# The physical build declares one resource for every physical qubit.
physical_mlir = kernel_physical.to_mlir()
physical_resources = physical_mlir.count("phys.resource @")
assert physical_resources == schedule_resources.physical_qubits
# [END_SAVE_TEMPS]
