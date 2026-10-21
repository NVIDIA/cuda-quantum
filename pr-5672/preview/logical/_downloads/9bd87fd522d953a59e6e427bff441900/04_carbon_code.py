# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Build an end-to-end Carbon target with kernel-backed gadgets from scratch.

The ``[[12,2,4]]`` code data and paired logical operations are based on the
Carbon code presentation in arXiv:2404.02280. The final physical binding is an
explicit teaching model: it reserves the code's 22 carriers per block and
assigns illustrative native-operation timing; it is not a hardware claim.
"""

# %%
# Import CUDA-Q and the CUDA-Q Logical code, gadget, and target APIs.
import cudaq
import cudaq.logical as cql


# %%
# Define Carbon's CSS checks and two protected logical ports.
@cql.code
class Carbon:
    """The self-dual ``[[12,2,4]]`` Carbon CSS code."""

    block = cql.codes.CSSBlock(data=12, sx=5, sz=5)
    d = 4
    hx = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (0, 1, 5, 7, 8, 11),
        (0, 3, 4, 5, 9, 11),
    )
    hz = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (0, 2, 6, 7, 8, 11),
        (0, 3, 4, 6, 10, 11),
    )
    lx = ((0, 3, 10, 11), (1, 3, 9, 10))
    lz = ((0, 3, 9, 11), (0, 1, 9, 10))


CARBON_H_SWAPS = ((1, 2), (4, 6), (4, 7), (4, 5), (8, 11))


# %%
# Express the paired ideal operations as ordinary CUDA-Q kernels.
@cudaq.kernel
def paired_h(left: cudaq.qubit, right: cudaq.qubit):
    h(left)
    h(right)


@cudaq.kernel
def paired_cx(
    control_left: cudaq.qubit,
    control_right: cudaq.qubit,
    target_left: cudaq.qubit,
    target_right: cudaq.qubit,
):
    x.ctrl(control_left, target_left)
    x.ctrl(control_right, target_right)


# %%
# Implement the paired operations on Carbon patches and retain their kernels.
@cql.gadget(
    implements=paired_h,
    logical_ports={
        "left": "block.q0",
        "right": "block.q1",
    },
)
def paired_h_gadget(block: cql.patch[Carbon]) -> cql.patch[Carbon]:
    block = cql.h(block.data)
    for left, right in CARBON_H_SWAPS:
        block = cql.cx(block.data, block.data, pairs=((left, right),))
        block = cql.cx(block.data, block.data, pairs=((right, left),))
        block = cql.cx(block.data, block.data, pairs=((left, right),))
    return block


@cql.gadget(
    implements=paired_cx,
    logical_ports={
        "control_left": "control.q0",
        "control_right": "control.q1",
        "target_left": "target.q0",
        "target_right": "target.q1",
    },
)
def paired_cx_gadget(
    control: cql.patch[Carbon],
    target: cql.patch[Carbon],
) -> tuple[cql.patch[Carbon], cql.patch[Carbon]]:
    return cql.cx(control.data, target.data)


# %%
# Build the entire software stack: logical capacity, Carbon QEC, and physical machine.
carbon_architecture = cql.devices.QECArchitecture(
    "CarbonArchitecture",
    Carbon.default_encoding,
    link_roots=(paired_h_gadget, paired_cx_gadget),
)

builder = cql.devices.DeviceBuilder("CarbonTeachingDevice")
compute = builder.logical.add_compute(capacity=4)
encoded = builder.qec.bind(compute, architecture=carbon_architecture)
carriers = builder.physical.add_qubits(
    2 * Carbon.block.size,
    native_actions=cql.architecture.physical_actions.clifford_set(),
    native_instruments=(
        cql.architecture.physical_instruments.MX,
        cql.architecture.physical_instruments.MZ,
        cql.architecture.physical_instruments.MPP,
    ),
)
builder.physical.bind(encoded, to=carriers)
builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
carbon_device = builder.build()

carbon_target = cql.targets.Target.from_device(
    "carbon",
    carbon_device,
    runtime_backend=cql.targets.estimator,
    estimate_options={
        "p_phys": 1.0e-3,
        "failure_budget": 0.1,
        "cycle_time": 1.0e-9,
    },
)
cudaq.set_target(carbon_target)

# %%
# Call the declarations exported by the gadgets from a larger CUDA-Q kernel.
paired_h_call = paired_h_gadget.kernel
paired_cx_call = paired_cx_gadget.kernel


@cudaq.kernel
def logical_bell_pairs():
    left_control = cudaq.qubit()
    right_control = cudaq.qubit()
    left_target = cudaq.qubit()
    right_target = cudaq.qubit()
    paired_h_call(left_control, right_control)
    paired_cx_call(left_control, right_control, left_target, right_target)


# %%
# Estimate through the complete Carbon target and inspect each resource layer.
estimate = cudaq.estimate(logical_bell_pairs)
logical = cql.estimate.LogicalEstimate.from_annotations(estimate.annotations)
static = cql.estimate.FabricCounts.from_annotations(estimate.annotations)
schedule = estimate.annotations["SCHEDULE"]

assert (Carbon.n, Carbon.k, Carbon.d.value) == (12, 2, 4)
assert logical.logical_qubits_peak == 4
assert logical.actions["paired_h"] == 1
assert logical.actions["paired_cx"] == 1
assert static.patches_peak == 2
assert static.gadget_calls["paired_h_gadget"] == 1
assert static.gadget_calls["paired_cx_gadget"] == 1
assert schedule["physical_qubits"] == 2 * Carbon.block.size

print("Carbon [[12,2,4]] target resources:")
print(f"  protected logical qubits per block: {Carbon.k}")
print(f"  reserved carriers per block: {Carbon.block.size}")
print(f"  selected gadgets: {dict(static.gadget_calls)}")
print(f"  physical qubits: {schedule['physical_qubits']}")
print(f"  scheduled events: {schedule['event_count']}")
print(f"  makespan: {schedule['makespan_ns']:.1f} ns")
