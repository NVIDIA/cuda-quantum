# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Recover a hidden bit string with Steane-encoded Bernstein-Vazirani.

Each input and the oracle ancilla occupy a [[7,1,3]] code block. With the
32-bit secret below, the circuit has 231 data carriers. The device also
reserves six syndrome carriers per block, for 429 carriers in total.
All gates are Clifford, so the circuit can run on the Stim simulator.

This is ideal encoded computation. The encoder is not a fault-tolerant
preparation protocol, and the example does not perform noisy error correction
or decoding. The preparation check verifies encoded |0> before the run.
"""

# %%
# Reuse the library's transversal gates and logical measurements.
import cudaq
import cudaq.logical as cql

CODE = cql.codes.Steane
SECRET = "10110010100110101100101100010110"
SHOTS = 32
logical_h = cql.qec.steane.transversal_h
logical_cx = cql.qec.steane.transversal_cx
logical_x = cql.gadgets.logical_pauli(CODE, basis="x")
measure_z = cql.gadgets.measure_z(CODE)


# %%
# The X-check rows have unique pivots at data carriers 3, 5, and 6. Three H
# gates and nine CX gates prepare the equal positive superposition over their
# row space. The rows are mutually orthogonal and even weight, so Z checks and
# logical Z are +1.
@cql.gadget(implements=cql.std.prepare_zero)
def encoded_zero(q: cql.patch[CODE]) -> cql.patch[CODE]:
    q = cql.ops.reset(q.data)
    for index in (3, 5, 6):
        q = cql.ops.h(q.data[index])
    for control, targets in ((3, (0, 1, 2)), (5, (0, 1, 4)), (6, (0, 2, 4))):
        for target in targets:
            q = cql.ops.cx(q.data[control], q.data[target])
    return q


def allocate_zero():
    return encoded_zero(cql.ops.allocate_patch(CODE, region="compute"))


# %%
# Check all six stabilizers and logical Z on a separate prepared block. The
# syndrome carrier measures each stabilizer as a whole product, and `cql.mpp`
# measures logical Z. No data carrier is measured individually. A False
# outcome means eigenvalue +1.
@cql.gadget(implements=cql.std.LogicalInstrumentRef("check_steane_zero", 1, 7))
def check_zero(
        q: cql.patch[CODE]) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    checks = []
    for basis, supports in (("X", CODE.hx), ("Z", CODE.hz)):
        for support in supports:
            q = cql.ops.reset(q.sx[0])
            if basis == "X":
                q = cql.ops.h(q.sx[0])
            for index in support:
                if basis == "X":
                    q = cql.ops.cx(q.sx[0], q.data[index])
                else:
                    q = cql.ops.cx(q.data[index], q.sx[0])
            if basis == "X":
                q = cql.ops.h(q.sx[0])
            q, bits = cql.ops.mz(q.sx[0])
            checks.append(cql.ops.parity(bits))
    q, logical_z = cql.mpp(cql.types.Z(q[0]))
    cql.discard(q)
    return (*checks, logical_z)


@cql.protocol
def preparation_check() -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    return check_zero(allocate_zero())


# %%
# The oracle is |x,y> -> |x,y XOR (SECRET dot x)>. Its encoded |-> ancilla
# turns each selected CX into a phase on the corresponding input. A final H
# on every input turns those phases into the secret's bits.
@cql.protocol
def bernstein_vazirani() -> tuple[(bool,) * len(SECRET)]:
    inputs = [logical_h(allocate_zero()) for _ in SECRET]
    ancilla = logical_h(logical_x(allocate_zero()))
    for index, bit in enumerate(SECRET):
        if bit == "1":
            inputs[index], ancilla = logical_cx(inputs[index], ancilla)
    cql.discard(ancilla)
    return tuple(measure_z(logical_h(q)) for q in inputs)


# %%
# Map each logical block to seven data and six syndrome carriers. The block
# count includes the oracle ancilla. The allocate_zero helper applies the
# explicit encoder, rather than assuming a reset produces encoded |0>.
def make_device():
    blocks = len(SECRET) + 1
    architecture = cql.devices.QECArchitecture(
        "bv_steane_architecture",
        CODE.default_encoding,
        link_roots=(encoded_zero, logical_h, logical_cx, logical_x, measure_z),
    )
    builder = cql.devices.DeviceBuilder("bv_steane_device", source_module=None)
    compute = builder.qec.bind(builder.logical.add_compute(capacity=blocks),
                               architecture=architecture)
    actions = cql.architecture.physical_actions
    instruments = cql.architecture.physical_instruments
    builder.physical.bind(compute,
                          to=builder.physical.add_qubits(
                              blocks * CODE.block.size,
                              native_actions=actions.clifford_set(),
                              native_instruments=(instruments.MZ,
                                                  instruments.MX,
                                                  instruments.MPP)))
    builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
    return builder.build()


def physical(protocol, device):
    return cql.compile(cql.compile(protocol),
                       pipeline=cql.compiler.pipelines.physical(),
                       device=device)


# %%
# cudaq.run translates the P3 Build to Quake and executes it on Stim. Only the
# explicit return becomes output: one list of Booleans per shot, ordered from
# the first input block to the last, matching SECRET from left to right.
device = make_device()
cudaq.set_target("stim")
checks = cudaq.run(physical(preparation_check, device), shots_count=SHOTS)
assert checks == [[False] * 7] * SHOTS
print(f"encoded preparation: all seven checks passed in {SHOTS} shots")

results = cudaq.run(physical(bernstein_vazirani, device), shots_count=SHOTS)
assert results == [[bit == "1" for bit in SECRET]] * SHOTS
recovered = "".join(str(int(bit)) for bit in results[0])
print(f"recovered secret: {recovered} in all {SHOTS} shots")
