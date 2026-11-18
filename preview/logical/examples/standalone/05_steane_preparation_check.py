# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check a Steane encoded-zero preparation, with and without noise.

Prepare one [[7,1,3]] block, extract its six stabilizers, then measure its
seven data carriers in Z. Compare noiseless execution with depolarizing gate
noise on Stim.

The block uses seven data and six syndrome carriers. Each shot returns 14
Booleans: three X checks, three Z checks, seven physical data results, and
logical Z (their XOR). False means eigenvalue +1 for a check. The raw data
bits can vary even for encoded zero.

The encoder and syndrome circuits are not fault-tolerant preparation
protocols. No error correction, decoding, or postselection is applied. Reset
and readout are ideal in the noise model. Decoding is out of scope; see the
last cell for where to go next.

Requires CUDA-Q with `cudaq.logical` and the Stim target installed.
"""

# %%
import cudaq
import cudaq.logical as cql

CODE = cql.codes.Steane
SHOTS = 100
NOISE_PROBABILITY = 0.001


# %%
# Prepare encoded zero. Reset alone does not prepare the code state; the H and
# CX gates below implement the encoder.
#
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
# Extract the six stabilizers without measuring the data carriers. X checks
# use `q.sx[i]` and Z checks use `q.sz[i]`. The gadget returns the live block
# followed by the three X-check and three Z-check results, in that order.
#
# The logical action describes ideal code-space behavior: a stabilizer check
# preserves the qubit and returns False. The gadget implements the checks
# explicitly with physical operations.
@cql.action(kind="instrument")
def syndrome_checks(
    q: cql.types.logical_qubit
) -> tuple[cql.types.logical_qubit, *((bool,) * 6)]:
    return (q, False, False, False, False, False, False)


@cql.gadget(implements=syndrome_checks)
def extract_syndrome(
        q: cql.patch[CODE]) -> tuple[cql.patch[CODE], *((bool,) * 6)]:
    checks = []
    for check_index, support in enumerate(CODE.hx):
        q = cql.ops.reset(q.sx[check_index])
        q = cql.ops.h(q.sx[check_index])
        for index in support:
            q = cql.ops.cx(q.sx[check_index], q.data[index])
        q = cql.ops.h(q.sx[check_index])
        q, bits = cql.ops.mz(q.sx[check_index])
        checks.append(cql.ops.parity(bits))
    for check_index, support in enumerate(CODE.hz):
        q = cql.ops.reset(q.sz[check_index])
        for index in support:
            q = cql.ops.cx(q.data[index], q.sz[check_index])
        q, bits = cql.ops.mz(q.sz[check_index])
        checks.append(cql.ops.parity(bits))
    return (q, *checks)


# %%
# Destructive readout returns the seven individual physical Z measurements.
@cql.gadget(implements=cql.std.LogicalInstrumentRef("readout_steane_z", 1, 7))
def measure_data_z(q: cql.patch[CODE]) -> tuple[(bool,) * 7]:
    data_bits = []
    for index in range(CODE.n):
        q, bits = cql.ops.mz(q.data[index])
        data_bits.append(cql.ops.parity(bits))
    cql.discard(q)
    return tuple(data_bits)


@cql.protocol
def preparation_check() -> tuple[(bool,) * 14]:
    q = allocate_zero()
    q, *checks = extract_syndrome(q)
    data_bits = measure_data_z(q)
    logical_z = data_bits[0]
    for bit in data_bits[1:]:
        logical_z = cql.ops.xor(logical_z, bit)
    return (*checks, *data_bits, logical_z)


# %%
# Map one logical block to seven data and six syndrome carriers.
def make_device():
    blocks = 1
    architecture = cql.devices.QECArchitecture(
        "preparation_steane_architecture",
        CODE.default_encoding,
        link_roots=(encoded_zero, extract_syndrome, measure_data_z),
    )
    builder = cql.devices.DeviceBuilder("preparation_steane_device",
                                        source_module=None)
    compute = builder.qec.bind(builder.logical.add_compute(capacity=blocks),
                               architecture=architecture)
    actions = cql.architecture.physical_actions
    instruments = cql.architecture.physical_instruments
    builder.physical.bind(compute,
                          to=builder.physical.add_qubits(
                              blocks * CODE.block.size,
                              native_actions=actions.clifford_set(),
                              native_instruments=(instruments.MZ,)))
    builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
    return builder.build()


def physical(protocol, device):
    return cql.compile(cql.compile(protocol),
                       pipeline=cql.compiler.pipelines.physical(),
                       device=device)


# %%
# Each shot is (X checks[3], Z checks[3], data bits[7], logical Z).
def split_shot(shot):
    return shot[:6], shot[6:13], shot[13]


def print_checks(checks, show=5):
    for i, shot in enumerate(checks[:show]):
        stabilizers, data_bits, logical_z = split_shot(shot)
        print(f"Shot {i + 1}: stabilizers={stabilizers}, "
              f"data_bits={data_bits}, logical_z={logical_z}")
    logical_zeros = sum(not shot[13] for shot in checks)
    print(f"(showing {min(show, len(checks))} of {len(checks)} shots)")
    print(f"Logical fidelity (logical_z=0): {logical_zeros}/{len(checks)} "
          f"({logical_zeros / len(checks):.2%})")


# %%
# Noiseless run. Every stabilizer should be False and logical Z should be 0.
device = make_device()
preparation = physical(preparation_check, device)

cudaq.set_target("stim")
checks = cudaq.run(preparation, shots_count=SHOTS)
print_checks(checks)
assert all(not any(split_shot(shot)[0]) and not shot[13] for shot in checks)


# %%
# Noisy run. Apply depolarizing noise after physical H and X gates and
# two-qubit depolarizing noise after CX gates. Adjust NOISE_PROBABILITY and
# SHOTS above. A nonzero stabilizer outcome flags a check; logical Z of 1
# records failure of the uncorrected readout. These events can overlap.
def build_noise():
    noise = cudaq.NoiseModel()
    for gate in ("h", "x"):
        noise.add_all_qubit_channel(
            gate, cudaq.DepolarizationChannel(NOISE_PROBABILITY))
    noise.add_all_qubit_channel("x",
                                cudaq.Depolarization2(NOISE_PROBABILITY),
                                num_controls=1)
    return noise


noisy_checks = cudaq.run(preparation,
                         shots_count=SHOTS,
                         noise_model=build_noise())
print_checks(noisy_checks)


# %%
# Read a syndrome (illustration only, not a decoder).
#
# The X checks flag Z errors and the Z checks flag X errors. Steane is a CSS
# code whose three check supports are the same for both types, so each data
# qubit has a distinct 3-bit signature: the checks that contain it. If a
# single data error caused a nonzero syndrome, it matches exactly one qubit.
#
# This table assumes one data error and ignores faults in the syndrome
# circuit itself, which real decoders must model. It says nothing about
# repeated rounds or correlated faults.
def locate_single_error(syndrome, supports):
    syndrome = tuple(syndrome)
    for qubit in range(CODE.n):
        if tuple(qubit in support for support in supports) == syndrome:
            return qubit
    return None


flagged = 0
for shot in noisy_checks:
    stabilizers, _, _ = split_shot(shot)
    x_syndrome, z_syndrome = stabilizers[:3], stabilizers[3:]
    if any(x_syndrome):
        flagged += 1
        print(f"X checks {x_syndrome}: Z error on data qubit "
              f"{locate_single_error(x_syndrome, CODE.hx)} if single")
    if any(z_syndrome):
        flagged += 1
        print(f"Z checks {z_syndrome}: X error on data qubit "
              f"{locate_single_error(z_syndrome, CODE.hz)} if single")
print(f"{flagged} nonzero syndrome(s) in {len(noisy_checks)} noisy shots")

# %%
# Next steps: decoding.
#
# This example stops at raw measurements. Identifying the most likely error
# from a syndrome, and applying or tracking the correction, is the job of a
# decoder. Two routes from here:
#
# * Export a detector error model. The `EDEM` workflow turns a gadget's fault
#   model into a file whose detectors and observables a decoder consumes. See
#   `docs/source/use-cases/gadget-edems.md` in this repository.
# * Use CUDA-Q QEC directly for decoders (including GPU and real-time
#   decoders) and numerical experiments:
#   https://nvidia.github.io/cudaq-qec/components/qec/index.html
#
# The parity-check supports used above, CODE.hx and CODE.hz, are the same
# information a decoder needs about the code.
