# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Evaluate 15-to-1 T-state distillation with `cudaq.run`.

Each shot runs one distillation attempt on BareQubit patches (one physical
qubit per patch). The attempt uses the rotation supports, final S, and X checks
of the library's 15-to-1 protocol (example 07). It returns its four check bits
and a check of its output, and the caller accepts an attempt when all four
checks are False.

Example 07's protocol requests 15 raw T states from an external stream. It
unpacks four as check rows and consumes eleven with resource-backed rotations.
Logical specifies the ideal operation of such a rotation, but it has no
concrete preparation for an external raw T state and no execution lowering for
the rotation, so that protocol estimates resources but does not run. This
example supplies both explicitly: a raw T state is T|+>, and each rotation is
gate teleportation that consumes one raw T state, the way `steane_t_injection`
consumes a T state. `cudaq.run` also rejects resource flows and
`cql.postselect`, so the caller post-selects in Python after the run.

The noisy run puts a Z error on each raw T state with probability p, and
nothing else. Distillation then rejects about half the attempts, and the
accepted outputs fail far less often than the raw states.
"""

# %%
# Gadgets on BareQubit patches, and a device with six of them: five rows of the
# circuit and one raw T state at a time.
import cudaq
import cudaq.logical as cql

CODE = cql.codes.BareQubit


@cql.gadget(implements=cql.std.t)
def bare_t(q: cql.patch[CODE]) -> cql.patch[CODE]:
    return cql.ops.t(q.data)


@cql.gadget(implements=cql.std.tdg)
def bare_tdg(q: cql.patch[CODE]) -> cql.patch[CODE]:
    return cql.ops.tdg(q.data)


@cql.gadget(implements=cql.std.h)
def bare_h(q: cql.patch[CODE]) -> cql.patch[CODE]:
    return cql.ops.h(q.data)


@cql.gadget(implements=cql.std.cx)
def bare_cx(a: cql.patch[CODE],
            b: cql.patch[CODE]) -> tuple[cql.patch[CODE], cql.patch[CODE]]:
    return cql.ops.cx(a.data, b.data)


measure_z = cql.gadgets.measure_z(CODE)


def make_device():
    actions = cql.architecture.physical_actions
    instruments = cql.architecture.physical_instruments
    architecture = cql.devices.QECArchitecture(
        "t_factory_architecture",
        CODE.default_encoding,
        link_roots=(cql.gadgets.prepare_plus(CODE), measure_z,
                    cql.protocols.bare_measure_x, cql.protocols.bare_s, bare_t,
                    bare_tdg, bare_h, bare_cx),
    )
    builder = cql.devices.DeviceBuilder("t_factory_device", source_module=None)
    compute = builder.qec.bind(builder.logical.add_compute(capacity=6),
                               architecture=architecture)
    builder.physical.bind(compute,
                          to=builder.physical.add_qubits(
                              6,
                              native_actions=(*actions.clifford_set(),
                                              actions.T, actions.TDG),
                              native_instruments=(instruments.MZ,
                                                  instruments.MX)))
    builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
    return builder.build()


# %%
# The distillation attempt.
def raw_t_state():
    """T|+> on a fresh patch: the input of the factory."""
    return bare_t(
        cql.prepare_plus(cql.ops.allocate_patch(CODE, region="compute")))


def rotate(rows, support):
    """Applies exp(-i pi/8 Z...Z) on `support` by gate teleportation.

    This is the library's rotation by the angle pi/4. A CX ladder collects the
    parity of `support` on its last qubit. A CX from that qubit onto a raw T
    state and a Z measurement of the raw state leave T on the parity, or
    T-dagger on outcome 1, which S corrects. The ladder is then undone.
    """
    *rest, last = support
    for q in rest:
        rows[q], rows[last] = bare_cx(rows[q], rows[last])
    rows[last], raw = bare_cx(rows[last], raw_t_state())
    rows[last], = cql.ops.cond(measure_z(raw),
                               carries=(rows[last],),
                               then=lambda q: (cql.protocols.bare_s(q),),
                               else_=lambda q: (q,))
    for q in reversed(rest):
        rows[q], rows[last] = bare_cx(rows[q], rows[last])
    return rows


def is_not_t(q):
    """False when `q` holds |T>: T-dagger, then H, then a Z measurement."""
    return measure_z(bare_h(bare_tdg(q)))


@cql.protocol
def distillation_attempt() -> tuple[bool, bool, bool, bool, bool]:
    # Rows 0 to 3 are the check qubits, each a raw T state. Row 4 is the output.
    rows = [raw_t_state() for _ in range(4)]
    rows.append(cql.prepare_plus(cql.ops.allocate_patch(CODE,
                                                        region="compute")))
    for support in cql.protocols.FIFTEEN_TO_ONE_ROTATION_SUPPORTS:
        rows = rotate(rows, support)
    # The rotations leave T-dagger|+> on the output row, and S turns it into
    # |T>, as in the library protocol.
    rows[4] = cql.protocols.bare_s(rows[4])
    checks = [cql.protocols.bare_measure_x(q) for q in rows[:4]]
    return (*checks, is_not_t(rows[4]))


@cql.protocol
def raw_state_check() -> bool:
    return is_not_t(raw_t_state())


# %%
# Compile both protocols to physical P3 for the device.
def physical(protocol, device):
    return cql.compile(cql.compile(protocol),
                       pipeline=cql.compiler.pipelines.physical(),
                       device=device)


device = make_device()
attempt = physical(distillation_attempt, device)
raw_check = physical(raw_state_check, device)

# %%
# Run on the GPU simulator when a GPU is present, and otherwise on the
# density-matrix simulator. Both apply the noise model below. Without noise,
# every attempt passes its checks and produces |T>.
target = "nvidia" if cudaq.num_available_gpus() > 0 else "density-matrix-cpu"
cudaq.set_target(target)
assert cudaq.run(attempt, shots_count=100) == [[False] * 5] * 100

# %%
# With a Z error on each raw T state with probability p, distillation rejects
# about half the attempts, and the accepted outputs fail less often.
p, shots = 0.05, 1000
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("t", cudaq.PhaseFlipChannel(p))
raw_failures = sum(cudaq.run(raw_check, shots_count=shots, noise_model=noise))
results = cudaq.run(attempt, shots_count=shots, noise_model=noise)
accepted = [result[4] for result in results if not any(result[:4])]
model = cql.protocols.DISTILL_15TO1_T
print(f"wrong raw T states: {raw_failures} of {shots} (model rate {p})")
print(f"accepted attempts:  {len(accepted)} of {shots} "
      f"(model rate {model.acceptance_probability(p):.3f})")
print(f"wrong outputs:      {sum(accepted)} of {len(accepted)} accepted "
      f"(model rate {model.output_error(p):.4f})")
