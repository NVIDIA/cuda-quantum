# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""`cudaq.run` on P3 physical graphs, through the `phys-to-quake` pass."""

import gc
import math

import pytest

import cudaq
import cudaq.logical as cql
import cudaq.mlir.ir as mlir_ir
from cudaq.logical import codes, gadgets, std
from cudaq.logical.compiler import Build, PhysicalBuilder
from cudaq.logical.gadgets import gadget, patch
from cudaq.logical.ops import _impl as ops
from cudaq.logical.programs import DefinitionHandle
from cudaq.mlir._mlir_libs import _qlxRuntime

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# Every test runs on CUDA-Q's default target unless it names its targets. A
# test names targets only when the simulator can change its outcome: gate
# support, noise, or memory. `SHOTS` is the default shot count.
# `RANDOM_SHOTS` is for a check that both values of a random bit appear.
# ---------------------------------------------------------------------------

skipIfNoGPU = pytest.mark.skipif(cudaq.num_available_gpus() == 0,
                                 reason="the `nvidia` target needs a GPU")

TARGETS = [
    pytest.param(("qpp-cpu", {}), id="qpp-cpu"),
    pytest.param(("density-matrix-cpu", {}), id="density-matrix-cpu"),
    pytest.param(("stim", {}), id="stim"),
    pytest.param(("nvidia", {}), id="nvidia-fp32", marks=skipIfNoGPU),
    pytest.param(("nvidia", dict(option="fp64")),
                 id="nvidia-fp64",
                 marks=skipIfNoGPU),
]


@pytest.fixture(autouse=True)
def target(request):
    # Only a test that names its targets passes one in.
    if hasattr(request, "param"):
        name, options = request.param
        cudaq.set_target(name, **options)
    else:
        cudaq.reset_target()
        name, options = cudaq.get_target().name, {}
    try:
        yield name, options
    finally:
        cudaq.reset_target()


SHOTS = 10
RANDOM_SHOTS = 32

# ---------------------------------------------------------------------------
# Program sources
# ---------------------------------------------------------------------------
# The tests get P3 in three ways:
# - `@cql.physical` and `PhysicalBuilder` write P3 directly for `machine`.
# - `physical_build()` compiles a `@cql.program` or `@cql.protocol` to P3
#   through the native projector, for `DEVICE`.
# - `P3Fixture` parses hand-written P3 for forms that the other two sources
#   do not produce.
# ---------------------------------------------------------------------------

arch = cql.architecture
actions = arch.physical_actions
instruments = arch.physical_instruments
machine = arch.PhysicalMachine(
    "run_physical_machine",
    resource_classes={
        "qubits":
            arch.ResourceClass("qubit",
                               16,
                               native_actions=(*actions.clifford_set(),
                                               actions.T),
                               native_instruments=(instruments.MZ,
                                                   instruments.MX))
    })


def fresh(count=1, state="zero"):
    return cql.ops.load(cql.ops.acquire(machine.qubits, count=count),
                        state=state)


def flip(state):
    return cql.ops.apply(actions.X, state)


def physical(definition):
    return lambda: cql.compile(definition)


BARE = codes.BareQubit
X, Y, Z = cql.types.X, cql.types.Y, cql.types.Z


@gadget(implements=std.x)
def bare_x(q: patch[BARE]) -> patch[BARE]:
    return ops.x(q.data)


@gadget(implements=std.h)
def bare_h(q: patch[BARE]) -> patch[BARE]:
    return ops.h(q.data)


@gadget(implements=std.cx)
def bare_cx(a: patch[BARE], b: patch[BARE]) -> tuple[patch[BARE], patch[BARE]]:
    return ops.cx(a.data, b.data)


bare_measure_x = gadgets.measure_x(BARE)


def bare_device():
    """One physical qubit per logical qubit, so programs stay small."""
    architecture = cql.devices.QECArchitecture(
        "run_physical_bare_architecture",
        BARE.default_encoding,
        link_roots=(gadgets.prepare_zero(BARE), gadgets.prepare_plus(BARE),
                    gadgets.measure_z(BARE), bare_measure_x, bare_x, bare_h,
                    bare_cx),
    )
    builder = cql.devices.DeviceBuilder("run_physical_bare_device",
                                        source_module=None)
    encoded = builder.qec.bind(builder.logical.add_compute(capacity=2),
                               architecture=architecture)
    builder.physical.bind(encoded,
                          to=builder.physical.add_qubits(
                              2 * BARE.block.size,
                              native_actions=(*actions.clifford_set(), "rpp"),
                              native_instruments=(instruments.MZ,
                                                  instruments.MX,
                                                  instruments.MPP)))
    builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
    return builder.build()


DEVICE = bare_device()


def physical_build(definition):
    """Compiles a program or protocol to P3 through the native projector."""
    if isinstance(definition, cql.protocols.ProtocolDefinition):
        source = cql.compile(definition)
    else:
        source = cql.compile(definition,
                             pipeline=cql.compiler.pipelines.qec(),
                             device=DEVICE)
    build = cql.compile(source,
                        pipeline=cql.compiler.pipelines.physical(),
                        device=DEVICE)
    # Pin the native projector. The Python fallback emits no
    # `phys.call_template`, so the template tests would pass without covering
    # templates.
    assert "fabric-to-phys@0.3" in [
        record.producer
        for record in build.evidence
        if record.kind == "physical_projection_verification"
    ]
    return build


def allocate():
    return cql.ops.allocate_patch(BARE, region="compute")


class P3Fixture:
    """Hand-written, verifier-checked P3 for forms that the other sources do
    not produce. `to_quake()` runs the same pass as `Build.to_quake()`."""

    def __init__(self, text, entry):
        self.text, self.entry = text, entry

    def to_quake(self):
        module = mlir_ir.Module.parse(self.text, context=mlir_ir.Context())
        _qlxRuntime.run_pass_capsule(module,
                                     f'phys-to-quake{{entry="{self.entry}"}}')
        return module


# ---------------------------------------------------------------------------
# Gates, measurements, and results
# ---------------------------------------------------------------------------
# These graphs acquire, prepare, reset, and release carriers, apply gates,
# and measure in the Z and X bases. One graph result comes back as a `bool`,
# even from a one-element list, and several as one `list[bool]` in return
# order.
# ---------------------------------------------------------------------------


@cql.physical(machine)
def x_then_mz():
    # No preparation: a carrier left in |1> by an earlier shot would give
    # False.
    q, = cql.ops.acquire(machine.qubits, count=1)
    return cql.measure(flip(q))


@cql.physical(machine)
def saved_values():
    # A saved bit survives a reset and a reacquired carrier. The measurement
    # of `b` is not returned, and the result order is the return order.
    a, b = fresh(2)
    a, saved = cql.measure(flip(a), destructive=False)
    cql.measure(b)
    later = cql.measure(cql.reset(a))
    reused, = fresh()
    return later, saved, cql.measure(flip(reused))


@cql.physical(machine)
def syndrome():
    # Three parity checks of a 4-bit chain with an error on bit 1.
    data, checks = list(fresh(4)), list(fresh(3))
    data[1] = flip(data[1])
    for i in range(3):
        data[i], checks[i] = cql.ops.apply(actions.CX, data[i], checks[i])
        data[i + 1], checks[i] = cql.ops.apply(actions.CX, data[i + 1],
                                               checks[i])
    bits = []
    for i in range(3):
        checks[i], bit = cql.measure(checks[i], destructive=False)
        bits.append(bit)
    cql.ops.release(tuple(checks) + tuple(data))
    return bits


@cql.physical(machine)
def x_basis():
    # A retained X measurement keeps the X eigenstate: without the basis
    # restore, the second bit would be random, and `h` would not give |0>.
    q, = fresh(state="plus")
    q, first = cql.measure(q, basis=arch.Basis.X, destructive=False)
    q, second = cql.measure(q, basis=arch.Basis.X, destructive=False)
    return first, second, cql.measure(cql.ops.apply(actions.H, q))


def reset_to_plus():
    builder = PhysicalBuilder("reset_to_plus", architecture=machine)
    q, = builder.acquire(machine.qubits, count=1)
    q, = builder.reset((q,), state="plus")
    q, first = builder.measure(q, basis="x", destructive=False)
    return builder.finish(first, builder.measure(q, basis="x"))


@cql.physical(machine)
def one_element_list():
    q, = fresh()
    return [cql.measure(flip(q))]


def alternating_bits(length):
    """One carrier at a time, so the list fits every target."""

    @cql.physical(machine, name=f"alternating_bits_{length}")
    def body():
        bits = []
        for i in range(length):
            q, = fresh()
            bits.append(cql.measure(flip(q) if i % 2 else q))
        return bits

    return lambda: cql.compile(body)


def types_of(results):
    return [[type(bit)
             for bit in result] if isinstance(result, list) else type(result)
            for result in results]


# 40 qubits allocated at once exceed the memory of a state-vector simulator
# such as `qpp-cpu` or `nvidia`, so `list_40` checks that each qubit is freed
# at its release.
@pytest.mark.parametrize("make, expected", [
    pytest.param(physical(x_then_mz), True, id="x_then_mz"),
    pytest.param(physical(saved_values), [False, True, True],
                 id="saved_values"),
    pytest.param(physical(syndrome), [True, True, False], id="syndrome"),
    pytest.param(physical(x_basis), [False, False, False], id="x_basis"),
    pytest.param(reset_to_plus, [False, False], id="reset_to_plus"),
    pytest.param(physical(one_element_list), True, id="one_element_list"),
    pytest.param(alternating_bits(40), [i % 2 == 1 for i in range(40)],
                 id="list_40"),
])
def test_bool_and_list_results(make, expected):
    results = cudaq.run(make(), shots_count=SHOTS)
    assert results == [expected] * SHOTS
    assert types_of(results) == types_of([expected] * SHOTS)


@cql.physical(machine)
def bell():
    a, b = fresh(2)
    a, b = cql.ops.apply(actions.CX, cql.ops.apply(actions.H, a), b)
    return [cql.measure(a), cql.measure(b)]


# Every target runs this program, so a simulator that cannot run the emitted
# Quake fails here.
@pytest.mark.parametrize("target", TARGETS, indirect=True)
def test_bell_pair(target):
    results = cudaq.run(cql.compile(bell), shots_count=RANDOM_SHOTS)
    assert len(results) == RANDOM_SHOTS
    assert set(map(tuple, results)) == {(False, False), (True, True)}


@cudaq.kernel
def ordinary_bool() -> bool:
    q = cudaq.qubit()
    x(q)
    return mz(q)


# ---------------------------------------------------------------------------
# Arguments and classical logic
# ---------------------------------------------------------------------------
# Arguments become the kernel's parameters. `arith` ops, `phys.condition`,
# `phys.xor`, and `phys.all_false` compute Booleans from arguments and
# records.
# ---------------------------------------------------------------------------


@cql.program
def flag_from_arguments(n: int, theta: float, flag: bool) -> bool:
    return flag


@cudaq.kernel
def ordinary_flag_from_arguments(n: int, theta: float, flag: bool) -> bool:
    return flag


@pytest.mark.parametrize("flag", [False, True])
def test_arguments_match_ordinary_run(flag):
    # Ordinary run is the control for the same signature.
    for kernel in (ordinary_flag_from_arguments,
                   physical_build(flag_from_arguments)):
        assert cudaq.run(kernel, 3, 0.5, flag, shots_count=2) == [flag] * 2


# `@cql.program` cannot compute with a traced `int`, so this graph is
# hand-written. Comparing `n` with 2**32 checks that `n` is not truncated
# to 32 bits.
COMPARE_ARGUMENTS = P3Fixture(
    """
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.graph @compare on @m : (index, i1) -> (i1, i1, i1) {
  ^bb0(%n: index, %flag: i1):
    %zero = arith.constant 0 : index
    %big = arith.constant 4294967296 : index
    %negative = arith.cmpi slt, %n, %zero : index
    %large = arith.cmpi sge, %n, %big : index
    %inverted = phys.condition %flag {event_id = "e0", expected = false} : i1 -> i1
    phys.return %negative, %large, %inverted : i1, i1, i1
  }
}""", "compare")


@pytest.mark.parametrize("n, flag, expected",
                         [(-5, True, [True, False, False]),
                          (2**33, False, [False, True, True]),
                          (7, True, [False, False, False])])
def test_arguments_in_computations(n, flag, expected):
    assert cudaq.run(COMPARE_ARGUMENTS, n, flag,
                     shots_count=2) == [expected] * 2


@cql.protocol
def computed_booleans() -> tuple[bool, bool]:
    a = cql.prepare_zero(allocate())
    b = cql.prepare_zero(allocate())
    a, first = cql.mpp(Z(a[0]))
    b, second = cql.mpp(Z(b[0]))
    cql.discard((a, b))
    return cql.xor(first, second), cql.ops.all_false(first, second)


def test_computed_booleans():
    assert cudaq.run(physical_build(computed_booleans),
                     shots_count=SHOTS) == [[False, True]] * SHOTS


# ---------------------------------------------------------------------------
# Gadget calls and product operations
# ---------------------------------------------------------------------------
# `phys.call` is inlined, and `phys.call_template` replays its canonical
# `phys.call`. `flip_after_loop`, under Control flow, replays a call made
# inside a loop. A Pauli product is measured on an ancilla, or rotated with
# a CX ladder around `rz`, which Stim rejects at every angle.
# ---------------------------------------------------------------------------


# Gadget calls compile to `phys.call`, and a repeated call to a
# `phys.call_template` that replays the first on another carrier.
@cql.program
def flip_first() -> tuple[bool, bool]:
    a = cql.prepare_zero()
    b = cql.prepare_zero()
    return cql.measure_z(cql.x(a)), cql.measure_z(b)


# Both measurements run one call body, so their records share a
# `record_id`. Each result must still come from its own measurement.
@cql.program
def same_label() -> tuple[bool, bool]:
    first = cql.measure_z(cql.prepare_zero())
    return first, cql.measure_z(cql.x(cql.prepare_zero()))


@cql.program(name="two-flips")
def two_flips() -> bool:
    return cql.measure_z(cql.x(cql.x(cql.prepare_zero())))


@cql.protocol
def inverted_product() -> tuple[bool, bool]:
    a = cql.prepare_zero(allocate())
    b = cql.prepare_zero(allocate())
    a, b, negated = cql.mpp(-(Z(a[0]) @ Z(b[0])))
    a, b, plain = cql.mpp(Z(a[0]) @ Z(b[0]))
    cql.discard((a, b))
    return negated, plain


# Under `exp(-i angle P / 2)`, Y@Z by pi/2 takes `a` to |+> while `b` is
# |0>, and -Y by pi/2 takes `b` to |->.
@cql.protocol
def signed_rotations() -> tuple[bool, bool]:
    a = cql.prepare_zero(allocate())
    b = cql.prepare_zero(allocate())
    a, b = cql.ops.rotate(Y(a[0]) @ Z(b[0]), angle=math.pi / 2)
    b, = cql.ops.rotate(-Y(b[0]), angle=math.pi / 2)
    return bare_measure_x(a), bare_measure_x(b)


@pytest.mark.parametrize("definition, expected", [
    (flip_first, [True, False]),
    (same_label, [False, True]),
    (two_flips, False),
    (inverted_product, [True, False]),
])
def test_calls_and_products(definition, expected):
    assert cudaq.run(physical_build(definition),
                     shots_count=SHOTS) == [expected] * SHOTS


# `bell` again, through `h` and `cx` gadget calls instead of physical actions.
@cql.program
def bell_pair() -> tuple[bool, bool]:
    a, b = cql.cx(cql.h(cql.prepare_zero()), cql.prepare_zero())
    return cql.measure_z(a), cql.measure_z(b)


# X@X on |00> is random. Z@Z commutes with it and stays +1, and a second
# X@X repeats the first outcome. Both hold only if each product is measured
# as a whole, not qubit by qubit.
@cql.protocol
def product_coherence() -> tuple[bool, bool, bool]:
    a = cql.prepare_zero(allocate())
    b = cql.prepare_zero(allocate())
    a, b, first = cql.mpp(X(a[0]) @ X(b[0]))
    a, b, middle = cql.mpp(Z(a[0]) @ Z(b[0]))
    a, b, last = cql.mpp(X(a[0]) @ X(b[0]))
    cql.discard((a, b))
    return first, middle, last


def test_correlated_results():
    bell = cudaq.run(physical_build(bell_pair), shots_count=RANDOM_SHOTS)
    assert all(a == b for a, b in bell) and {a for a, _ in bell
                                            } == {False, True}
    products = cudaq.run(physical_build(product_coherence),
                         shots_count=RANDOM_SHOTS)
    assert all(not middle and last == first for first, middle, last in products)
    assert {first for first, _, _ in products} == {False, True}


# `h` and `s` prepare |+i>, the +1 eigenstate of Y, so both Y measurements
# read 0. The second reads 0 only if the first undoes its basis change. The
# gadgets on `DEVICE` cannot prepare a Y eigenstate, and a rotation becomes
# `rz`, which Stim rejects, so this graph is hand-written.
Y_PRODUCT = P3Fixture(
    """
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@h, @s], native_instruments = [@mpp]}
  }
  phys.action @h {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"h\\",\\"parameters\\":{}}"}
  phys.action @s {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"s\\",\\"parameters\\":{}}"}
  phys.instrument @mpp {kind = "measure_product", preserves_inputs, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"measure_product\\",\\"parameters\\":{}}", record_schema = "bit", variadic}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @y_product on @m : () -> (!phys.record<@bit>, !phys.record<@bit>) {
    %0 = phys.acquire [@a] {event_id = "acquire"} : !phys.state<@a>
    %1 = phys.apply @h(%0) {event_id = "h"} : (!phys.state<@a>) -> !phys.state<@a>
    %2 = phys.apply @s(%1) {event_id = "s"} : (!phys.state<@a>) -> !phys.state<@a>
    %3:2 = phys.measure_product %2 {event_id = "y0", instrument = @mpp, paulis = ["Y"], record_id = "y0"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %4:2 = phys.measure_product %3#0 {event_id = "y1", instrument = @mpp, paulis = ["Y"], record_id = "y1"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    phys.release %4#0 {event_id = "release"} : !phys.state<@a>
    phys.return %3#1, %4#1 : !phys.record<@bit>, !phys.record<@bit>
  }
}""", "y_product")


def test_y_product_measurements():
    assert cudaq.run(Y_PRODUCT, shots_count=SHOTS) == [[False, False]] * SHOTS


# Three rotations by pi/3 make a pi rotation about X, so the result is
# deterministic although each rotation is not Clifford.
@cql.protocol
def non_clifford_rotations() -> bool:
    a = cql.prepare_zero(allocate())
    for _ in range(3):
        a, = cql.ops.rotate(X(a[0]), angle=math.pi / 3)
    return gadgets.measure_z(BARE)(a)


# Every product rotation becomes `rz`, which Stim rejects at any angle, as it
# does for an ordinary kernel.
@pytest.mark.parametrize("definition, expected", [
    (signed_rotations, [False, True]),
    (non_clifford_rotations, True),
])
@pytest.mark.parametrize(
    "target", [param for param in TARGETS if param.id in ("qpp-cpu", "stim")],
    indirect=True)
def test_product_rotations(target, definition, expected):
    build = physical_build(definition)
    if target[0] == "stim":
        with pytest.raises(RuntimeError, match="Gate not supported by Stim"):
            cudaq.run(build)
        return
    assert cudaq.run(build, shots_count=SHOTS) == [expected] * SHOTS


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------
# `cflow.if` becomes a branch that carries qubits. `cflow.repeat` and
# `cflow.while` become loops that carry qubits, and in `LOOPS` also records.
# The native projector does not emit `cflow.while`, and it rejects a branch
# on another carrier's measurement, so `TELEPORT` and `LOOPS` are
# hand-written P3.
# ---------------------------------------------------------------------------


def keep(q):
    return (q,)


# A random bit is corrected to 0 at run time.
@cql.protocol
def feedback() -> tuple[bool, bool]:
    a = cql.prepare_plus(allocate())
    a, first = cql.mpp(Z(a[0]))
    a, = cql.ops.cond(first,
                      carries=(a,),
                      then=lambda q: (bare_x(q),),
                      else_=keep)
    return first, gadgets.measure_z(BARE)(a)


def flip_with_ancilla(q):
    # The ancilla is acquired and released inside the branch.
    ancilla = bare_x(cql.prepare_zero(allocate()))
    ancilla, _ = cql.mpp(Z(ancilla[0]))
    cql.discard(ancilla)
    return (bare_x(q),)


@cql.protocol
def branch_local_ancilla() -> tuple[bool, bool]:
    a = cql.prepare_plus(allocate())
    a, first = cql.mpp(Z(a[0]))
    a, = cql.ops.cond(first, carries=(a,), then=flip_with_ancilla, else_=keep)
    return first, gadgets.measure_z(BARE)(a)


@pytest.mark.parametrize("definition", [feedback, branch_local_ancilla])
def test_feedback(definition):
    results = cudaq.run(physical_build(definition), shots_count=RANDOM_SHOTS)
    assert all(not corrected for _, corrected in results)
    assert {first for first, _ in results} == {False, True}


# `teleport` sends |0>, |1>, |+>, or |-> (chosen by `flip` and `xbasis`) from
# @a to @c and reads @c in the input's basis.
TELEPORT = P3Fixture(
    """
module {
  phys.machine @m {
    phys.resource_class @q {count = 3 : i64, kind = "qubit", native_actions = [@x, @z, @h, @cx], native_instruments = [@mz]}
  }
  phys.action @x {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"x\\",\\"parameters\\":{}}"}
  phys.action @z {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"z\\",\\"parameters\\":{}}"}
  phys.action @h {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"h\\",\\"parameters\\":{}}"}
  phys.action @cx {arity = 2 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"cx\\",\\"parameters\\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"measure_z\\",\\"parameters\\":{}}", record_schema = "bit"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.resource @c {index = 2 : i64, kind = "qubit", resource_class = @q}
  phys.graph @teleport on @m : (i1, i1, i1) -> !phys.record<@bit> {
  ^bb0(%flip: i1, %xbasis: i1, %corrected: i1):
    %0:3 = phys.acquire [@a, @b, @c] {event_id = "acquire"} : !phys.state<@a>, !phys.state<@b>, !phys.state<@c>
    %a1 = cflow.if %flip event_id = "prep_flip" -> !phys.state<@a> {
      %t = phys.apply @x(%0#0) {event_id = "prep_x"} : (!phys.state<@a>) -> !phys.state<@a>
      cflow.yield %t : !phys.state<@a>
    } else {
      cflow.yield %0#0 : !phys.state<@a>
    }
    %a2 = cflow.if %xbasis event_id = "prep_basis" -> !phys.state<@a> {
      %t = phys.apply @h(%a1) {event_id = "prep_h"} : (!phys.state<@a>) -> !phys.state<@a>
      cflow.yield %t : !phys.state<@a>
    } else {
      cflow.yield %a1 : !phys.state<@a>
    }
    %b1 = phys.apply @h(%0#1) {event_id = "bell_h"} : (!phys.state<@b>) -> !phys.state<@b>
    %bc:2 = phys.apply @cx(%b1, %0#2) {event_id = "bell_cx"} : (!phys.state<@b>, !phys.state<@c>) -> (!phys.state<@b>, !phys.state<@c>)
    %ab:2 = phys.apply @cx(%a2, %bc#0) {event_id = "bsm_cx"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>)
    %a3 = phys.apply @h(%ab#0) {event_id = "bsm_h"} : (!phys.state<@a>) -> !phys.state<@a>
    %ma = phys.measure @mz(%a3) {destructive, event_id = "ma", record_id = "ma"} : (!phys.state<@a>) -> !phys.record<@bit>
    %mb = phys.measure @mz(%ab#1) {destructive, event_id = "mb", record_id = "mb"} : (!phys.state<@b>) -> !phys.record<@bit>
    %x_needed = phys.condition %mb {event_id = "x_needed"} : !phys.record<@bit> -> i1
    %z_needed = phys.condition %ma {event_id = "z_needed"} : !phys.record<@bit> -> i1
    %c1 = cflow.if %corrected event_id = "corrections" -> !phys.state<@c> {
      %cx = cflow.if %x_needed event_id = "x_fix" -> !phys.state<@c> {
        %t = phys.apply @x(%bc#1) {event_id = "fix_x"} : (!phys.state<@c>) -> !phys.state<@c>
        cflow.yield %t : !phys.state<@c>
      } else {
        cflow.yield %bc#1 : !phys.state<@c>
      }
      %cz = cflow.if %z_needed event_id = "z_fix" -> !phys.state<@c> {
        %t = phys.apply @z(%cx) {event_id = "fix_z"} : (!phys.state<@c>) -> !phys.state<@c>
        cflow.yield %t : !phys.state<@c>
      } else {
        cflow.yield %cx : !phys.state<@c>
      }
      cflow.yield %cz : !phys.state<@c>
    } else {
      cflow.yield %bc#1 : !phys.state<@c>
    }
    %c2 = cflow.if %xbasis event_id = "read_basis" -> !phys.state<@c> {
      %t = phys.apply @h(%c1) {event_id = "read_h"} : (!phys.state<@c>) -> !phys.state<@c>
      cflow.yield %t : !phys.state<@c>
    } else {
      cflow.yield %c1 : !phys.state<@c>
    }
    %out = phys.measure @mz(%c2) {destructive, event_id = "out", record_id = "out"} : (!phys.state<@c>) -> !phys.record<@bit>
    phys.return %out : !phys.record<@bit>
  }
}""", "teleport")


def test_teleportation():
    for flip in (False, True):
        for xbasis in (False, True):
            assert cudaq.run(TELEPORT, flip, xbasis, True,
                             shots_count=SHOTS) == [flip] * SHOTS
            # Without corrections the bit is random.
            uncorrected = cudaq.run(TELEPORT,
                                    flip,
                                    xbasis,
                                    False,
                                    shots_count=RANDOM_SHOTS)
            assert set(uncorrected) == {False, True}


@cql.protocol
def counted_loop() -> bool:
    a = cql.prepare_zero(allocate())
    a = cql.ops.repeat(2, carries=(a,), body=bare_x)
    return gadgets.measure_z(BARE)(a)


@cql.protocol
def zero_trip_loop() -> bool:
    a = cql.prepare_zero(allocate())
    a = cql.ops.repeat(0, carries=(a,), body=bare_x)
    return gadgets.measure_z(BARE)(a)


# The last flip replays, as a call template, the call inside the loop body.
@cql.protocol
def flip_after_loop() -> bool:
    a = cql.prepare_zero(allocate())
    a = cql.ops.repeat(3, carries=(a,), body=bare_x)
    return gadgets.measure_z(BARE)(bare_x(a))


@pytest.mark.parametrize("definition, expected", [(counted_loop, False),
                                                  (zero_trip_loop, False),
                                                  (flip_after_loop, False)])
def test_counted_loops(definition, expected):
    assert cudaq.run(physical_build(definition),
                     shots_count=SHOTS) == [expected] * SHOTS


# A record saved before a loop keeps its value, a counted loop measures on
# every iteration, and a dynamic loop measures until it reads 0.
LOOPS = P3Fixture(
    """
module {
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@x, @h], native_instruments = [@mz]}
  }
  phys.action @x {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"x\\",\\"parameters\\":{}}"}
  phys.action @h {arity = 1 : i64, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"h\\",\\"parameters\\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\\"kind\\":\\"builtin\\",\\"name\\":\\"measure_z\\",\\"parameters\\":{}}", record_schema = "bit"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @loops on @m : () -> (!phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>) {
    %0 = phys.acquire [@a] {event_id = "acquire"} : !phys.state<@a>
    %1, %saved = phys.measure @mz(%0) {event_id = "saved", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %2:2 = cflow.repeat 3 iter(%s: !phys.state<@a> = %1, %r: !phys.record<@bit> = %saved) event_id = "repeat" {
      %3 = phys.apply @x(%s) {event_id = "flip"} : (!phys.state<@a>) -> !phys.state<@a>
      %4, %m = phys.measure @mz(%3) {event_id = "each", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
      cflow.yield %4, %m : !phys.state<@a>, !phys.record<@bit>
    }
    %5:2 = cflow.while(%2#0, %2#1) event_id = "until_zero" : (!phys.state<@a>, !phys.record<@bit>) -> (!phys.state<@a>, !phys.record<@bit>) {
    ^bb0(%s: !phys.state<@a>, %r: !phys.record<@bit>):
      %one = phys.condition %r {event_id = "is_one"} : !phys.record<@bit> -> i1
      cflow.while_condition %one, %s, %r : !phys.state<@a>, !phys.record<@bit>
    } do {
    ^bb0(%s: !phys.state<@a>, %r: !phys.record<@bit>):
      %6 = phys.apply @h(%s) {event_id = "spread"} : (!phys.state<@a>) -> !phys.state<@a>
      %7, %m = phys.measure @mz(%6) {event_id = "retry", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
      cflow.yield %7, %m : !phys.state<@a>, !phys.record<@bit>
    }
    phys.release %5#0 {event_id = "release"} : !phys.state<@a>
    phys.return %saved, %2#1, %5#1 : !phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>
  }
}""", "loops")


def test_loops_carry_records():
    assert cudaq.run(LOOPS, shots_count=SHOTS) == [[False, True, False]] * SHOTS


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
# A noise model acts on the gates that `phys-to-quake` emits, as it does on
# an ordinary kernel's gates. `qpp-cpu` ignores noise models, so these tests
# run on `stim`, except one that needs `density-matrix-cpu`.
# ---------------------------------------------------------------------------


@cql.physical(machine)
def first_of_three():
    # Only the first carrier sees `x`, so an `x` channel can change bit 0 only.
    a, b, c = fresh(3)
    return [cql.measure(flip(a)), cql.measure(b), cql.measure(c)]


def bit_flips(probability):
    model = cudaq.NoiseModel()
    model.add_all_qubit_channel("x", cudaq.BitFlipChannel(probability))
    return model


NOISY_TARGETS = [param for param in TARGETS if param.id == "stim"]
NOISE_SHOTS = 64


def flipped_shots(results):
    """How many shots lost the True that `x` produced in bit 0."""
    return sum(not (result[0] if isinstance(result, list) else result)
               for result in results)


@pytest.mark.parametrize("target", NOISY_TARGETS, indirect=True)
def test_noise_acts_on_emitted_gates(target):
    # Each kernel runs clean, then noisy, and repeats both. A clean run after
    # a noisy one checks that no noise model stays set. A bit flip with
    # probability 0.5 changes about half of the 64 shots: the bound is 5.5
    # standard deviations.
    cudaq.set_random_seed(13)
    scalar, listed = cql.compile(x_then_mz), cql.compile(first_of_three)
    expected = range(10, 55)
    for _ in range(2):
        for kernel, clean in ((ordinary_bool, True), (scalar, True),
                              (listed, [True, False, False])):
            assert cudaq.run(kernel,
                             shots_count=NOISE_SHOTS) == [clean] * NOISE_SHOTS
            noisy = cudaq.run(kernel,
                              shots_count=NOISE_SHOTS,
                              noise_model=bit_flips(0.5))
            assert types_of(noisy) == types_of([clean] * NOISE_SHOTS)
            assert flipped_shots(noisy) in expected
            if isinstance(clean, list):
                assert all(result[1:] == [False, False] for result in noisy)
    assert cudaq.run(listed, shots_count=0, noise_model=bit_flips(0.5)) == []
    assert cudaq.run(
        listed, shots_count=2,
        noise_model=cudaq.NoiseModel()) == [[True, False, False]] * 2


# Of the targets that apply noise, only `density-matrix-cpu` takes
# probability 1.
@pytest.mark.parametrize(
    "target",
    [pytest.param(("density-matrix-cpu", {}), id="density-matrix-cpu")],
    indirect=True)
def test_certain_noise_is_deterministic(target):
    for kernel, noisy in ((ordinary_bool, False), (cql.compile(x_then_mz),
                                                   False),
                          (cql.compile(first_of_three), [False, False, False])):
        assert cudaq.run(kernel, shots_count=SHOTS,
                         noise_model=bit_flips(1.0)) == [noisy] * SHOTS
    assert cudaq.run(cql.compile(x_then_mz),
                     shots_count=SHOTS) == [True] * SHOTS


# ---------------------------------------------------------------------------
# Unsupported Builds
# ---------------------------------------------------------------------------
# A Build without results cannot run, but `to_quake()` still converts it. An
# unsupported Build fails with an error that names its result type, its kind,
# or its root symbol.
# ---------------------------------------------------------------------------


def test_to_quake_without_results(target):

    @cql.physical(machine)
    def no_result():
        q, = fresh()
        cql.ops.release((flip(q),))

    no_results = cql.compile(no_result)
    assert ("func.func @__nvqpp__mlirgen__no_result() attributes"
            in str(no_results.to_quake()))


@cql.program
def int_result(n: int) -> int:
    return n


@pytest.mark.parametrize("make, error, message", [
    pytest.param(lambda: physical_build(int_result),
                 RuntimeError,
                 "returns result 0 of type 'index', but `phys-to-quake` "
                 "translates only Boolean results",
                 id="int"),
    pytest.param(physical(two_flips),
                 ValueError,
                 "requires a `Build` of a physical graph, not of a program",
                 id="not_physical"),
    pytest.param(lambda: Build.replay(
        cql.compile(bell).to_mlir().replace("@bell on", '@"b}" on').encode(),
        root=DefinitionHandle("b}", "physical_graph", "p3"),
        profile="p3"),
                 ValueError,
                 "requires a root symbol of letters, digits",
                 id="root_symbol"),
])
def test_unsupported_builds(target, make, error, message):
    with pytest.raises(error, match=message):
        cudaq.run(make())


# ---------------------------------------------------------------------------
# The `to_quake()` interface
# ---------------------------------------------------------------------------
# Each call returns a new module that outlives its Build. A run on another
# target does not change later runs.
# ---------------------------------------------------------------------------


class QuakeSource:
    """An object whose `to_quake()` returns `make()`."""

    def __init__(self, make):
        self.make = make

    def to_quake(self):
        return self.make()


def test_to_quake_inspection_and_lifetime(target):
    build = cql.compile(x_then_mz)
    source = build.to_mlir()
    module = build.to_quake()
    text = str(module)
    assert "func.func @__nvqpp__mlirgen__x_then_mz() -> i1" in text
    assert "quake.optimizationLevel = 0" in text
    assert "phys." not in text and "qlx." not in text
    assert build.to_quake() is not module and build.to_mlir() == source

    # The module outlives its Build, and a replayed Build runs without the
    # callable that defined it.
    @cql.physical(machine)
    def local_flip():
        q, = fresh()
        return cql.measure(flip(q))

    payload = cql.compile(local_flip).serialize()
    del build, local_flip
    gc.collect()
    assert cudaq.run(QuakeSource(lambda: module), shots_count=2) == [True, True]
    assert cudaq.run(Build.replay(payload), shots_count=2) == [True, True]


def test_runs_are_independent(target):
    # Two programs whose graphs share one name.
    @cql.physical(machine, name="shared_name")
    def flipped():
        q, = fresh()
        return cql.measure(flip(q))

    @cql.physical(machine, name="shared_name")
    def unflipped():
        q, = fresh()
        return cql.measure(q)

    first = cql.compile(flipped)
    second = cql.compile(unflipped)

    # A run on another target leaves the next run unaffected.
    name, options = target
    try:
        cudaq.set_target("stim" if name == "qpp-cpu" else "qpp-cpu")
        assert cudaq.run(first, shots_count=2) == [True, True]
    finally:
        cudaq.set_target(name, **options)
    assert cudaq.run(second, shots_count=2) == [False, False]
