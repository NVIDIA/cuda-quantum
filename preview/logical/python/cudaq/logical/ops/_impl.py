# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from typing import Any, Iterable

from ..types.semantic import (
    LogicalState,
    plus,
    zero,
)
from ..algebra.pauli import (
    PauliProduct,
    X,
    Y,
    Z,
)
from ..programs.context import require_trace


def prepare_zero(value=None):
    trace = require_trace("prepare_zero")
    return (trace.prepare("zero") if value is None else trace.prepare_patch(
        value, "zero"))


def prepare_plus(value=None):
    trace = require_trace("prepare_plus")
    return (trace.prepare("plus") if value is None else trace.prepare_patch(
        value, "plus"))


def prepare(value=None, *, state=zero):
    if not isinstance(state, LogicalState):
        raise TypeError("prepare state= must be a cudaq.logical.LogicalState")
    trace = require_trace("prepare")
    return (trace.prepare(state.name) if value is None else trace.prepare_patch(
        value, state.name))


def allocate(count: int, *, state=zero, name: str | None = None):
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise TypeError("allocate count must be a nonnegative Python int")
    if not isinstance(state, LogicalState):
        raise TypeError("allocate state= must be a cudaq.logical.LogicalState")
    return require_trace("allocate").allocate(count,
                                              state=state.name,
                                              name=name)


def apply(action, *values, parameters=None, **options):
    bindings = dict(parameters or {})
    overlap = set(bindings) & set(options)
    if overlap:
        raise TypeError(f"duplicate apply parameter(s): {sorted(overlap)!r}")
    bindings.update(options)
    return require_trace("apply").apply_definition(action, values, bindings)


def measure(value, *, basis="z", destructive=True, record=None):
    return require_trace("measure").measure(value,
                                            basis=basis,
                                            destructive=destructive,
                                            record=record)


def barrier(values=(), *, domains=()):
    trace = require_trace("barrier")
    if not hasattr(trace, "barrier"):
        raise TypeError(
            "cudaq.logical.barrier is unavailable in this authoring trace")
    scalar = not isinstance(values, (tuple, list))
    operands = (values,) if scalar else tuple(values)
    results = trace.barrier(operands, domains=tuple(domains))
    return results[0] if scalar and results else results


def h(q):
    return require_trace("h").apply_standard("h", (q,))[0]


def s(q):
    return require_trace("s").apply_standard("s", (q,))[0]


def sdg(q):
    return require_trace("sdg").apply_standard("sdg", (q,))[0]


def x(q):
    return require_trace("x").apply_standard("x", (q,))[0]


def y(q):
    return require_trace("y").apply_standard("y", (q,))[0]


def z(q):
    return require_trace("z").apply_standard("z", (q,))[0]


def t(q):
    return require_trace("t").apply_standard("t", (q,))[0]


def tdg(q):
    return require_trace("tdg").apply_standard("tdg", (q,))[0]


def reset(q):
    trace = require_trace("reset")
    if hasattr(trace, "reset"):
        scalar = not isinstance(q, (tuple, list))
        values = (q,) if scalar else tuple(q)
        results = trace.reset(values)
        return results[0] if scalar else results
    return trace.apply_standard("reset", (q,))[0]


def cx(control, target, *, schedule=None, pairs=None):
    results = tuple(
        require_trace("cx").apply_standard("cx", (control, target),
                                           schedule=schedule,
                                           pairs=pairs))
    return results[0] if len(results) == 1 else results


def cz(left, right, *, schedule=None, pairs=None):
    results = tuple(
        require_trace("cz").apply_standard("cz", (left, right),
                                           schedule=schedule,
                                           pairs=pairs))
    return results[0] if len(results) == 1 else results


def ccz(a, b, c):
    return tuple(require_trace("ccz").apply_standard("ccz", (a, b, c)))


def rotate(product: PauliProduct, *, angle, precision=None):
    if not isinstance(product, PauliProduct):
        raise TypeError("rotate expects a cudaq.logical.PauliProduct")
    trace = require_trace("rotate")
    if precision is None:
        return tuple(trace.rotate(product, angle=angle))
    return tuple(trace.rotate(product, angle=angle, precision=precision))


def resource_rotate(resource, product: PauliProduct, *, angle):
    """Apply one exact P2 product rotation while consuming a raw resource."""
    if not isinstance(product, PauliProduct):
        raise TypeError("resource_rotate expects a cudaq.logical.PauliProduct")
    trace = require_trace("resource_rotate")
    if not hasattr(trace, "resource_rotate"):
        raise TypeError("qlx.resource_rotate is available only in P2 gadgets")
    return tuple(trace.resource_rotate(resource, product, angle=angle))


def mpp(product: PauliProduct):
    if not isinstance(product, PauliProduct):
        raise TypeError("mpp expects a cudaq.logical.PauliProduct")
    return tuple(require_trace("mpp").mpp(product))


def readout(product: PauliProduct):
    """Destructive Pauli-product readout: one bool, no surviving operands.

    ``qlx.mpp`` is always nondestructive; ``cudaq.logical.readout`` is its destructive
    counterpart and consumes every covered operand, including explicit
    ``cudaq.logical.I`` identity factors.
    """
    if not isinstance(product, PauliProduct):
        raise TypeError("readout expects a cudaq.logical.PauliProduct")
    trace = require_trace("readout")
    if not hasattr(trace, "readout"):
        raise TypeError(
            "cudaq.logical.readout requires a trace with destructive product readout"
        )
    return trace.readout(product)


def rx(q, angle, *, precision=None):
    """``exp(-i angle X / 2)`` on one logical value (spec rotation convention)."""
    return rotate(X(q), angle=angle, precision=precision)[0]


def ry(q, angle, *, precision=None):
    """``exp(-i angle Y / 2)`` on one logical value (spec rotation convention)."""
    return rotate(Y(q), angle=angle, precision=precision)[0]


def rz(q, angle, *, precision=None):
    """``exp(-i angle Z / 2)`` on one logical value (spec rotation convention)."""
    return rotate(Z(q), angle=angle, precision=precision)[0]


def x_if(bit, q):
    """Byproduct-correction sugar: apply X to ``q`` exactly when ``bit`` is set."""
    (result,) = cond(
        bit,
        then=lambda live: (x(live),),
        else_=lambda live: (live,),
        carries=(q,),
    )
    return result


def z_if(bit, q):
    """Byproduct-correction sugar: apply Z to ``q`` exactly when ``bit`` is set."""
    (result,) = cond(
        bit,
        then=lambda live: (z(live),),
        else_=lambda live: (live,),
        carries=(q,),
    )
    return result


def measure_z(q):
    return require_trace("measure_z").measure("z", q)


def measure_x(q):
    return require_trace("measure_x").measure("x", q)


def mz(partition, *, record: str | None = None):
    return require_trace("mz").mz(partition, record=record)


def measure_pauli(partition, *, paulis, record: str | None = None):
    """Measure one explicitly selected physical Pauli product in P2."""

    return require_trace("measure_pauli").measure_pauli(partition,
                                                        paulis=paulis,
                                                        record=record)


def read_syndrome_ancillas(patch, *, record: str | None = None):
    """Read already-entangled ``sx``/``sz`` ancillas in canonical order.

    This operation never inserts preparation or entangling gates. Use
    :func:`extract_syndrome` for the standard explicit CSS round in CUDA-Q Logical.
    """

    return require_trace("read_syndrome_ancillas").read_syndrome_ancillas(
        patch, record=record)


def extract_syndrome(
    patch,
    *,
    record: str | None = None,
    schedule=None,
    cx_schedule=None,
    prime: bool | None = None,
    final_cycle: bool = False,
):
    """Emit the standard CUDA-Q Logical CSS ancilla extraction and return its records.

    The active P2 gadget builder expands this call into reset, H, explicit
    code-derived CX interactions, measurements, and typed syndrome records.
    Backends therefore consume one visible realization instead of inventing
    extraction gates contextually.

    ``cx_schedule`` optionally orders the check CXs as explicit layers -- pass
    the ``(x_layers, z_layers)`` plain data from
    ``code.colored_schedule(...)``: the same stabilizers in a chosen order,
    which fixes the circuit-level hook structure and code distance.

    A :class:`cudaq.logical.BBSyndromeSchedule` selects the BB paper's interleaved
    eight-moment cycle. ``prime=True`` explicitly initializes ``q(Z)`` before
    the first cycle; subsequent adjacent cycles use ``prime=False`` because
    moment 8 initialized ``q(Z)`` for their moment 1. Omitting ``prime`` for
    this cyclic schedule fails closed. A non-terminal result carries an internal
    continuation proof and must be passed immediately to the same code's next
    call with ``prime=False``; no allocation, call, other authored
    operation, or gadget boundary may intervene. Rejected requests preserve
    linear operands and automatic record naming. ``final_cycle=True`` omits the
    otherwise reusable round-8 ``q(Z)`` reset and next-cycle boundary. The
    returned patch is terminal and may only be discarded.

    ``schedule`` may otherwise be a callable
    ``(check_index, canonical_support) -> reordered_support``; a mapping with
    separate ``"x"``/``"z"`` callables or per-check sequences is also
    accepted. Every result must be a permutation of that check's support. It
    controls within-check order without introducing timing operations.
    ``schedule`` and ``cx_schedule`` are mutually exclusive. ``prime`` is
    accepted only for a ``BBSyndromeSchedule``; so is ``final_cycle=True``.
    """

    return require_trace("extract_syndrome").extract_syndrome(
        patch,
        record=record,
        schedule=schedule,
        cx_schedule=cx_schedule,
        prime=prime,
        final_cycle=final_cycle,
    )


def measure_gauges(patch,
                   *,
                   operators,
                   record: str | None = None,
                   phase: str | None = None):
    """Measure an explicit gauge basis and return its raw record bundle."""

    return require_trace("measure_gauges").measure_gauges(patch,
                                                          operators=operators,
                                                          record=record,
                                                          phase=phase)


def transition_epoch(patch, *, to, evidence, logical_map=None):
    """Apply one declared dynamic-code phase transition."""

    return require_trace("transition_epoch").transition_epoch(
        patch, to=to, evidence=evidence, logical_map=logical_map)


def xor(lhs, rhs):
    """Return the runtime parity of two traced Boolean values."""

    return require_trace("xor").xor(lhs, rhs)


def all_zero(bits):
    """Return true exactly when every bit in a measured bundle is zero."""

    return require_trace("all_zero").all_zero(bits)


def parity(*bits):
    """Return the XOR parity of one or more measurement bundles."""

    return require_trace("parity").parity(*bits)


def all_false(*events):
    """Return true exactly when none of the supplied events is set."""

    return require_trace("all_false").all_false(*events)


def permute(patch, permutation):
    return require_trace("permute").permute(patch, permutation)


def idle(values, *, rounds):
    from ..types.values import LogicalRegister

    scalar = not isinstance(values, (tuple, list, LogicalRegister))
    operands = (values,) if scalar else tuple(values)
    results = tuple(require_trace("idle").idle(operands, rounds=rounds))
    return results[0] if scalar else results


def discard(values, *, reason: str | None = None) -> None:
    from ..types.values import LogicalRegister

    operands = (tuple(values) if isinstance(values,
                                            (tuple, list, LogicalRegister)) else
                (values,))
    require_trace("discard").discard(operands, reason=reason)


def request(kind):
    trace = require_trace("request")
    if getattr(trace, "_qlx_authoring_scope", None) != "p2_protocol":
        raise TypeError(
            "cudaq.logical.ops.request is available only inside an @cudaq.logical.protocol body"
        )
    return trace.request(kind)


def request_many(kind, *, count):
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise TypeError("request_many count must be a nonnegative Python int")
    trace = require_trace("request_many")
    if getattr(trace, "_qlx_authoring_scope", None) != "p2_protocol":
        raise TypeError(
            "cudaq.logical.ops.request_many is available only inside an @cudaq.logical.protocol body"
        )
    return tuple(trace.request(kind) for _ in range(count))


def produce(kind, *, region=None, protocol=None):
    trace = require_trace("produce")
    if not hasattr(trace, "produce"):
        raise TypeError(
            "cudaq.logical.produce is available only in a P2 protocol trace")
    return trace.produce(kind, region=region, protocol=protocol)


def transport(resource, *, source, destination, protocol=None):
    trace = require_trace("transport")
    if not hasattr(trace, "transport"):
        raise TypeError(
            "cudaq.logical.transport is available only in a P2 protocol trace")
    return trace.transport(
        resource,
        source=source,
        destination=destination,
        protocol=protocol,
    )


def unpack_resource(resource, *, like, encoding=None, logical_ports=None):
    """Transfer an encoded resource payload into a patch like ``like``.

    This is a P2 ownership operation.  It consumes both owners and returns the
    successor anchor plus a distinct encoded payload patch; it never applies a
    logical action implicitly.
    """
    trace = require_trace("unpack_resource")
    if not hasattr(trace, "unpack_resource"):
        raise TypeError(
            "cudaq.logical.unpack_resource is available only in P2 authoring")
    return trace.unpack_resource(
        resource,
        like=like,
        encoding=encoding,
        logical_ports=logical_ports,
    )


def pack_resource(payload, *, kind):
    """Transfer one encoded patch into a typed linear resource payload."""
    trace = require_trace("pack_resource")
    if not hasattr(trace, "pack_resource"):
        raise TypeError(
            "cudaq.logical.pack_resource is available only in P2 authoring")
    return trace.pack_resource(payload, kind=kind)


def allocate_patch(encoding=None, *, region=None):
    """Allocate an encoded owner, inferring device-bound entry context."""
    trace = require_trace("allocate_patch")
    if not hasattr(trace, "allocate_patch"):
        raise TypeError(
            "cudaq.logical.allocate_patch is available only in P2 gadgets and protocols"
        )
    return trace.allocate_patch(encoding, region=region)


def postselect(predicate, *, expected=False):
    """Require one P2 runtime predicate to equal ``expected``."""
    trace = require_trace("postselect")
    if not hasattr(trace, "postselect"):
        raise TypeError(
            "cudaq.logical.postselect is available only in P2 protocols")
    return trace.postselect(predicate, expected=expected)


def event_test(event):
    return require_trace("event_test").event_test(event)


def event_poll(event):
    return require_trace("event_poll").event_poll(event)


def event_is(status, state):
    return require_trace("event_is").event_is(status, state)


def event_select_ready(*events, policy="priority"):
    return require_trace("event_select_ready").event_select_ready(events,
                                                                  policy=policy)


def event_try_take(event, *, carries=(), ready, pending, failed):
    scalar = not isinstance(carries, (tuple, list))
    values = (carries,) if scalar else tuple(carries)
    results = require_trace("event_try_take").event_try_take(
        event,
        values,
        ready=ready,
        pending=pending,
        failed=failed,
    )
    return results[0] if scalar else results


def event_cancel(event, *, reason=None):
    return require_trace("event_cancel").event_cancel(event, reason=reason)


def event_await(event):
    return require_trace("event_await").event_await(event)


def fence(*effects):
    return require_trace("fence").fence(effects)


def consume(resource, *values, action=None):
    results = require_trace("consume").consume_resource(resource,
                                                        values,
                                                        action=action)
    return results[0] if len(results) == 1 else results


def frame(domain):
    return require_trace("frame").frame(domain)


def frame_update(frame, *, from_, update="outcome"):
    return require_trace("frame_update").frame_update(frame,
                                                      from_,
                                                      update=update)


def frame_s(frame):
    return require_trace("frame_s").frame_transform(frame, "s")


def cond(condition, *, then, else_, carries=()):
    return require_trace("cond").cond(condition, then, else_, tuple(carries))


def if_(condition, *, carries=()):
    """Author an explicit structured branch with named regions and yields."""
    return require_trace("if_").explicit_if(condition, tuple(carries))


def repeat(count, *, carries, body):
    return require_trace("repeat").repeat(count, tuple(carries), body)


def while_(condition, *, carries, body, max_iterations=None):
    """Author a dynamic loop over explicitly carried traced values.

    ``condition`` receives the current carries and returns either one traced
    Boolean or ``(predicate, *forwarded_carries)``. The latter form permits the
    condition region to update frame/event state before entering the body.
    ``max_iterations`` is an optional runtime bound and never requests Python
    unrolling.
    """

    return require_trace("while_").while_loop(
        condition,
        tuple(carries),
        body,
        max_iterations=max_iterations,
    )


def encoding_unpack(parent, *, hierarchy=None, slot_group=None):
    return require_trace("encoding_unpack").encoding_unpack(
        parent, hierarchy=hierarchy, slot_group=slot_group)


def map_children(callee, children):
    return require_trace("map_children").map_children(callee, children)


def encoding_pack(children, *, encoding=None):
    return require_trace("encoding_pack").encoding_pack(children,
                                                        encoding=encoding)


def retry(
    carries,
    *,
    until,
    max_attempts,
    exhaustion=None,
    commit_point=None,
):
    from ..gadgets import RetryExhaustion

    exhaustion = (RetryExhaustion.REPORT_FAILURE
                  if exhaustion is None else exhaustion)
    scalar = not isinstance(carries, (tuple, list))
    values = (carries,) if scalar else tuple(carries)
    results = require_trace("retry").retry(
        values,
        until=until,
        max_attempts=max_attempts,
        exhaustion=exhaustion,
        commit_point=commit_point,
    )
    return results[0] if scalar else results
